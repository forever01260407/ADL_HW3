"""Build the B0 and B1 sample schedules for E24 (runbook 6.2) and check them.

    python code/ops/make_schedule.py --pool TRAIN_POOL --out-dir DIR

300 updates x 16 = 4,800 positions. Part 2 takes 1,627 fixed positions (813 in the first 150
updates, 814 in the second); the other 3,173 are Part 1. B0 puts original Part 1 steps (data_v5,
everything that is not source "focus") on all of them. B1 is identical except at 635 fixed Part 1
positions (317 + 318 over the two halves), where it puts focus steps. Every id is a training
example after encoding (train.prepare drops steps over max_length; nothing is skipped later).
Separate RNGs draw the positions, Part 2 ids, original Part 1 ids and focus ids. Writes
B0_schedule.json, B1_schedule.json and schedule_report.json; exits 1 if a check fails.
"""
import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, "code")

UPDATES, PER_UPDATE, HALF = 300, 16, 150
P2, REPLACE = 1627, 635


def split_positions(rng, candidates, n, half_end):
    first = [p for p in candidates if p < half_end]
    second = [p for p in candidates if p >= half_end]
    k1 = n // 2
    return sorted(rng.sample(first, k1) + rng.sample(second, n - k1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    import model_setup as M
    import train as T
    config = {"train_file": a.pool, "max_examples": None, "seed": 42, "max_length": 8192}
    tok = M.F.tokenizer()
    examples, too_long = T.prepare(config, tok)
    by = collections.defaultdict(list)
    tokens = {}
    for e in examples:
        src = "focus" if e["id"].startswith("focus_") else e["task"]
        by[src].append(e["id"])
        tokens[e["id"]] = (len(e["input_ids"]), sum(label != M.IGNORE for label in e["labels"]))
    total = UPDATES * PER_UPDATE
    half_end = HALF * PER_UPDATE
    positions = list(range(total))
    p2_pos = split_positions(random.Random(4201), positions, P2, half_end)
    p1_pos = [p for p in positions if p not in set(p2_pos)]
    rep_pos = split_positions(random.Random(4202), p1_pos, REPLACE, half_end)
    p2_ids = random.Random(4203).sample(by["part2"], P2)
    p1_ids = random.Random(4204).sample(by["part1"], len(p1_pos))
    focus_ids = random.Random(4205).sample(by["focus"], REPLACE)
    b0 = [None] * total
    for p, i in zip(p2_pos, p2_ids):
        b0[p] = i
    for p, i in zip(p1_pos, p1_ids):
        b0[p] = i
    b1 = list(b0)
    for p, i in zip(rep_pos, focus_ids):
        b1[p] = i
    rep_set, p2_set = set(rep_pos), set(p2_pos)
    checks = {
        "lengths": len(b0) == len(b1) == total and None not in b0,
        "part2_identical_in_order": [b0[p] for p in p2_pos] == [b1[p] for p in p2_pos],
        "non_replaced_identical": all(b0[p] == b1[p] for p in positions if p not in rep_set),
        "focus_only_at_replaced": all((b1[p].startswith("focus_")) == (p in rep_set) for p in positions) and not any(i.startswith("focus_") for i in b0),
        "replaced_are_part1_positions": not (rep_set & p2_set),
        "all_ids_are_examples": all(i in tokens for i in b1 + b0),
        "no_repeat_within_schedule": len(set(b0)) == total and len(set(b1)) == total,
    }

    def stats(s):
        out = {}
        for name, lo, hi in (("first_half", 0, half_end), ("second_half", half_end, total)):
            part = s[lo:hi]
            c = collections.Counter("focus" if i.startswith("focus_") else ("part2" if i in set(p2_ids) else "part1_original") for i in part)
            out[name] = {**c, "input_tokens": sum(tokens[i][0] for i in part), "target_tokens": sum(tokens[i][1] for i in part)}
        return out
    os.makedirs(a.out_dir, exist_ok=True)
    for name, s in (("B0", b0), ("B1", b1)):
        with open(f"{a.out_dir}/{name}_schedule.json", "x") as f:
            json.dump(s, f)
    report = {"pool": a.pool, "pool_sha256": M.file_hash(a.pool), "examples": len(examples), "too_long_dropped": too_long,
              "available": {k: len(v) for k, v in by.items()}, "positions": {"total": total, "part2": P2, "part1": len(p1_pos), "replaced_in_B1": REPLACE},
              "checks": checks, "B0": stats(b0), "B1": stats(b1),
              "B0_sha256": M.file_hash(f"{a.out_dir}/B0_schedule.json"), "B1_sha256": M.file_hash(f"{a.out_dir}/B1_schedule.json")}
    with open(f"{a.out_dir}/schedule_report.json", "x") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({"checks": checks, "available": report["available"], "B1_halves": report["B1"]}, indent=1))
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()

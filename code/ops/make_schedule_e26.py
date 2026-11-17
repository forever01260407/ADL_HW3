"""Build the C0 and C1 sample schedules for E26 and check them (copy of make_schedule_e25.py).

    python code/ops/make_schedule_e26.py --pool TRAIN_POOL --patch PATCH --out-dir DIR [--expect-c0 SHA256]

E26 (user decision): C1 replaces ALL 1,084 Part 2 positions with E26 buyer steps (ids "e26t_"),
drawn uniformly without repeats; positions, seeds and Part 1 are those of E25, so C0 must come out
identical to E25's C0 (checked with --expect-c0). The E25 description follows for the rest.

200 updates x 16 = 3,200 positions (user setting before T0: lr 3e-5, 200 updates). Part 2 takes
1,084 fixed positions (33.875%, 542 in each half of 100 updates); the other 2,116 are Part 1.
C0 puts original data_v5 steps everywhere (Part 1 on Part 1 positions, Part 2 on Part 2 positions).
C1 is identical except at 324 fixed Part 2 positions (162 per half), where it puts E25 patch
steps (ids starting "e25p_"). The patch steps are drawn by fixed quotas over what the step's
visible facts and target action are (check_buyer, from the rendered messages): 130 legal orders,
97 continue/leave on illegal terms or before an offer, 97 keep negotiating on a legal offer above
target (40/30/30). Every id is a training example after encoding (train.prepare drops steps over
max_length; nothing is skipped later). Separate RNGs draw positions and each id list. Writes
C0_schedule.json, C1_schedule.json, schedule_report.json; exits 1 if a check fails.
"""
import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, "code")
sys.path.insert(0, "code/data")

UPDATES, PER_UPDATE, HALF = 200, 16, 100
P2, REPLACE = 1084, 1084
QUOTA = {"legal_order": 130, "illegal_or_no_offer_continue_or_leave": 97, "legal_keep_negotiating": 97}


def split_positions(rng, candidates, n, half_end):
    first = [p for p in candidates if p < half_end]
    second = [p for p in candidates if p >= half_end]
    k1 = n // 2
    return sorted(rng.sample(first, k1) + rng.sample(second, n - k1))


def patch_categories(path):
    import check_buyer_patch_e25 as K  # sets the E25 final wording on check_buyer
    C = K.C
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        skip = set(r.get("no_target") or ())
        for step in C.steps(r):
            if step["index"] in skip:
                continue
            case = C.case_of(step)
            name = step["calls"][0]["function"]["name"] if step["calls"] else "text_only"
            cat = "legal_order" if name == "order" else "legal_keep_negotiating" if case.startswith("legal") else "illegal_or_no_offer_continue_or_leave"
            out[f"{r['id']}#{step['index']}"] = cat
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--patch", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--expect-c0", help="sha256 the C0 schedule must have (E25 C0) so E25's C0 model can serve as the control")
    a = ap.parse_args()
    import model_setup as M
    import train as T
    config = {"train_file": a.pool, "max_examples": None, "seed": 42, "max_length": 8192}
    tok = M.F.tokenizer()
    examples, too_long = T.prepare(config, tok)
    by = collections.defaultdict(list)
    tokens = {}
    for e in examples:
        src = "patch" if e["id"].startswith("e26t_") else e["task"]
        by[src].append(e["id"])
        tokens[e["id"]] = (len(e["input_ids"]), sum(label != M.IGNORE for label in e["labels"]))
    patch_ids = set(by["patch"])
    cats = {i: "all" for i in patch_ids}
    total, half_end = UPDATES * PER_UPDATE, HALF * PER_UPDATE
    positions = list(range(total))
    p2_pos = split_positions(random.Random(4301), positions, P2, half_end)
    p2_set = set(p2_pos)
    p1_pos = [p for p in positions if p not in p2_set]
    rep_pos = split_positions(random.Random(4302), p2_pos, REPLACE, half_end)
    p2_ids = random.Random(4303).sample(by["part2"], P2)
    p1_ids = random.Random(4304).sample(by["part1"], len(p1_pos))
    rng_patch = random.Random(4305)
    chosen = rng_patch.sample(sorted(patch_ids), REPLACE)
    c0 = [None] * total
    for p, i in zip(p2_pos, p2_ids):
        c0[p] = i
    for p, i in zip(p1_pos, p1_ids):
        c0[p] = i
    c1 = list(c0)
    for p, i in zip(rep_pos, chosen):
        c1[p] = i
    rep_set = set(rep_pos)
    checks = {
        "lengths": len(c0) == len(c1) == total and None not in c0,
        "part1_identical_in_order": [c0[p] for p in p1_pos] == [c1[p] for p in p1_pos],
        "non_replaced_identical": all(c0[p] == c1[p] for p in positions if p not in rep_set),
        "patch_only_at_replaced": all((c1[p].startswith("e26t_")) == (p in rep_set) for p in positions) and not any(i.startswith("e26t_") for i in c0),
        "replaced_are_part2_positions": rep_set <= p2_set,
        "replaced_per_half": [sum(p < half_end for p in rep_pos), sum(p >= half_end for p in rep_pos)] == [REPLACE // 2, REPLACE - REPLACE // 2],
        "part2_per_half": [sum(p < half_end for p in p2_pos), sum(p >= half_end for p in p2_pos)] == [P2 // 2, P2 - P2 // 2],
        "all_ids_are_examples": all(i in tokens for i in c1 + c0),
        "no_repeat_within_schedule": len(set(c0)) == total and len(set(c1)) == total,
        "patch_steps_all_encoded": len(patch_ids) == sum(1 for l in open(a.patch) for m in json.loads(l)["messages"] if m["role"] == "assistant"),
    }

    def stats(s):
        out = {}
        for name, lo, hi in (("first_half", 0, half_end), ("second_half", half_end, total)):
            part = s[lo:hi]
            c = collections.Counter("patch" if i.startswith("e26t_") else ("part2" if i in set(p2_ids) else "part1") for i in part)
            out[name] = {**c, "input_tokens": sum(tokens[i][0] for i in part), "target_tokens": sum(tokens[i][1] for i in part)}
        return out
    os.makedirs(a.out_dir, exist_ok=True)
    for name, s in (("C0", c0), ("C1", c1)):
        with open(f"{a.out_dir}/{name}_schedule.json", "x") as f:
            json.dump(s, f)
    report = {"pool": a.pool, "pool_sha256": M.file_hash(a.pool), "patch": a.patch, "patch_sha256": M.file_hash(a.patch), "examples": len(examples),
              "too_long_dropped": too_long, "available": {k: len(v) for k, v in by.items()},
              "patch_steps_available": len(patch_ids),
              "positions": {"total": total, "part2": P2, "part1": len(p1_pos), "replaced_in_C1": REPLACE},
              "checks": checks, "C0": stats(c0), "C1": stats(c1),
              "C0_sha256": M.file_hash(f"{a.out_dir}/C0_schedule.json"), "C1_sha256": M.file_hash(f"{a.out_dir}/C1_schedule.json")}
    with open(f"{a.out_dir}/schedule_report.json", "x") as f:
        json.dump(report, f, indent=1)
    if a.expect_c0:
        checks["C0_identical_to_E25"] = report["C0_sha256"] == a.expect_c0
        report["checks"] = checks
        json.dump(report, open(f"{a.out_dir}/schedule_report.json", "w"), indent=1)
    print(json.dumps({"checks": checks, "available": report["available"], "too_long": too_long, "patch_steps": len(patch_ids), "C1": report["C1"]}, indent=1))
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()

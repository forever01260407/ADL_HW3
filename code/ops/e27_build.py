"""E27 training pool and schedule (user decision 2026-11-09 ~16:40: aim Part 1 at 70%).

    python code/ops/e27_build.py pool --v5 TRAIN_V5 --verify E27_VERIFY [--buyer E26_BUYER] --out POOL
    python code/ops/e27_build.py schedule --pool POOL --updates 400 --out-dir DIR

pool:
- Part 1 records (data_v5 Part 1 and the E27 verify chains): the official first-step rule counts a
  non-empty text reply as right at every step whose previous message is a user message. For each
  such step whose target is a tool call, one extra record holds the same history with a short
  confirmation text as the target (the original call stays in the history of later steps, as in
  the teacher-forced test); the original step is moved to no_target. Steps after a user message
  whose target is text are kept as they are. Part 2 is never changed this way (a buyer must order
  when the offer meets the acceptance rule).
- Part 2: the E26 buyer records when --buyer is given (E26 selected), else data_v5 Part 2.
schedule: updates x 16 positions; Part 2 share 0.339; Part 1 positions: 35% E27 verify steps, 65%
other Part 1 steps (data_v5, with the first-step text targets). No repeats; seed 4401.
"""
import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, "code")
sys.path.insert(0, "code/data")

CONFIRM = ("Sure, I'll {fn}{args} now.", "Got it. Let me {fn}{args} first.", "Thanks! I'll start by running {fn}{args}.", "Okay, one moment while I {fn}{args}.")


def confirm_text(rng, call):
    fn = call["function"]["name"].replace("_", " ").replace(".", " ")
    args = call["function"].get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            args = {}
    parts = [f"{k.replace('_', ' ')} {json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v}" for k, v in list(args.items())[:3]]
    s = (" for " + ", ".join(parts)) if parts else ""
    if len(s) > 140:
        s = s[:137] + "..."
    return rng.choice(CONFIRM).format(fn=fn, args=s)


def pool(a):
    rng = random.Random(4400)
    out, stats = [], collections.Counter()

    def part1(r):
        skip = set(r.get("no_target") or ())
        msgs = r["messages"]
        moved = []
        for i, m in enumerate(msgs):
            if i and m["role"] == "assistant" and i not in skip and msgs[i - 1]["role"] == "user" and m.get("tool_calls"):
                extra = dict(r)
                extra["id"] = f"{r['id']}~fs{i}"
                extra["messages"] = msgs[:i] + [{"role": "assistant", "content": confirm_text(rng, m["tool_calls"][0])}]
                extra["no_target"] = [j for j in range(i) if msgs[j]["role"] == "assistant"]
                extra["first_step_text"] = True
                out.append(extra)
                moved.append(i)
                stats["first_step_text_targets"] += 1
        r = dict(r)
        r["no_target"] = sorted(skip | set(moved))
        out.append(r)
        stats["part1_records"] += 1

    for line in open(a.v5, encoding="utf-8"):
        r = json.loads(line)
        if r.get("task") == "part1":
            part1(r)
        elif not a.buyer:
            out.append(r)
            stats["part2_v5"] += 1
    for line in open(a.verify, encoding="utf-8"):
        part1(json.loads(line))
    if a.buyer:
        for line in open(a.buyer, encoding="utf-8"):
            out.append(json.loads(line))
            stats["part2_e26"] += 1
    with open(a.out, "x", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(dict(stats)), "->", a.out)


def schedule(a):
    import model_setup as M
    import train as T
    tok = M.F.tokenizer()
    examples, too_long = T.prepare({"train_file": a.pool, "max_examples": None, "seed": 42, "max_length": 8192}, tok)
    by = collections.defaultdict(list)
    for e in examples:
        src = "part2" if e["task"] == "part2" else ("verify" if e["id"].startswith("e27t_") else "part1")
        by[src].append(e["id"])
    total = a.updates * 16
    p2 = round(total * 0.339)
    n_verify = round((total - p2) * 0.35)
    n_p1 = total - p2 - n_verify
    rng = random.Random(4401)
    ids = rng.sample(by["part2"], p2) + rng.sample(by["verify"], n_verify) + rng.sample(by["part1"], n_p1)
    rng.shuffle(ids)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(f"{a.out_dir}/E27_schedule.json", "x") as f:
        json.dump(ids, f)
    fs = sum(1 for i in ids if "~fs" in i)
    rep = {"pool": a.pool, "pool_sha256": M.file_hash(a.pool), "examples": len(examples), "too_long_dropped": too_long, "available": {k: len(v) for k, v in by.items()},
           "positions": {"total": total, "part2": p2, "verify": n_verify, "part1_other": n_p1, "first_step_text_targets": fs},
           "checks": {"no_repeat": len(set(ids)) == total, "length": len(ids) == total}, "schedule_sha256": M.file_hash(f"{a.out_dir}/E27_schedule.json")}
    json.dump(rep, open(f"{a.out_dir}/schedule_report.json", "x"), indent=1)
    print(json.dumps(rep, indent=1))
    sys.exit(0 if all(rep["checks"].values()) else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pool")
    p.add_argument("--v5", required=True)
    p.add_argument("--verify", required=True)
    p.add_argument("--buyer")
    p.add_argument("--out", required=True)
    s = sub.add_parser("schedule")
    s.add_argument("--pool", required=True)
    s.add_argument("--updates", type=int, required=True)
    s.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    pool(a) if a.cmd == "pool" else schedule(a)


if __name__ == "__main__":
    main()

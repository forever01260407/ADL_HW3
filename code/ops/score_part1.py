"""Per-step Part 1 correctness on our own sets, and paired family-bootstrap comparisons (runbook 10.3 C).

    python code/ops/score_part1.py score --steps STEPS --records CONVERSATIONS --pred PRED --out OUT.json
    python code/ops/score_part1.py pair --base A.json --new B.json [--seed 42 --B 2000] [--out CMP.json]

`score`: each step is read with the official lenient reader and compared with the official call
comparison (as code/eval_own.py does). A call step is right when the first predicted call equals
the gold call; a text step is right when the reply is valid, makes no call and has text (the
wording itself is not judged). must-call = a gold call step whose previous message is a tool
result (not the first call after a user message). Writes per-step results and totals overall,
for must-call, non-call, call-after-user, and per family.
`pair`: difference new - base in must-call right / must-call steps, with a paired bootstrap that
resamples families (both models get the same families and steps), 2,000 draws, seed 42; the 2.5
and 97.5 percentiles describe the interval. Also the non-call difference.
"""
import argparse
import collections
import json
import random
import sys

sys.path.insert(0, "code")
sys.path.insert(0, "code/data")
sys.path.insert(0, "hw3")

from eval import format as F  # noqa: E402
import eval_own as E  # noqa: E402


def score(a):
    rows = list(F.iter_jsonl(a.steps))
    preds, dups = F.load_predictions([a.pred])
    per = {}
    totals = collections.defaultdict(collections.Counter)
    for row in rows:
        p = preds.get(row["id"])
        reply = F.read_reply(p, "lenient")
        usable = reply is not None and not reply["invalid"]
        calls = reply["calls"] if usable else []
        gold = row["gold"]
        if gold["calls"]:
            right = usable and E.same_calls(gold["calls"], calls)
            group = "must_call" if row["after"] == "tool" else "call_after_user"
        else:
            right = usable and not calls and bool(reply["text"].strip())
            group = "non_call"
        per[row["id"]] = {"group": group, "right": bool(right), "conversation": row["id"].rsplit("#", 1)[0], "missing": p is None}
        for g in ("all", group):
            totals[g]["steps"] += 1
            totals[g]["right"] += bool(right)
    with open(a.records, encoding="utf-8") as f:
        fam = {json.loads(l)["id"]: json.loads(l)["family"] for l in f}
    for sid, r in per.items():
        r["family"] = fam[r["conversation"]]
        totals[f"family {r['family']} {r['group']}"]["steps"] += 1
        totals[f"family {r['family']} {r['group']}"]["right"] += r["right"]
    out = {"steps": a.steps, "pred": a.pred, "missing": sum(r["missing"] for r in per.values()), "duplicates": dups,
           "totals": {k: {**v, "rate": v["right"] / v["steps"]} for k, v in sorted(totals.items())}, "per_step": per}
    with open(a.out, "x", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    t = out["totals"]
    print(json.dumps({g: f"{t[g]['right']}/{t[g]['steps']} = {100 * t[g]['rate']:.1f}%" for g in ("all", "must_call", "non_call", "call_after_user") if g in t}), "missing", out["missing"])


def pair(a):
    base, new = json.load(open(a.base))["per_step"], json.load(open(a.new))["per_step"]
    if set(base) != set(new):
        raise SystemExit("the two results cover different steps")
    fams = collections.defaultdict(list)
    for sid, r in base.items():
        fams[r["family"]].append(sid)
    names = sorted(fams)

    def diff(group, chosen):
        sids = [s for f in chosen for s in fams[f] if base[s]["group"] == group]
        if not sids:
            return None
        return (sum(new[s]["right"] for s in sids) - sum(base[s]["right"] for s in sids)) / len(sids)
    rng = random.Random(a.seed)
    out = {}
    for group in ("must_call", "non_call"):
        point = diff(group, names)
        draws = sorted(d for d in (diff(group, [rng.choice(names) for _ in names]) for _ in range(a.B)) if d is not None)
        n = sum(1 for s in base if base[s]["group"] == group)
        out[group] = {"steps": n, "base_right": sum(base[s]["right"] for s in base if base[s]["group"] == group),
                      "new_right": sum(new[s]["right"] for s in new if new[s]["group"] == group), "diff": point,
                      "ci95": [draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]] if draws else None, "families": len(names)}
    print(json.dumps(out))
    if a.out:
        json.dump(out, open(a.out, "x"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--steps", required=True)
    s.add_argument("--records", required=True, help="the conversations file, for the family of each step")
    s.add_argument("--pred", required=True)
    s.add_argument("--out", required=True)
    p = sub.add_parser("pair")
    p.add_argument("--base", required=True)
    p.add_argument("--new", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--B", type=int, default=2000)
    p.add_argument("--out")
    a = ap.parse_args()
    score(a) if a.cmd == "score" else pair(a)


if __name__ == "__main__":
    main()

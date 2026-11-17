"""Part 1 score on our own step files under the official rules (weighted, with the first-step rule).

    python code/ops/e27_score_p1.py --steps STEPS --pred PRED [--out OUT.json]

STEPS: code/eval_own.py export output (fields: messages = history, gold = {calls, text}, after).
Rules, as in hw3/eval/score_tools.py (README R:159-163):
  previous message is a user message:
     gold call -> right if the first call matches the gold call (defaults not available here: exact
                  arguments) OR the reply is non-empty text without calls (first-step rule)
     gold text -> right if the reply has no call
  previous message is a tool result:
     gold call -> must-call, weight 10, right if the first call matches
     gold text -> right if the reply has no call
  unparsable reply / length cap / missing prediction -> wrong.
Reports the weighted accuracy and each group (n, right, weight).
"""
import argparse
import collections
import json
import sys

sys.path.insert(0, "hw3")
sys.path.insert(0, "code")
from eval import format as F  # noqa: E402
from eval_own import same_calls  # noqa: E402


def score(steps, preds):
    g = collections.defaultdict(collections.Counter)
    for row in steps:
        if row.get("task") != "part1":
            continue
        pred = preds.get(row["id"])
        rep = F.read_reply(pred, "lenient")
        ok_reply = rep is not None and not rep["invalid"] and bool(pred) and pred.get("finish_reason") == "stop"
        calls = rep["calls"] if ok_reply else []
        gold = row["gold"]["calls"]
        after = row["messages"][-1]["role"]
        if gold and after == "tool":
            key, w, right = "must_call after tool", 10, ok_reply and same_calls(gold, calls)
        elif gold:
            key, w = "call after user", 1
            right = ok_reply and (same_calls(gold, calls) or (not calls and bool(rep["text"].strip())))
        else:
            key, w, right = f"text after {after}", 1, ok_reply and not calls
        c = g[key]
        c["n"] += 1
        c["right"] += bool(right)
        c["weight"] += w
        c["weighted_right"] += w * bool(right)
        if gold and after == "user" and ok_reply and not calls and rep["text"].strip():
            c["credited_text"] += 1
    W = sum(c["weight"] for c in g.values())
    R = sum(c["weighted_right"] for c in g.values())
    return {"weighted": 100 * R / W if W else None, "groups": {k: dict(v) for k, v in g.items()},
            "must_call": 100 * g["must_call after tool"]["right"] / g["must_call after tool"]["n"] if g["must_call after tool"]["n"] else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    preds, _ = F.load_predictions([a.pred])
    out = score(list(F.iter_jsonl(a.steps)), preds)
    print(json.dumps({"weighted": out["weighted"], "must_call": out["must_call"]}), {k: (v["right"], v["n"]) for k, v in out["groups"].items()})
    if a.out:
        json.dump(out, open(a.out, "x"), indent=1)


if __name__ == "__main__":
    main()

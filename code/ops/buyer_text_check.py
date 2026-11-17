"""Rule screen of buyer reply text on 40 fixed buyer_select steps (runbook 10.3 B, text check).

    python code/ops/buyer_text_check.py pick --steps STEPS --out IDS.json        (once, before training)
    python code/ops/buyer_text_check.py check --steps STEPS --ids IDS.json --pred PRED --out HITS.json

Rules (hits are candidates; each must be read before it counts as a defect):
  leak        the reply names the budget or the target amount, or says "my budget"/"my target"
  over_budget a dollar amount in the reply is above the budget
  fake_done   claims an order/purchase/payment happened while the reply makes no order call
  wrong_item  names a product id that is not acceptable
  role        speaks as an AI/assistant/seller, refuses, or reports to a user
The ids are drawn once with seed 42 from steps that have reply text in the gold history, spread
over the six flows. This is a screen, not the official response judge.
"""
import argparse
import json
import random
import re
import sys

sys.path.insert(0, "hw3")
from eval import format as F  # noqa: E402


def situation(messages):
    text = messages[0]["content"]
    body = json.loads(text.split("YOUR SITUATION (private to you, fixed for this conversation):\n", 1)[1])
    return body["private_profile"]["budget"], body["private_profile"]["target_price"], body["acceptable_product_ids"], [c["product_id"] for c in body["catalog"]]


def pick(a):
    rows = list(F.iter_jsonl(a.steps))
    rng = random.Random(42)
    by_flow = {}
    for r in rows:
        by_flow.setdefault(r["id"].split("_")[2], []).append(r["id"])
    ids = []
    flows = sorted(by_flow)
    while len(ids) < 40:
        for f in flows:
            pool = [i for i in by_flow[f] if i not in ids]
            if pool and len(ids) < 40:
                ids.append(rng.choice(pool))
    json.dump(sorted(ids), open(a.out, "x"), indent=1)
    print(len(ids), "ids over", len(flows), "flows")


def check(a):
    rows = {r["id"]: r for r in F.iter_jsonl(a.steps)}
    ids = json.load(open(a.ids))
    preds, _ = F.load_predictions([a.pred])
    hits = []
    for i in ids:
        r = rows[i]
        budget, target, ok_ids, catalog = situation(r["messages"])
        reply = F.read_reply(preds.get(i), "lenient", ("order", "search_catalog", "end_session", "wait"))
        if reply is None or reply["invalid"]:
            hits.append({"id": i, "rule": "invalid_reply", "text": ""})
            continue
        text = reply["text"]
        ordered = any(c["name"] == "order" for c in reply["calls"][:1])
        amounts = [float(x) for x in re.findall(r"\$(\d+(?:\.\d{1,2})?)", text)]
        found = []
        if any(abs(x - budget) < 0.005 or abs(x - target) < 0.005 for x in amounts) or re.search(r"\bmy (budget|target)\b", text, re.I):
            found.append("leak")
        if any(x > budget + 1e-9 for x in amounts):
            found.append("over_budget")
        if not ordered and re.search(r"\b(I('ve| have) (ordered|purchased|paid|placed)|order (is )?(placed|confirmed)|purchase (is )?complete)\b", text, re.I):
            found.append("fake_done")
        if any(p in text for p in catalog if p not in ok_ids):
            found.append("wrong_item")
        if re.search(r"\b(as an (ai|assistant)|I am (the|a) seller|I can(no|')t (help|assist)|language model)\b", text, re.I):
            found.append("role")
        for rule in found:
            hits.append({"id": i, "rule": rule, "text": text[:300]})
    json.dump({"ids": len(ids), "hits": hits}, open(a.out, "x"), indent=1)
    print(len(hits), "rule hits", sorted({h["rule"] for h in hits}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pick")
    p.add_argument("--steps", required=True)
    p.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("--steps", required=True)
    c.add_argument("--ids", required=True)
    c.add_argument("--pred", required=True)
    c.add_argument("--out", required=True)
    a = ap.parse_args()
    pick(a) if a.cmd == "pick" else check(a)


if __name__ == "__main__":
    main()

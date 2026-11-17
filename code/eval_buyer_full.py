"""Whole-negotiation buyer metrics on our own evaluation sets, through the official scorer (runbook 9.2).

    python code/eval_buyer_full.py score --records FILE --steps STEPS --pred PRED --out OUT.json
    python code/eval_buyer_full.py compare --base E23.json --new B1.json [--out CMP.json]

STEPS is `code/eval_own.py export --data FILE`; PRED is the official generate output on STEPS.
`score` builds official-format rows (negotiation, msg_idx, gold_call, gold_price = the flow's
reference price, gold_has_text, step_meta rebuilt from the visible messages by check_buyer)
and runs hw3/eval/score_negotiation.run with the lenient reader, so order accuracy, deal volume
and deal price are the official definitions. It adds, per negotiation: whether a legal order was
possible at any step, the first valid order's step and price ratio, invalid / capped replies,
orders over budget, on the wrong product or quantity, or at an outdated price. Response quality
comes only from the official judge (--judgments, E25); without it, it is reported as not measured,
never as a number. Also reported: the share of steps whose reference has text where the model
replied with text (text coverage), and invalid orders per negotiation.
`compare` lists negotiations gained and lost, and the mean price ratio on the deals both have.
"""
import argparse
import collections
import json
import sys
from decimal import Decimal

sys.path.insert(0, "code/data")
sys.path.insert(0, "hw3")

from eval import format as F  # noqa: E402
from eval import judge as SJ  # noqa: E402
from eval import score_negotiation as N  # noqa: E402


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def order_fault(meta, args):
    if args.get("product_id") not in meta["acceptable_product_ids"]:
        return "wrong_product"
    vo = meta.get("valid_order")
    if vo is None:
        return "no_offer"
    try:
        price, qty = Decimal(str(args.get("agree_price"))), int(args.get("qty"))
    except Exception:
        return "unreadable_terms"
    if price * qty > Decimal(str(meta["budget"])):
        return "over_budget"
    if qty != int(vo["qty"]):
        return "wrong_qty"
    if price != Decimal(vo["agree_price"]):
        return "outdated_or_wrong_price"
    return None


def score(a):
    records = {r["id"]: r for r in read(a.records)}
    steps = read(a.steps)
    preds, dups = F.load_predictions(a.pred)
    rows = []
    for s in steps:
        conv, index = s["id"].rsplit("#", 1)
        ref = records[conv]["reference_price"]
        calls = s["gold"]["calls"]
        rows.append({"id": s["id"], "negotiation": conv, "msg_idx": int(index), "kind": "text+call" if s["gold"]["text"] and calls else "call" if calls else "text",
                     "gold_call": calls[0]["name"] if calls else None, "gold_price": ref / 100 if ref else None, "gold_has_text": bool(s["gold"]["text"]),
                     "step_meta": s["step_meta"], "case": s["case"]})
    ids = {r["id"] for r in rows}
    missing, extra = sorted(ids - set(preds)), sorted(set(preds) - ids)
    judge = None
    if a.judgments:  # official buyer-response judge (hw3/eval/judge.py); every text reply must be judged
        judge = (SJ.SayRender(SJ.Template(None), {s["id"]: s for s in steps}), SJ.load_judgments(a.judgments))
    metrics, replies, n_missing = N.run(rows, preds, "lenient", judge=judge, boot=1000)
    gold_text = [r for r in rows if r["gold_has_text"]]
    covered = sum(1 for r in gold_text if replies.get(r["id"]) and not replies[r["id"]]["invalid"] and replies[r["id"]]["text"].strip())
    per = {}
    faults = collections.Counter()
    shapes = collections.Counter()
    for r in sorted(rows, key=lambda r: (r["negotiation"], r["msg_idx"])):
        neg = per.setdefault(r["negotiation"], {"flow": records[r["negotiation"]]["flow"], "opportunity": False, "deal": False,
                                                "first_valid_step": None, "price_ratio": None, "orders": 0, "valid_orders": 0})
        neg["opportunity"] |= r["case"].startswith("legal")
        reply = replies.get(r["id"])
        shapes["missing" if reply is None else "capped" if reply.get("capped") else N.shape_of(reply)] += 1
        call = N.effective_call(reply)
        if call and call["name"] == "order":
            args = call["arguments"] if isinstance(call["arguments"], dict) else {}
            neg["orders"] += 1
            ok = bool(N.order_ok_hw(r["step_meta"], args, require_citation=False))
            fault = None if ok else (order_fault(r["step_meta"], args) or "other")
            faults[fault or "valid"] += 1
            neg["invalid_orders"] = neg.get("invalid_orders", 0) + (not ok)
            if ok:
                neg["valid_orders"] += 1
                if not neg["deal"]:
                    neg["deal"], neg["first_valid_step"] = True, r["msg_idx"]
                    neg["price_ratio"] = float(r["step_meta"]["valid_order"]["agree_price"]) / r["gold_price"]
    flows = collections.defaultdict(lambda: collections.Counter())
    for neg in per.values():
        flows[neg["flow"]]["negotiations"] += 1
        flows[neg["flow"]]["deals"] += neg["deal"]
        flows[neg["flow"]]["with_opportunity"] += neg["opportunity"]
    opp = [n for n in per.values() if n["opportunity"]]
    out = {"official_metrics": N.clean(metrics) if hasattr(N, "clean") else metrics,
           "response": "official judge" if judge else "not measured (no judge)", "judgments": a.judgments,
           "text_coverage_gold_text_steps": covered / len(gold_text) if gold_text else None, "n_gold_text_steps": len(gold_text),
           "invalid_orders_per_negotiation": sum(n.get("invalid_orders", 0) for n in per.values()) / len(per) if per else None,
           "n_missing_predictions": n_missing, "missing_ids": missing[:20], "extra_ids": len(extra), "duplicate_predictions": dups,
           "complete": not missing and not extra and not dups,
           "fill_rate_with_opportunity": sum(n["deal"] for n in opp) / len(opp) if opp else None, "n_with_opportunity": len(opp),
           "order_outcomes": dict(faults), "reply_shapes": dict(shapes), "by_flow": {k: dict(v) for k, v in sorted(flows.items())}, "negotiations": per}
    with open(a.out, "x", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    m = out["official_metrics"]
    print(json.dumps({k: m.get(k) for k in ("n_negotiations", "n_model_orders", "order_accuracy", "fill_rate", "relative_price", "n_relative_price")}),
          "complete", out["complete"], "outcomes", dict(faults))
    sys.exit(0 if out["complete"] else 1)


def compare(a):
    base, new = json.load(open(a.base))["negotiations"], json.load(open(a.new))["negotiations"]
    both = [k for k in base if base[k]["deal"] and new.get(k, {}).get("deal")]
    out = {"gained": sorted(k for k in new if new[k]["deal"] and not base.get(k, {}).get("deal")),
           "lost": sorted(k for k in base if base[k]["deal"] and not new.get(k, {}).get("deal")),
           "common_deals": len(both),
           "common_price_base": sum(base[k]["price_ratio"] for k in both) / len(both) if both else None,
           "common_price_new": sum(new[k]["price_ratio"] for k in both) / len(both) if both else None}
    out["common_price_change"] = (out["common_price_new"] - out["common_price_base"]) if both else None
    out["n_gained"], out["n_lost"] = len(out["gained"]), len(out["lost"])
    print(json.dumps({k: out[k] for k in ("n_gained", "n_lost", "common_deals", "common_price_base", "common_price_new", "common_price_change")}))
    if a.out:
        json.dump(out, open(a.out, "x"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--records", required=True)
    s.add_argument("--steps", required=True)
    s.add_argument("--pred", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--judgments", help="judgments file from `hw3/eval.py judge run` on requests built from STEPS (E25)")
    c = sub.add_parser("compare")
    c.add_argument("--base", required=True)
    c.add_argument("--new", required=True)
    c.add_argument("--out")
    a = ap.parse_args()
    score(a) if a.cmd == "score" else compare(a)


if __name__ == "__main__":
    main()

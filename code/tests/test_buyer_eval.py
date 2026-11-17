"""Acceptance tests for code/eval_buyer_full.py with hand-built predictions (runbook 9.3).

    python code/tests/test_buyer_eval.py --records FILE --steps STEPS --work DIR --out JSON
"""
import argparse
import json
import os
import subprocess
import sys
from decimal import Decimal


def read(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def pred(step_id, name=None, args=None, text=""):
    return {"id": step_id, "tool_calls": [{"name": name, "arguments": args or {}}] if name else [], "text": text}


def score(a, name, preds):
    p, out = f"{a.work}/{name}.pred.jsonl", f"{a.work}/{name}.json"
    write(p, preds)
    if os.path.exists(out):
        os.remove(out)
    r = subprocess.run([sys.executable, "code/eval_buyer_full.py", "score", "--records", a.records, "--steps", a.steps, "--pred", p, "--out", out],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return r.returncode, json.load(open(out)) if os.path.exists(out) else {"error": r.stdout[-1500:]}


def legal(meta):
    vo = meta.get("valid_order")
    return vo and vo["product_id"] in meta["acceptable_product_ids"] and int(vo["qty"]) == 1 and Decimal(vo["agree_price"]) <= Decimal(str(meta["budget"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--steps", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    steps = read(a.steps)
    records = {r["id"]: r for r in read(a.records)}
    with_ref = {k for k, r in records.items() if r["reference_price"]}
    results = {}

    # 1. gold echo: every reference order valid, volume = negotiations with a reference order, price 1.0
    code, m = score(a, "echo", [pred(s["id"], *(s["gold"]["calls"][0]["name"], s["gold"]["calls"][0]["arguments"]) if s["gold"]["calls"] else (None, None), text=s["gold"]["text"]) for s in steps])
    om = m["official_metrics"]
    results["gold_echo"] = {"passed": code == 0 and om["order_accuracy"] == 1.0 and abs(om["fill_rate"] - len(with_ref) / len(records)) < 1e-9 and abs(om["relative_price"] - 1.0) < 1e-9,
                            "order_accuracy": om["order_accuracy"], "fill_rate": om["fill_rate"], "expected_fill": len(with_ref) / len(records), "relative_price": om["relative_price"]}

    # 2. never order, no text: volume 0, order accuracy undefined
    code, m = score(a, "silent", [pred(s["id"], "wait") for s in steps])
    om = m["official_metrics"]
    results["never_order"] = {"passed": om["fill_rate"] == 0 and om["n_model_orders"] == 0 and not isinstance(om["order_accuracy"], float), "order_accuracy": om["order_accuracy"]}

    # 3. order the latest offer at every legal step: one deal per negotiation, price = first legal offer
    preds, first_legal = [], {}
    for s in steps:
        conv = s["id"].rsplit("#", 1)[0]
        if legal(s["step_meta"]):
            vo = s["step_meta"]["valid_order"]
            preds.append(pred(s["id"], "order", {"product_id": vo["product_id"], "agree_price": float(vo["agree_price"]), "qty": 1}))
            first_legal.setdefault(conv, float(vo["agree_price"]))
        else:
            preds.append(pred(s["id"], "wait"))
    code, m = score(a, "eager", preds)
    deals = [k for k, n in m["negotiations"].items() if n["deal"]]
    ratios_ok = all(abs(m["negotiations"][k]["price_ratio"] - first_legal[k] / (records[k]["reference_price"] / 100)) < 1e-9 for k in deals)
    results["order_every_legal_step"] = {"passed": set(deals) == set(first_legal) and m["official_metrics"]["order_accuracy"] == 1.0 and ratios_ok,
                                         "deals": len(deals), "negotiations_with_legal_offer": len(first_legal)}

    # 4. one cent off at the first legal step, then right at the next legal step
    preds, seen = [], collections_counter()
    for s in steps:
        conv = s["id"].rsplit("#", 1)[0]
        if legal(s["step_meta"]):
            vo = s["step_meta"]["valid_order"]
            price = float(vo["agree_price"]) + (0.01 if seen[conv] == 0 else 0)
            seen[conv] += 1
            preds.append(pred(s["id"], "order", {"product_id": vo["product_id"], "agree_price": round(price, 2), "qty": 1}))
        else:
            preds.append(pred(s["id"], "wait"))
    code, m = score(a, "cent_off", preds)
    twice = {k for k, v in seen.items() if v >= 2}
    deals = {k for k, n in m["negotiations"].items() if n["deal"]}
    results["cent_off_then_right"] = {"passed": deals == twice and m["official_metrics"]["order_accuracy"] < 1.0 and m["order_outcomes"].get("outdated_or_wrong_price", 0) >= len(seen),
                                      "deals": len(deals), "expected": len(twice), "outcomes": m["order_outcomes"]}

    # 5. after a raised price, ordering at the first (older) price is wrong
    preds, hit = [], 0
    for s in steps:
        meta, conv = s["step_meta"], s["id"].rsplit("#", 1)[0]
        offers = list(meta.get("standing_offers", {}).values())
        if records[conv]["flow"] == "price_up_latest" and len(offers) >= 2 and offers[0]["agree_price"] != meta["valid_order"]["agree_price"]:
            preds.append(pred(s["id"], "order", {"product_id": offers[0]["product_id"], "agree_price": float(offers[0]["agree_price"]), "qty": 1}))
            hit += 1
        else:
            preds.append(pred(s["id"], "wait"))
    if hit:
        code, m = score(a, "old_price", preds)
        results["old_price_rejected"] = {"passed": m["official_metrics"]["fill_rate"] == 0 and m["order_outcomes"].get("outdated_or_wrong_price") == hit, "orders": hit}
    else:
        results["old_price_rejected"] = {"passed": True, "note": "this split has no price_up_latest flow"}

    # 6. missing predictions -> not complete, exit 1
    code, m = score(a, "missing", [pred(s["id"], "wait") for s in steps[1:]])
    results["missing_predictions"] = {"passed": code == 1 and m.get("complete") is False and m.get("n_missing_predictions") == 1}

    passed = all(v["passed"] for v in results.values())
    json.dump({"passed": passed, **results}, open(a.out, "x"), indent=1)
    for k, v in results.items():
        print("PASS" if v["passed"] else "FAIL", k, {x: y for x, y in v.items() if x != "passed"})
    sys.exit(0 if passed else 1)


def collections_counter():
    import collections
    return collections.Counter()


if __name__ == "__main__":
    main()

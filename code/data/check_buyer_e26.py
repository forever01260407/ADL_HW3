"""Check E26 buyer files from the rendered messages, without trusting gen_buyer_e26.py.

    python code/data/check_buyer_e26.py FILE [--self-test] [--report OUT.json]

Facts come from check_buyer.steps (budget, target, acceptable ids, qty from the system JSON; the
latest offer from the seller's Offer lines). The E26 rule is re-implemented here:
  offers_seen  = Offer lines in all seller messages before the step
  legal        = acceptable id, required qty, price x qty <= budget (Decimal)
  accept       = legal and (price <= target or price x 100 < target x 105 or offers_seen >= 2)
  accept       -> the step must be an order of the latest offer (official order_ok_hw with citation,
                  Decimal, same terms, cites the latest offer's message)
  otherwise    -> no order; end_session only on an over-budget offer once 3 over-budget offers for the
                  right item were seen or the latest seller message says the price is final/lowest.
Also: check_buyer's structure and text faults, the official scorer on the data's own replies, and
--self-test mutations that must each be caught by their own rule.
"""
import argparse
import collections
import copy
import json
import re
import sys
from decimal import Decimal

sys.path.insert(0, "code/data")

import check_buyer as C  # noqa: E402

C.FINAL_WORDS = re.compile(r"final|any lower|can't go lower|lowest I can", re.I)
OFFER = re.compile(r"^Offer: (\S+) at \$(\d+\.\d{2}) per unit, qty (\d+)\.", re.M)


def seen_offers(messages, upto):
    out = []
    for m in messages[:upto]:
        if m["role"] == "user":
            out += [(o[0], Decimal(o[1]), int(o[2])) for o in OFFER.findall(m["content"])]
    return out


def expected(step, messages):
    sit, latest = step["situation"], step["latest"]
    offers = seen_offers(messages, step["index"])
    if latest is None or latest["product_id"] not in sit["accept"] or latest["qty"] != sit["qty"]:
        return "no_order", offers
    price = Decimal(latest["agree_price"])
    if price * latest["qty"] > sit["budget"]:
        over = sum(1 for pid, p, q in offers if pid in sit["accept"] and q == sit["qty"] and p * q > sit["budget"])
        return ("leave_allowed" if over >= 3 or step["final"] else "no_order"), offers
    if price <= sit["target"] or price * 100 < sit["target"] * 105 or len(offers) >= 2:
        return "order", offers
    return "no_order", offers


def check(record):
    faults = C.structure_faults(record)
    cases = collections.Counter()
    msgs = record["messages"]
    for step in C.steps(record):
        want, offers = expected(step, msgs)
        names = [c["function"]["name"] for c in step["calls"]]
        if len(names) > 1 or any(n not in C.TOOLS for n in names):
            faults.append("bad_call_count_or_tool")
        name = names[0] if names else "text_only"
        is_order = name == "order"
        if want == "order":
            if not is_order:
                faults.append("no_order_when_accept")
            else:
                faults += C.order_faults(step, step["calls"][0]["function"]["arguments"])
        else:
            if is_order:
                faults.append("order_when_not_accept")
            if name == "end_session" and want != "leave_allowed":
                faults.append("leave_too_early")
        faults += C.text_faults(step, is_order)
        cases[f"{want}:{name}"] += 1
    last = (msgs[-1].get("tool_calls") or [{}])[0].get("function", {}).get("name")
    if last not in ("order", "end_session"):
        faults.append("does_not_end_in_order_or_leave")
    return faults, cases


def mutate(record, kind):
    r = copy.deepcopy(record)
    m = r["messages"]
    last = m[-1]
    is_order = (last.get("tool_calls") or [{}])[0].get("function", {}).get("name") == "order"
    if kind in ("price_one_cent_off", "skip_order") and not is_order:
        return None
    if kind == "price_one_cent_off":
        a = last["tool_calls"][0]["function"]["arguments"]
        a["agree_price"] = round(a["agree_price"] + 0.01, 2)
        return r
    if kind == "skip_order":
        r["messages"] = m[:-1] + [{"role": "assistant", "content": "Could you go a little lower?", "tool_calls": [{"type": "function", "function": {"name": "wait", "arguments": {}}}]}]
        return r
    target = "no_order" if kind == "order_when_not_accept" else None
    for step in C.steps(record):
        want, _ = expected(step, record["messages"])
        latest = step["latest"]
        if want == target and latest and step["calls"] and step["calls"][0]["function"]["name"] == "wait":
            args = {"product_id": latest["product_id"], "agree_price": float(latest["agree_price"]), "qty": latest["qty"], "evidence_message_ids": [step["latest_id"]]}
            r["messages"] = m[:step["index"]] + [{"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "order", "arguments": args}}]}]
            return r
    return None


MUTATIONS = {"price_one_cent_off": "order_rejected_by_official_check", "skip_order": "no_order_when_accept", "order_when_not_accept": "order_when_not_accept"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--report")
    a = ap.parse_args()
    records = [json.loads(l) for l in open(a.file, encoding="utf-8")]
    bad, faults, cases = [], collections.Counter(), collections.Counter()
    for r in records:
        f, c = check(r)
        faults.update(f)
        cases.update(c)
        if f:
            bad.append(r["id"])
    print(f"{a.file}: {len(records)} negotiations, families {dict(collections.Counter(r['family'] for r in records))}")
    print("  violations:", dict(faults) or "none", ("first bad " + str(bad[:5])) if bad else "")
    print("  step cases (expected:actual):", dict(sorted(cases.items())))
    metrics, missing = C.official_echo(records)
    print(f"  official scorer on the data's own replies: order {metrics['order_accuracy']}, volume {metrics['fill_rate']}, price {metrics['relative_price']}, negotiations {metrics['n_negotiations']}")
    failed = bool(bad)
    st = {}
    if a.self_test:
        for kind, rule in MUTATIONS.items():
            tried = caught = 0
            for r in records:
                x = mutate(r, kind)
                if x is None:
                    continue
                tried += 1
                caught += rule in check(x)[0]
            st[kind] = [caught, tried]
            print(f"  self test {kind:24s} caught {caught}/{tried}")
            failed = failed or caught != tried or not tried
    if a.report:
        json.dump({"file": a.file, "negotiations": len(records), "violations": dict(faults), "bad": bad[:50], "cases": dict(cases), "self_test": st,
                   "official_echo": {k: metrics.get(k) for k in ("order_accuracy", "fill_rate", "relative_price", "n_negotiations")}}, open(a.report, "x"), indent=1)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

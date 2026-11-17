"""See when the model orders at public steps where an order would be valid.

    python code/diagnose_orders.py --pred FILE

Groups the orderable steps by facts a buyer can see (the offer against the target price, a
repeated offer, turns left) and compares how often the model orders. Output is counts only,
never public content, and nothing here goes to a data generator. Run from the project root.
"""
import argparse
import collections
import json
import re
import sys

sys.path.insert(0, "hw3")

from eval import format as F  # noqa: E402
from eval import score_negotiation as N  # noqa: E402

SITUATION = "YOUR SITUATION (private to you, fixed for this conversation):\n"
TURNS = re.compile(r'"turns_remaining_in_phase":\s*(\d+)')
# Our own word list for a seller saying the price will not move. It is not taken from the data.
CLOSING = re.compile(r"\bfinal\b|\bbest\b|\blowest\b|\bbottom\b|\bfirm\b|\babsolute\b|\blast\b|can't go|cannot go|no more room|take it or leave it|any lower|as low as", re.I)


def situation_of(row):
    return json.loads(row["messages"][0]["content"].split(SITUATION, 1)[1])


def turns_left(row):
    found = TURNS.findall(row["messages"][-1].get("content") or "")
    return int(found[-1]) if found else None


def offer_args(offer):
    return {"product_id": offer["product_id"], "agree_price": offer["agree_price"], "qty": offer["qty"]}


def orderable(meta):
    offer = meta.get("valid_order")
    return bool(offer) and N.order_ok_hw(meta, offer_args(offer), require_citation=False)


def price_band(price, target, budget):
    if price <= target:
        return "at or under target"
    middle = target + (budget - target) / 2
    return "above target, lower half to budget" if price <= middle else "above target, upper half to budget"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    args = ap.parse_args()

    predictions, _ = F.load_predictions([args.pred])
    rows = [row for row in F.read_jsonl(F.DATA_NEGOTIATION) if row["id"] in predictions]
    negotiations = collections.defaultdict(list)
    for row in rows:
        negotiations[row["negotiation"]].append(row)

    groups = collections.defaultdict(collections.Counter)
    for its in negotiations.values():
        its.sort(key=lambda row: row["msg_idx"])
        previous = None
        filled = False
        seen = 0
        for index, row in enumerate(its):
            meta = row["step_meta"]
            reply = N.reply_of(predictions[row["id"]], "lenient")
            call = N.effective_call(reply)
            ordered = bool(call) and call["name"] == "order"
            offer = meta.get("valid_order")
            terms = (offer["product_id"], offer["agree_price"], offer["qty"]) if offer else None
            if orderable(meta):
                seen += 1
                situation = situation_of(row)
                profile = situation["private_profile"]
                price = float(offer["agree_price"]) * int(offer["qty"])
                left = turns_left(row)
                last = index == len(its) - 1
                target = float(profile["target_price"])
                gap = (price - target) / target
                band = next(name for limit, name in ((-0.10, "under by over 10%"), (-0.02, "under by 2-10%"), (0.0, "under by up to 2%"), (0.02, "over by up to 2%"), (0.10, "over by 2-10%"), (float("inf"), "over by more than 10%")) if gap <= limit)
                facts = [
                    ("offer against target", band),
                    ("seller says the price will not move", bool(CLOSING.search(row["messages"][-1].get("content") or ""))),
                    ("offer", price_band(price, float(profile["target_price"]), float(profile["budget"]))),
                    ("same as previous offer", terms == previous),
                    ("turns left", "unknown" if left is None else "0-2" if left <= 2 else "3-6" if left <= 6 else "7+"),
                    ("orderable offers so far", min(seen, 4)),
                    ("already filled", filled),
                ]
                for fact in facts:
                    group = groups[fact]
                    group["n"] += 1
                    group["model orders"] += ordered
                    group["gold orders"] += last
                filled = filled or ordered
            previous = terms or previous

    print(f"{'fact':58s} {'steps':>6s} {'model orders':>13s} {'gold orders':>12s}")
    for fact in sorted(groups, key=lambda item: (item[0], str(item[1]))):
        group = groups[fact]
        print(f"{fact[0] + ' = ' + str(fact[1]):58s} {group['n']:6d} {100 * group['model orders'] / group['n']:12.1f}% {100 * group['gold orders'] / group['n']:11.1f}%")


if __name__ == "__main__":
    main()

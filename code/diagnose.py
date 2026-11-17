"""Count the kinds of mistakes a model makes on the public steps.

    python code/diagnose.py tools --pred FILE [--out FILE]
    python code/diagnose.py negotiation --pred FILE [--out FILE]

Every verdict comes from the official functions, and the totals are compared with the official
scorer before anything is printed. Output is counts only, never public content. Only the steps
that have a prediction are read, so a fixed sample works. Run from the project root.
"""
import argparse
import collections
import json
import sys

sys.path.insert(0, "hw3")

from eval import format as F  # noqa: E402
from eval import score_negotiation as N  # noqa: E402
from eval import score_tools as T  # noqa: E402

TOOL_BUCKETS = ((15, "up to 15 tools"), (30, "16-30 tools"), (10 ** 6, "more than 30 tools"))
CHAIN_BUCKETS = ("1 tool result since user", "2 tool results since user", "3+ tool results since user")


def tools_outcome(step, made_call):
    if step["capped"]:
        return "capped"
    if step["invalid"]:
        return "invalid"
    if step["is_call"]:
        if step["step"]:
            return "right call" if made_call else "text accepted"
        if not made_call:
            return "text instead of call"
        return "right tool, wrong arguments" if step["tool"] else "wrong tool"
    if step["step"]:
        return "right text"
    return "call instead of text" if made_call else "empty"


def previous_call(messages):
    for message in reversed(messages):
        if message["role"] == "assistant" and message.get("tool_calls"):
            return message["tool_calls"][-1]["function"]
    return None


def results_since_user(messages):
    count = 0
    for message in reversed(messages):
        if message["role"] == "user":
            break
        count += message["role"] == "tool"
    return count


def tool_bucket(row):
    return next(name for limit, name in TOOL_BUCKETS if len(row["tools"]) <= limit)


def system_bucket(row):  # same split as code/public_profile.py
    first = row["messages"][0]
    if first["role"] != "system":
        return "no system"
    return "long system" if len(first.get("content") or "") >= 2000 else "short system"


def diagnose_tools(args):
    rows = F.read_jsonl(F.DATA_TOOLS)
    predictions, _ = F.load_predictions([args.pred])
    result, steps = T.score_tools(rows, predictions, "lenient", set(predictions))
    by_id = {row["id"]: row for row in rows}

    groups = collections.defaultdict(collections.Counter)
    earned = total = 0
    for step in steps:
        row = by_id[step["id"]]
        reply = F.read_reply(predictions.get(step["id"]), "lenient")
        made_call = bool(reply and not reply["invalid"] and reply["calls"])
        outcome = tools_outcome(step, made_call)
        weight = F.settings()["must_call_weight"] if T.must_call(step) else 1
        total += weight
        earned += weight * step["step"]

        after = row["messages"][-1]["role"]
        kind = "must-call" if T.must_call(step) else "first call" if step["is_call"] else f"text after {after}"
        names = [kind]
        if T.must_call(step):
            names.append(f"must-call, {tool_bucket(row)}")
            names.append(f"must-call, {CHAIN_BUCKETS[min(results_since_user(row['messages']), 3) - 1]}")
            names.append(f"must-call, {system_bucket(row)}")
            names.append(f"must-call, {system_bucket(row)}, {CHAIN_BUCKETS[min(results_since_user(row['messages']), 3) - 1]}")
        elif kind == "text after tool":
            names.append(f"text after tool, {tool_bucket(row)}")
        for name in names:
            groups[name][outcome] += 1
            groups[name]["n"] += 1
        if made_call and not step["step"] and after == "tool":
            last = previous_call(row["messages"])
            call = reply["calls"][0]
            if last and last["name"] == call["name"]:
                groups[kind]["wrong call repeating the previous tool"] += 1
                groups[kind]["wrong call repeating the previous call exactly"] += T._same_call(T._canon(last, {}), T._canon(call, {}))

    official = result["with_first_step_rule"]["all"]["weighted_step_acc"]
    if abs(earned / total - official) > 1e-12:
        sys.exit(f"diagnosis {earned / total} does not match the official score {official}")

    print(f"weighted step accuracy {100 * official:.2f}")
    for name in sorted(groups):
        group = groups[name]
        parts = [f"{key} {value} ({100 * value / group['n']:.1f}%)" for key, value in sorted(group.items(), key=lambda item: -item[1]) if key != "n"]
        print(f"{name:42s} n={group['n']:5d}  " + "; ".join(parts))
    return {"weighted_step_acc": official, "groups": {name: dict(group) for name, group in groups.items()}}


def offer_args(offer):
    return {"product_id": offer["product_id"], "agree_price": offer["agree_price"], "qty": offer["qty"]}


def situation_of(meta):
    offer = meta.get("valid_order")
    if not offer:
        return "no offer yet"
    return "orderable offer" if N.order_ok_hw(meta, offer_args(offer), require_citation=False) else "offer that cannot be ordered"


def order_fault(meta, arguments):
    """Why an order failed. The rule itself is order_ok_hw; this only names the first broken part."""
    offer = meta.get("valid_order")
    if not offer:
        return "no offer yet"
    if arguments.get("product_id") not in (meta.get("acceptable_product_ids") or []):
        return "product not acceptable"
    if meta.get("budget") is not None and not N.budget_ok(arguments, meta["budget"]):
        return "over budget"
    if arguments.get("product_id") != offer["product_id"]:
        return "other product than the latest offer"
    if N.cents(arguments.get("agree_price")) != N.cents(offer["agree_price"]):
        return "other price than the latest offer"
    return "other quantity than the latest offer"


def action_of(reply):
    call = N.effective_call(reply)
    return call["name"] if call else N.shape_of(reply)


def diagnose_negotiation(args):
    predictions, _ = F.load_predictions([args.pred])
    rows = [row for row in F.read_jsonl(F.DATA_NEGOTIATION) if row["id"] in predictions]
    replies = {row["id"]: N.reply_of(predictions[row["id"]], "lenient") for row in rows}
    official = N.score_replies(rows, replies)

    by_negotiation = collections.defaultdict(list)
    for row in rows:
        by_negotiation[row["negotiation"]].append(row)

    actions = collections.defaultdict(collections.Counter)
    faults = collections.Counter()
    outcomes = collections.Counter()
    last_step = collections.Counter()
    first_valid_at = collections.Counter()
    after_first = collections.Counter()
    order_total = order_valid = 0

    for its in by_negotiation.values():
        its.sort(key=lambda row: row["msg_idx"])
        filled = ordered = False
        orderable_seen = 0
        for index, row in enumerate(its):
            meta, reply = row["step_meta"], replies[row["id"]]
            situation = situation_of(meta)
            orderable_seen += situation == "orderable offer"
            actions[situation][action_of(reply)] += 1
            actions[situation]["n"] += 1
            call = N.effective_call(reply)
            if index == len(its) - 1:
                last_step[action_of(reply)] += 1
            if not call or call["name"] != "order":
                continue
            ordered = True
            arguments = call["arguments"] if isinstance(call["arguments"], dict) else {}
            order_total += 1
            valid = bool(N.order_ok_hw(meta, arguments, require_citation=False))
            order_valid += valid
            if valid and not filled:
                filled = True
                first_valid_at["at the last step" if index == len(its) - 1 else f"at orderable offer no. {min(orderable_seen, 4)}{'+' if orderable_seen >= 4 else ''}"] += 1
            elif filled:
                after_first["valid" if valid else "invalid"] += 1
            if not valid:
                faults[order_fault(meta, arguments)] += 1
        outcomes["filled" if filled else "only invalid orders" if ordered else "no order at all"] += 1

    # These totals are what the official scorer counted. A mismatch means the steps were read differently.
    if order_total != official["n_model_orders"] or abs(outcomes["filled"] / len(by_negotiation) - official["fill_rate"]) > 1e-12:
        sys.exit("diagnosis does not match the official negotiation score")
    if order_total and abs(order_valid / order_total - official["order_accuracy"]) > 1e-12:
        sys.exit("diagnosis does not match the official order accuracy")

    print(f"{len(by_negotiation)} negotiations, {len(rows)} steps; order {official['order_accuracy']:.3f} (n={order_total}), "
          f"volume {official['fill_rate']:.3f}, price {official['relative_price']:.4f} (n={official['n_relative_price']})")
    print("negotiation outcomes:", dict(outcomes.most_common()))
    print("first valid order:", dict(sorted(first_valid_at.items())))
    print("orders after the first valid one:", dict(after_first))
    print("reason for invalid orders:", dict(faults.most_common()))
    print("action at the last step, where the gold order is:", dict(last_step.most_common()))
    for situation in sorted(actions):
        group = actions[situation]
        parts = [f"{key} {value} ({100 * value / group['n']:.1f}%)" for key, value in group.most_common() if key != "n"]
        print(f"{situation:30s} n={group['n']:5d}  " + "; ".join(parts))
    return {"official": {key: value for key, value in official.items() if isinstance(value, (int, float))},
            "outcomes": dict(outcomes), "first_valid_order": dict(first_valid_at), "orders_after_first": dict(after_first), "invalid_order_reasons": dict(faults),
            "last_step_action": dict(last_step), "actions": {name: dict(group) for name, group in actions.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=("tools", "negotiation"))
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()
    report = (diagnose_tools if args.task == "tools" else diagnose_negotiation)(args)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()

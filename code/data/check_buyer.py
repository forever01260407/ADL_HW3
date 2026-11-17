"""Check buyer conversations without trusting the generator.

    python code/data/check_buyer.py FILE
    python code/data/check_buyer.py --self-test FILE

Shares no code with gen_buyer.py. Every fact is parsed back out of the rendered messages:
budget and target from the system JSON, offers from the seller's `Offer:` lines. Orders
go through the official order_ok_hw, then a second time through exact Decimal arithmetic.
The whole file is also scored by the official negotiation scorer as a gold echo.
--self-test tampers with copies of the data and confirms each fault is caught.
Run from the project root. Exits non-zero on any violation.
"""
import argparse
import collections
import copy
import json
import re
import sys
from decimal import Decimal

sys.path.insert(0, "hw3")

from eval import score_negotiation as N  # noqa: E402

from common import first_problem  # noqa: E402

HEADER = "YOUR SITUATION (private to you, fixed for this conversation):\n"
OFFER = re.compile(r"^Offer: (\S+) at \$(\d+\.\d{2}) per unit, qty (\d+)\.")
QTY_RULE = re.compile(r"buy exactly (\d+) unit")
PRICE = re.compile(r"\$(\d+(?:\.\d{1,2})?)")
FINAL_WORDS = re.compile(r"price is final|any lower|drop it further|last offer|no more room", re.I)
TOOLS = {"order", "search_catalog", "wait", "end_session"}
MUST_ORDER = ("legal_at_target", "legal_final", "legal_deadline", "legal_stalled")
MIN_ROUNDS_BEFORE_CLOSING = 2  # the buyer may close above its target only after haggling this long
LEAKS = re.compile(
    r"\b(?:my|our)\s+(?:budget|target|limit|cap|ceiling|max(?:imum)?)\b|most I can pay|can go up to|private_profile|"
    r"as an AI|language model|\bI (?:cannot|can't) (?:help|assist)\b|\btools?\b|\bfunctions?\b|"
    r"order (?:is|has been|was) placed|already (?:paid|ordered|placed)|payment (?:sent|made)|I (?:have )?(?:ordered|purchased|paid)\b",
    re.I,
)


def as_json(line):
    try:
        value = json.loads(line)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def read_situation(messages):
    text = messages[0]["content"]
    situation = json.loads(text.split(HEADER, 1)[1])
    profile = situation["private_profile"]
    return {
        "budget": Decimal(str(profile["budget"])), "target": Decimal(str(profile["target_price"])),
        "accept": situation["acceptable_product_ids"], "qty": int(QTY_RULE.search(text)[1]),
        "lists": {Decimal(price) for item in situation["catalog"] for price in PRICE.findall(item["spec"])},
    }


def steps(record):
    """One dict per assistant step, holding what a buyer could see at that step."""
    messages = record["messages"]
    situation = read_situation(messages)
    standing, latest, latest_id, previous = {}, None, None, None
    final, spoke, text_price, turns, fresh = False, False, False, None, False
    for index, message in enumerate(messages):
        if message["role"] == "user":
            final, spoke, text_price, fresh = False, False, False, False
            sender = None
            for line in message["content"].split("\n"):
                value = as_json(line.strip()) if line.strip().startswith("{") else None
                if value and "from_id" in value:
                    sender, spoke = value["message_id"], True
                elif value and "phase" in value:
                    turns = value["turns_remaining_in_phase"]
                elif line.strip():
                    match = OFFER.match(line)
                    if match:
                        previous = latest
                        latest = {"product_id": match[1], "agree_price": match[2], "qty": int(match[3])}
                        latest_id, fresh = sender, True
                        standing[sender] = latest
                    elif PRICE.search(line):
                        text_price = True
                    final = final or bool(FINAL_WORDS.search(line))
        elif message["role"] == "assistant" and index:
            calls = message.get("tool_calls") or []
            yield {
                "index": index, "message": message, "text": (message.get("content") or "").strip(), "calls": calls, "situation": situation, "latest": latest,
                "latest_id": latest_id, "previous": previous, "fresh": fresh, "standing": dict(standing), "final": final, "spoke": spoke, "text_price": text_price, "turns": turns,
            }


def case_of(step):
    """What the visible facts say about this step."""
    latest, situation = step["latest"], step["situation"]
    if latest is None:
        return "no_offer"
    if latest["product_id"] not in situation["accept"]:
        return "wrong_product"
    if latest["qty"] != situation["qty"]:
        return "wrong_qty"
    price = Decimal(latest["agree_price"])
    if price * latest["qty"] > situation["budget"]:
        return "over_budget"
    if price <= situation["target"]:
        return "legal_at_target"
    if step["final"]:
        return "legal_final"
    if step["turns"] <= 2:
        return "legal_deadline"
    if step["fresh"] and step["previous"] == latest:
        return "legal_stalled"
    return "legal_above_target"


def meta_of(step):
    latest, situation = step["latest"], step["situation"]
    return {"acceptable_product_ids": situation["accept"], "budget": float(situation["budget"]), "valid_order": latest, "standing_offers": step["standing"]}


def order_faults(step, arguments):
    faults = []
    situation, latest = step["situation"], step["latest"]
    if not N.order_ok_hw(meta_of(step), arguments, require_citation=True):
        faults.append("order_rejected_by_official_check")
        return faults
    price = Decimal(str(arguments["agree_price"]))
    if price * arguments["qty"] > situation["budget"]:
        faults.append("order_over_budget_decimal")
    if price != Decimal(latest["agree_price"]) or arguments["qty"] != latest["qty"] or arguments["product_id"] != latest["product_id"]:
        faults.append("order_differs_from_latest_offer")
    if arguments["qty"] != situation["qty"]:
        faults.append("order_wrong_quantity")
    if arguments.get("evidence_message_ids") != [step["latest_id"]]:
        faults.append("order_cites_wrong_message")
    return faults


def text_faults(step, is_order):
    faults = []
    text, situation, latest = step["text"], step["situation"], step["latest"]
    if not text:
        return faults
    if LEAKS.search(text):
        faults.append("text_leak_or_role")
    for raw in PRICE.findall(text):
        price = Decimal(raw)
        if is_order:
            if latest is None or price != Decimal(latest["agree_price"]):
                faults.append("text_price_differs_from_order")
            continue
        # Checked before the list-price exemption: a target can equal some item's list price.
        if price in (situation["budget"], situation["target"]):
            faults.append("text_states_budget_or_target")
        if price in situation["lists"]:
            continue
        if price > situation["budget"]:
            faults.append("text_price_over_budget")
        if latest and latest["product_id"] in situation["accept"] and price >= Decimal(latest["agree_price"]):
            faults.append("text_counter_not_below_offer")
    return faults


def structure_faults(record):
    faults = []
    messages = record["messages"]
    if first_problem(record["tools"], messages):
        faults.append("schema")
    if messages[0]["role"] != "system" or messages[-1]["role"] != "assistant":
        faults.append("conversation_shape")
    seen = []
    previous_turns = None
    for index, message in enumerate(messages):
        if message["role"] == "user":
            lines = [line for line in message["content"].split("\n") if line.strip()]
            state = as_json(lines[-1])
            if not state or "phase" not in state:
                faults.append("user_message_without_state")
                continue
            turns = state["turns_remaining_in_phase"]
            if turns <= 0 or (previous_turns is not None and turns != previous_turns - 1):
                faults.append("turn_count_jump")
            previous_turns = turns
            for line in lines:
                value = as_json(line)
                if value and "from_id" in value:
                    seen.append(int(value["message_id"][2:]))
        elif message["role"] == "tool":
            body = json.loads(message["content"])
            caller = messages[index - 1]
            if ("reply_message_id" in body["data"]) != bool((caller.get("content") or "").strip()):
                faults.append("reply_id_mismatch")
            if "reply_message_id" in body["data"]:
                seen.append(int(body["data"]["reply_message_id"][2:]))
    if seen != sorted(set(seen)):
        faults.append("message_ids_not_increasing")
    return faults


def check(record):
    """Violations, and the case of each assistant step."""
    faults = structure_faults(record)
    cases = []
    legal_rounds = 0
    for step in steps(record):
        case = case_of(step)
        names = [call["function"]["name"] for call in step["calls"]]
        if len(names) > 1 or any(name not in TOOLS for name in names):
            faults.append("bad_call_count_or_tool")
        is_order = names[:1] == ["order"]
        if is_order:
            faults.extend(order_faults(step, step["calls"][0]["function"]["arguments"]))
            if case == "legal_above_target" and legal_rounds >= MIN_ROUNDS_BEFORE_CLOSING:
                cases.append("order_after_rounds")
            elif case not in MUST_ORDER:
                faults.append(f"order_at_{case}")
        elif case in MUST_ORDER:
            faults.append(f"no_order_at_{case}")
        legal_rounds += case == "legal_above_target"
        faults.extend(text_faults(step, is_order))
        if step["text_price"]:
            cases.append("text_price_only")
        if not step["spoke"] and step["index"] > 2:
            cases.append("seller_silent")
        if step["fresh"] and step["previous"] and step["previous"] != step["latest"]:
            cases.append("offer_updated")
            if Decimal(step["latest"]["agree_price"]) > Decimal(step["previous"]["agree_price"]):
                cases.append("offer_raised")
        cases.append(case)
        cases.append(f"action_{names[0] if names else 'text_only'}")
    if not (record["messages"][-1].get("tool_calls") or [{}])[0].get("function", {}).get("name") in ("order", "end_session"):
        faults.append("does_not_end_in_order_or_leave")
    return faults, cases


def official_echo(records):
    """Score the data's own replies with the official scorer."""
    rows, predictions = [], {}
    for number, record in enumerate(records):
        all_steps = list(steps(record))
        order_step = all_steps[-1]
        closing = order_step["calls"][0]["function"]
        if closing["name"] != "order":
            continue
        gold_price = float(closing["arguments"]["agree_price"])
        for position, step in enumerate(all_steps):
            step_id = f"{number}_{position}"
            call = step["calls"][0]["function"] if step["calls"] else None
            rows.append({"id": step_id, "negotiation": str(number), "msg_idx": position, "kind": "text", "gold_call": call and call["name"], "gold_price": gold_price, "gold_has_text": bool(step["text"]), "step_meta": meta_of(step)})
            predictions[step_id] = {"id": step_id, "tool_calls": [{"name": call["name"], "arguments": call["arguments"]}] if call else [], "text": step["text"]}
    metrics, _, missing = N.run(rows, predictions, "lenient")
    return metrics, missing


def tamper(record, kind):
    """A copy with one known fault, or None when the fault does not apply."""
    copied = copy.deepcopy(record)
    messages = copied["messages"]
    last = messages[-1]
    if not (last.get("tool_calls") and last["tool_calls"][0]["function"]["name"] == "order"):
        return None
    arguments = last["tool_calls"][0]["function"]["arguments"]
    if kind == "price_one_cent_off":
        arguments["agree_price"] = round(arguments["agree_price"] + 0.01, 2)
    elif kind == "wrong_quantity":
        arguments["qty"] = 2
    elif kind == "wrong_product":
        situation = json.loads(messages[0]["content"].split(HEADER, 1)[1])
        arguments["product_id"] = next(item["product_id"] for item in situation["catalog"] if item["product_id"] not in situation["acceptable_product_ids"])
    elif kind == "budget_below_price":
        head, body = messages[0]["content"].split(HEADER, 1)
        situation = json.loads(body)
        situation["private_profile"]["budget"] = round(arguments["agree_price"] - 0.01, 2)
        situation["private_profile"]["target_price"] = round(arguments["agree_price"] - 0.02, 2)
        messages[0]["content"] = head + HEADER + json.dumps(situation, indent=1, sort_keys=True) + "\n"
    elif kind == "offer_replaced_after_order_terms":
        seller = next(m for m in reversed(messages) if m["role"] == "user" and "Offer:" in m["content"])
        seller["content"] = re.sub(r"(Offer: \S+ at \$)(\d+)\.(\d{2})", lambda m: f"{m[1]}{int(m[2]) + 1}.{m[3]}", seller["content"])
    elif kind == "order_before_any_offer":
        messages[2] = {"role": "assistant", "content": "", "tool_calls": copy.deepcopy(last["tool_calls"])}
        copied["messages"] = messages[:3]
    elif kind == "text_leaks_budget":
        last["content"] = "My budget is tight."
    elif kind == "text_counter_equals_target":
        situation = json.loads(messages[0]["content"].split(HEADER, 1)[1])
        target = situation["private_profile"]["target_price"]
        copied["messages"] = messages[:2] + [{"role": "assistant", "content": f"Could you do ${target:.2f}?"}]
    elif kind == "skip_order_at_good_offer":
        if case_of(list(steps(record))[-1]) not in MUST_ORDER:
            return None
        copied["messages"] = messages[:-1] + [{"role": "assistant", "content": "Could you go lower?", "tool_calls": [{"type": "function", "function": {"name": "wait", "arguments": {}}}]}]
    elif kind == "order_at_first_legal_offer_above_target":
        first = next((step for step in steps(record) if case_of(step) == "legal_above_target"), None)
        if first is None:
            return None
        latest = first["latest"]
        arguments = {"product_id": latest["product_id"], "agree_price": float(latest["agree_price"]), "qty": latest["qty"], "evidence_message_ids": [first["latest_id"]]}
        copied["messages"] = messages[:first["index"]] + [{"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "order", "arguments": arguments}}]}]
    else:
        raise ValueError(kind)
    return copied


# Each fault must be caught by the rule written for it, not by a side effect of the edit.
MUTATIONS = {
    "price_one_cent_off": ("order_rejected_by_official_check",),
    "wrong_quantity": ("order_rejected_by_official_check",),
    "wrong_product": ("order_rejected_by_official_check",),
    "budget_below_price": ("order_rejected_by_official_check",),
    "offer_replaced_after_order_terms": ("order_rejected_by_official_check",),
    "order_before_any_offer": ("order_at_no_offer",),
    "text_leaks_budget": ("text_leak_or_role",),
    "text_counter_equals_target": ("text_states_budget_or_target",),
    "skip_order_at_good_offer": tuple(f"no_order_at_{case}" for case in MUST_ORDER),
    "order_at_first_legal_offer_above_target": ("order_at_legal_above_target",),
}


def self_test(records):
    failed = False
    for kind, expected in MUTATIONS.items():
        caught = tried = 0
        for record in records:
            changed = tamper(record, kind)
            if changed is None:
                continue
            tried += 1
            caught += any(fault in expected for fault in check(changed)[0])
        print(f"  {kind:36s} caught {caught}/{tried}")
        failed = failed or caught != tried or not tried
    return failed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    with open(args.file, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    violations, cases = collections.Counter(), collections.Counter()
    bad = []
    for record in records:
        faults, record_cases = check(record)
        violations.update(faults)
        cases.update(record_cases)
        if faults:
            bad.append(record["id"])
    print(f"{args.file}: {len(records)} negotiations")
    print("  violations:", dict(sorted(violations.items())) or "none")
    if bad:
        print("  first bad records:", bad[:5])
    print("  cases:")
    for key, value in sorted(cases.items()):
        print(f"    {key:24s} {value:6d}")
    metrics, missing = official_echo(records)
    print(f"  official scorer on the data's own replies: order {metrics['order_accuracy']}, cited {metrics['order_accuracy_cited']}, volume {metrics['fill_rate']}, "
          f"price {metrics['relative_price']}, negotiations {metrics['n_negotiations']}, missing {missing}")
    failed = bool(bad)
    if args.self_test:
        print("  self test:")
        failed = self_test(records) or failed
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

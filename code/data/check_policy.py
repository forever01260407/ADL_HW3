"""Check service conversations against the rules their policy was written from.

    python code/data/check_policy.py [--self-test] FILE

This imports nothing from the generator. It reads the messages and the `rules` saved with each
record and replays the conversation: verification before account access, the opening check,
pre-checks, confirmation before a change, what must follow a change, refund limits and the
codes and priorities the policy lists. --self-test breaks valid records in known ways and
fails unless each break raises the exact fault meant for it. Exits non-zero on any violation.
"""
import argparse
import collections
import copy
import json
import sys

IDENTITY = {"find_email", "find_person", "verify_member"}
ACCOUNT = {"list_mids", "get_leaf", "check", "quote", "act", "credit", "move", "update_contact", "open_case", "note", "notify", "escalate"}
CHANGES = {"act", "credit", "move", "update_contact"}
CONFIRMED = {"act", "credit", "move"}
AGREE = ("yes", "that's right", "confirmed")


def result_of(message):
    data = json.loads(message["content"])
    if isinstance(data, dict) and set(data) == {"status", "data"}:
        return data["data"]
    return data


def verified_by(result):
    return bool(result.get("results")) or result.get("verified") is True


def policy_faults(record, logical):
    rules, system = record["rules"], record["messages"][0]
    faults = []
    if system["role"] != "system":
        return ["no_system_message"]
    text = system["content"]
    offered = {tool["function"]["name"] for tool in record["tools"]}
    named = {"change": ("act",), "refund": ("quote", "credit", "escalate"), "move": ("move",), "contact": ("update_contact",), "case": ("open_case",)}
    for capability in rules["capabilities"]:
        for key in named[capability]:
            real = rules["tool_names"].get(key)
            if real is None or real not in offered or real not in text:
                faults.append("policy_misses_tool")
    if "refund" in rules["capabilities"]:
        if f"${rules['credit_limit']}" not in text or any(code not in text for code in rules["reason_codes"].values()):
            faults.append("policy_misses_refund_rule")
    if "case" in rules["capabilities"] and any(level not in text for level in rules["priorities"].values()):
        faults.append("policy_misses_priority")
    if rules["open_check"] and rules["tool_names"][rules["open_check"]] not in text:
        faults.append("policy_misses_open_check")
    return faults


def conversation_faults(record, logical):
    rules, messages = record["rules"], record["messages"]
    faults = []
    started = verified = listed = False
    eligible, quotes = {}, {}
    follow_ups = []
    previous_call = None
    awaiting_result = None
    for index, message in enumerate(messages):
        if message["role"] == "tool":
            result = result_of(message)
            key, arguments = awaiting_result
            if key in IDENTITY and verified_by(result):
                verified = True
            if key == "check":
                eligible[str(list(arguments.values())[0])] = result.get("eligible")
            if key == "quote":
                quotes[str(list(arguments.values())[0])] = result.get("refundable_amount")
                awaiting_quote = result.get("refundable_amount")
                following = messages[index + 1] if index + 1 < len(messages) else {}
                if awaiting_quote > rules["credit_limit"]:
                    calls = following.get("tool_calls") or []
                    if not calls or logical.get(calls[0]["function"]["name"]) != "escalate":
                        faults.append("over_limit_not_escalated")
                elif not awaiting_quote and following.get("tool_calls"):
                    faults.append("call_after_zero_quote")
            continue
        if message["role"] != "assistant":
            continue
        calls = message.get("tool_calls") or []
        if not calls:
            if follow_ups:
                faults.append("skipped_follow_up")
                follow_ups = []
            continue
        for call in calls:
            name, arguments = call["function"]["name"], call["function"]["arguments"]
            key = logical.get(name)
            if key is None:
                faults.append("unknown_tool")
                continue
            if rules["open_check"]:
                if not started and key != rules["open_check"]:
                    faults.append("first_call_not_open_check")
                elif started and key == rules["open_check"]:
                    faults.append("repeated_open_check")
            started = True
            if key in ACCOUNT and not verified:
                faults.append("before_verified")
            if key in IDENTITY and verified:
                faults.append("verified_twice")
            if key == "list_mids":
                if listed:
                    faults.append("repeated_list")
                listed = True
            if follow_ups:
                if key != follow_ups[0]:
                    faults.append("skipped_follow_up")
                    follow_ups = []
                else:
                    follow_ups.pop(0)
            # Calls list their arguments in schema order, so the record id comes first (second for escalate).
            target = None
            if key in ("check", "quote", "act", "credit", "move", "get_leaf"):
                target = str(next(iter(arguments.values())))
            if key in CONFIRMED and rules["confirm"]:
                before = messages[index - 1]
                asked = messages[index - 2] if index >= 2 else {}
                if before["role"] != "user" or not before["content"].lower().startswith(AGREE) or "?" not in (asked.get("content") or ""):
                    faults.append("no_confirmation")
            if key == "act":
                if rules["precheck"] and target not in eligible:
                    faults.append("missing_precheck")
                if eligible.get(target) is False:
                    faults.append("change_after_not_eligible")
            if key == "credit":
                amount = arguments["amount"]
                if quotes.get(target) != amount:
                    faults.append("refund_differs_from_quote")
                if amount > rules["credit_limit"]:
                    faults.append("refund_over_limit")
                if rules["credit_method"] and arguments["refund_method"] != rules["credit_method"]:
                    faults.append("wrong_refund_method")
                if arguments["reason_code"] not in rules["reason_codes"].values():
                    faults.append("unknown_reason_code")
            if key == "escalate":
                if arguments["amount"] <= rules["credit_limit"]:
                    faults.append("escalated_under_limit")
            if key == "open_case" and rules["priorities"].get(arguments["category"]) != arguments["priority"]:
                faults.append("wrong_priority")
            if key == "notify" and rules["notify"] and (arguments["channel"] != rules["notify"] or arguments["template"] != rules["template"]):
                faults.append("wrong_notice")
            if key == "move" and rules["move_channel"] and arguments["contact_channel"] != rules["move_channel"]:
                faults.append("wrong_contact_channel")
            if key == "transfer" and rules["case_before_transfer"] and "case" in rules["capabilities"] and previous_call != "open_case":
                faults.append("transfer_without_case")
            if key in CHANGES:
                follow_ups = (["note"] if rules["note_after"] else []) + (["notify"] if rules["notify"] else [])
            previous_call = key
            awaiting_result = (key, arguments)
    if follow_ups:
        faults.append("skipped_follow_up")
    return faults


def check(record):
    logical = {real: key for key, real in record["rules"]["tool_names"].items()}
    return policy_faults(record, logical) + conversation_faults(record, logical)


# ---- self-test


def indexes(record, key):
    real = record["rules"]["tool_names"].get(key)
    return [i for i, m in enumerate(record["messages"]) if any(c["function"]["name"] == real for c in m.get("tool_calls") or [])]


def drop_call(record, key):
    found = indexes(record, key)
    if not found:
        return False
    del record["messages"][found[0]:found[0] + 2]
    return True


def drop_open_check(record):
    return bool(record["rules"]["open_check"]) and drop_call(record, record["rules"]["open_check"])


def drop_identity(record):
    """Removes the lookup that succeeded. A failed one before it changes nothing."""
    found = [index for key in sorted(IDENTITY) for index in indexes(record, key)]
    if not found:
        return False
    del record["messages"][max(found):max(found) + 2]
    return True


def drop_precheck(record):
    """Removes every check of the record that is then changed. A person can back out and ask again."""
    changes = indexes(record, "act")
    if not record["rules"]["precheck"] or not changes:
        return False
    target = next(iter(record["messages"][changes[0]]["tool_calls"][0]["function"]["arguments"].values()))
    found = [index for index in indexes(record, "check") if next(iter(record["messages"][index]["tool_calls"][0]["function"]["arguments"].values())) == target]
    for index in reversed(found):
        del record["messages"][index:index + 2]
    return bool(found)


def skip_confirmation(record):
    if not record["rules"]["confirm"]:
        return False
    for key in ("act", "credit", "move"):
        found = indexes(record, key)
        if found:
            # The question and the yes both go, so the change follows a tool result.
            del record["messages"][found[0] - 2:found[0]]
            return True
    return False


def drop_note(record):
    return record["rules"]["note_after"] and drop_call(record, "note")


def edit_call(record, key, change):
    found = indexes(record, key)
    if not found:
        return False
    arguments = record["messages"][found[0]]["tool_calls"][0]["function"]["arguments"]
    change(arguments)
    return True


def raise_refund(record):
    return edit_call(record, "credit", lambda arguments: arguments.update(amount=record["rules"]["credit_limit"] + 1))


def swap_priority(record):
    return edit_call(record, "open_case", lambda arguments: arguments.update(priority="not-a-priority"))


def wrong_channel(record):
    return bool(record["rules"]["notify"]) and edit_call(record, "notify", lambda arguments: arguments.update(channel="not-a-channel"))


def skip_escalation(record):
    return drop_call(record, "escalate")


def repeat_list(record):
    found = indexes(record, "list_mids")
    if not found:
        return False
    record["messages"][found[0] + 2:found[0] + 2] = copy.deepcopy(record["messages"][found[0]:found[0] + 2])
    return True


MUTATIONS = {
    drop_open_check: "first_call_not_open_check", drop_identity: "before_verified", drop_precheck: "missing_precheck", skip_confirmation: "no_confirmation", drop_note: "skipped_follow_up",
    raise_refund: "refund_over_limit", swap_priority: "wrong_priority", wrong_channel: "wrong_notice", skip_escalation: "over_limit_not_escalated", repeat_list: "repeated_list",
}


def self_test(records):
    failed = 0
    for mutation, fault in MUTATIONS.items():
        tried = caught = 0
        for record in records:
            if check(record):
                continue
            broken = copy.deepcopy(record)
            if mutation(broken):
                tried += 1
                caught += fault in check(broken)
        print(f"  {mutation.__name__:22s} caught {caught}/{tried}")
        failed += tried == 0 or caught != tried
    return failed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    with open(args.file, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    faults = collections.Counter(fault for record in records for fault in check(record))
    bad = sum(bool(check(record)) for record in records)
    print(f"{args.file}: {len(records)} conversations, {bad} with violations {dict(faults.most_common())}")
    failed = self_test(records) if args.self_test else 0
    if bad or failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

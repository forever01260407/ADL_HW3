"""Check conversation files before they are used for training.

    python code/data/qa_conversations.py [--tokens] FILE [FILE ...]

Checks every record for schema problems, arguments that cannot be traced to an earlier
message, and numbers in a reply that no earlier message supports. --tokens also renders
each step with the official template and reports lengths, which needs the tokenizer.
Free-text arguments such as an email body are written by the assistant, so they are
counted separately instead of being traced.
"""
import argparse
import collections
import json
import re

from common import first_problem, step_kinds, wrong_type

FREE_TEXT = {"subject", "body", "expression", "reason", "query", "message", "content", "text", "description", "title", "note", "summary"}
NUMBER = re.compile(r"\d+(?:\.\d+)?")


def leaves(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from leaves(item)
    elif isinstance(value, list):
        for item in value:
            yield from leaves(item)
    else:
        yield value


def flat(text):
    return text.replace(",", "")


def untraced_arguments(messages):
    """(checked, untraced) over the non-free-text arguments of all calls."""
    seen, checked, untraced = "", 0, 0
    for message in messages:
        for call in message.get("tool_calls") or []:
            for name, value in call["function"]["arguments"].items():
                if name in FREE_TEXT:
                    continue
                for leaf in leaves(value):
                    checked += 1
                    untraced += str(leaf) not in seen and json.dumps(leaf) not in seen
        seen += "\n" + str(message.get("content") or "")
        for call in message.get("tool_calls") or []:
            seen += "\n" + json.dumps(call["function"]["arguments"], ensure_ascii=False)
    return checked, untraced


def wrong_argument_types(tools, messages):
    functions = {tool["function"]["name"]: tool["function"] for tool in tools}
    return sum(
        wrong_type(functions[call["function"]["name"]], call["function"]["arguments"]) is not None
        for message in messages for call in message.get("tool_calls") or []
        if call["function"]["name"] in functions
    )


def numbers_in(text):
    return {float(number) for number in NUMBER.findall(flat(text))}


def unsupported_numbers(messages):
    """Numbers in assistant text that appear in no earlier message. Small integers are
    usually counts of listed items, which a reply states without a tool saying them."""
    seen, count = set(), 0
    for message in messages:
        text = message.get("content") or ""
        if message["role"] == "assistant" and not message.get("tool_calls"):
            count += sum(number not in seen and not (number.is_integer() and number <= 20) for number in numbers_in(text))
        seen |= numbers_in(text)
    return count


def token_report(records):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    totals, targets, broken = [], [], 0
    for record in records:
        tools, messages = record["tools"], record["messages"]
        for index, message in enumerate(messages):
            if message["role"] != "assistant" or not index:
                continue
            prompt = tok.apply_chat_template(messages[:index], tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            full = tok.apply_chat_template(messages[:index + 1], tools=tools, tokenize=False, enable_thinking=False)
            if not full.startswith(prompt):
                broken += 1
                continue
            target = full[len(prompt):].rstrip("\n")
            prompt_tokens = len(tok(prompt, add_special_tokens=False)["input_ids"])
            target_tokens = len(tok(target, add_special_tokens=False)["input_ids"])
            totals.append(prompt_tokens + target_tokens)
            targets.append(target_tokens)
    totals.sort()
    targets.sort()
    at = lambda values, share: values[min(len(values) - 1, int(share * len(values)))]  # noqa: E731
    print(f"  tokens per step: median {at(totals, 0.5)}, p99 {at(totals, 0.99)}, max {totals[-1]}; over 4096: {sum(t > 4096 for t in totals)}")
    print(f"  target tokens: median {at(targets, 0.5)}, p99 {at(targets, 0.99)}, max {targets[-1]}; over 1024: {sum(t > 1024 for t in targets)}")
    print(f"  prompt not a prefix of the full text: {broken}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--tokens", action="store_true")
    args = ap.parse_args()
    for path in args.files:
        with open(path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        problems, kinds = collections.Counter(), collections.Counter()
        checked = untraced = numbers = types = 0
        for record in records:
            found = first_problem(record["tools"], record["messages"])
            if found:
                problems[found[1]] += 1
            kinds.update(step_kinds(record["messages"]))
            done, missing = untraced_arguments(record["messages"])
            checked += done
            untraced += missing
            numbers += unsupported_numbers(record["messages"])
            types += wrong_argument_types(record["tools"], record["messages"])
        print(f"\n{path}: {len(records)} conversations, {sum(kinds.values())} assistant steps")
        print(f"  schema problems: {dict(problems) or 'none'}")
        print(f"  arguments traced: {checked - untraced}/{checked}")
        print(f"  arguments of the wrong type: {types}")
        print(f"  reply numbers with no support: {numbers}")
        if args.tokens:
            token_report(records)


if __name__ == "__main__":
    main()

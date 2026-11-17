"""Print a stratified sample of data_v0 steps for reading by hand.

    python code/data/sample_steps.py --per-group 4 --seed 7 data_work/processed/data_v0.jsonl

Part 1 steps are grouped by source and step kind, Part 2 steps by what check_buyer sees at
the step, so every behaviour shows up. Long tool lists and system prompts are left out;
only the context needed to judge the target is printed.
"""
import argparse
import collections
import json
import random

import check_buyer
from common import expand_steps, step_kinds


def target_text(message):
    parts = [repr(message.get("content") or "")] if message.get("content") else []
    for call in message.get("tool_calls") or []:
        parts.append(f"CALL {call['function']['name']} {json.dumps(call['function']['arguments'], ensure_ascii=False)}")
    return " + ".join(parts)


def groups_of(records):
    groups = collections.defaultdict(list)
    for record in records:
        if record["task"] == "part2":
            parsed = {step["index"]: step for step in check_buyer.steps(record)}
            for example in expand_steps(record):
                index = len(example["messages"]) - 1
                groups[f"part2 {check_buyer.case_of(parsed[index])}"].append(example)
        else:
            for example in expand_steps(record):  # a no_target step is skipped, so find each kind from its own messages
                kind = step_kinds(example["messages"])[-1]
                groups[f"part1 {record['source']} {kind}"].append(example)
    return groups


def show(example):
    messages = example["messages"]
    print(f"--- {example['id']} ({len(example['tools'])} tools)")
    if example["task"] == "part2":
        profile = json.loads(messages[0]["content"].split(check_buyer.HEADER, 1)[1])["private_profile"]
        print(f"    budget {profile['budget']} target {profile['target_price']}")
    for message in messages[-4:-1]:
        if message["role"] == "system":
            continue
        print(f"    {message['role']}: {target_text(message) if message['role'] == 'assistant' else str(message['content'])[:400]}")
    print(f"  > TARGET: {target_text(messages[-1])[:500]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--per-group", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    with open(args.file, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    rng = random.Random(args.seed)
    groups = groups_of(records)
    shown = 0
    for name in sorted(groups):
        picks = rng.sample(groups[name], min(args.per_group, len(groups[name])))
        print(f"\n=========== {name}: {len(groups[name])} steps, showing {len(picks)}")
        for example in picks:
            show(example)
        shown += len(picks)
    print(f"\nshown {shown}")


if __name__ == "__main__":
    main()

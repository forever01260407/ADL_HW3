"""Assemble the small data_v0 set from the processed sources.

    python code/data/assemble_v0.py

Writes data_work/processed/data_v0.jsonl, one conversation per line, and a manifest with
counts and hashes. Training turns each assistant message into one example at load time
(common.expand_steps), so the conversation, source and family ids stay with every step.
Part 1 conversations are kept only when they pass every check: valid schema, right
argument types, all arguments traceable, no unsupported number in a reply. Part 2 goes
through check_buyer.py. Whole conversations are taken, so a family never splits.
BFCL is absent until its tools can run.
"""
import collections
import hashlib
import json
import random
import re

import check_buyer
from common import first_problem, stable_hash, step_kinds
from qa_conversations import unsupported_numbers, untraced_arguments, wrong_argument_types

SELF_TALK = re.compile(r"\bas an AI\b|language model|given functions|provided functions|functions? (?:provided|available)|the function \"", re.I)
OUT = "data_work/processed/data_v0.jsonl"
MANIFEST = "data_work/processed/data_v0.manifest.json"
SEED = 20261107
# Assistant steps to take from each source. Part 2 takes the rest of the 500 step ceiling.
QUOTAS = (
    ("chains_v0", "part1", 170), ("toolace", "part1", 70), ("glaive", "part1", 45), ("hermes", "part1", 25),
    ("buyer_v0", "part2", 170),
)


def source_noise(messages):
    """Boilerplate about being an AI could reach buyer replies, where the judge marks talk of
    one's own abilities as unreasonable. The wrapper is a source artifact, not a user turn."""
    for message in messages:
        text = str(message.get("content") or "")
        if (message["role"] == "assistant" and SELF_TALK.search(text)) or (message["role"] == "user" and text.lstrip().startswith("Role definition:")):
            return True
    return False


def clean(record, task):
    """Buyer data has its own checker: a buyer's counter-offer is a new number on purpose,
    so the rule that a reply's numbers must appear earlier does not apply to it."""
    if task == "part2":
        return not check_buyer.check(record)[0]
    tools, messages = record["tools"], record["messages"]
    return (
        not source_noise(messages) and
        first_problem(tools, messages) is None and untraced_arguments(messages)[1] == 0
        and unsupported_numbers(messages) == 0 and wrong_argument_types(tools, messages) == 0
    )


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    rng = random.Random(SEED)
    kept, seen = [], set()
    counts = collections.defaultdict(collections.Counter)
    inputs = {}
    for name, task, quota in QUOTAS:
        path = f"data_work/processed/{name}.jsonl"
        inputs[name] = sha256(path)
        with open(path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        rng.shuffle(records)
        steps = 0
        for record in records:
            if steps >= quota:
                break
            if not clean(record, task):
                counts[name]["rejected_by_checks"] += 1
                continue
            key = stable_hash([record["tools"], record["messages"]])
            if key in seen:
                counts[name]["duplicates"] += 1
                continue
            seen.add(key)
            kinds = step_kinds(record["messages"])
            steps += len(kinds)
            counts[name].update(kinds)
            counts[name]["conversations"] += 1
            kept.append({**record, "task": task})
    rng.shuffle(kept)
    with open(OUT, "w", encoding="utf-8") as f:
        for record in kept:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {"seed": SEED, "inputs": inputs, "output_sha256": sha256(OUT), "counts": {name: dict(count) for name, count in counts.items()}}
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    total = collections.Counter(record["task"] for record in kept)
    steps = collections.Counter()
    for record in kept:
        steps[record["task"]] += len(step_kinds(record["messages"]))
    print(f"conversations {dict(total)}, assistant steps {dict(steps)}, families {len({record['family'] for record in kept})}")
    for name, count in counts.items():
        print(f"  {name}: {dict(sorted(count.items()))}")


if __name__ == "__main__":
    main()

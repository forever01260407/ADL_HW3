"""Build and freeze data_v1, the first full train and dev sets.

    python code/data/assemble_v1.py

Reads the processed sources and writes data_work/frozen/data_v1/{train,dev}.jsonl with a
manifest of hashes, counts, seeds and the reason for every rejected conversation. The split
is by family and checked: dev holds generator domains and product categories that training
never sees, and tool sets from the third-party sources that training never sees. Part 1
goes through the checks in clean_part1, Part 2 through check_buyer.py. Quotas count
assistant steps, and whole conversations are taken.
"""
import collections
import hashlib
import json
import os
import random
import re
import sys

import check_buyer
from common import first_problem, stable_hash, step_kinds
from qa_conversations import unsupported_numbers, untraced_arguments, wrong_argument_types

SEED = 20261108
OUT = "data_work/frozen/data_v1"
PROCESSED = "data_work/processed"
COLLISIONS = f"{PROCESSED}/collisions_v1.json"
#            source      task     train  dev
SOURCES = (
    ("chains_v1", "part1", 8000, 350), ("toolace", "part1", 2600, 100), ("glaive", "part1", 2800, 100), ("hermes", "part1", 600, 50),
    ("buyer_v1", "part2", 6000, 400),
)
DEV_DOMAINS = 3      # of the 20 generator domains
DEV_CATEGORIES = 2   # of the 10 product categories
DEV_FAMILY_SHARE = 50  # one third-party tool set in this many goes to dev
SELF_TALK = re.compile(
    r"\bas an AI\b|language model|given functions|provided functions|functions? (?:provided|available)|the function \"|the \"[^\"]+\" function|"
    r"capabilit|my (?:current )?functions?\b|not equipped",
    re.I,
)


def source_noise(messages):
    """Boilerplate about being an AI could reach buyer replies, where the judge marks talk of
    one's own abilities as unreasonable. The wrapper is a source artifact, not a user turn."""
    for message in messages:
        text = str(message.get("content") or "")
        if message["role"] == "assistant" and SELF_TALK.search(text):
            return "talks_about_itself"
        if message["role"] == "user" and text.lstrip().startswith("Role definition:"):
            return "wrapped_history"
    return None


def clean_part1(record):
    """Why this Part 1 conversation is left out, or None."""
    tools, messages = record["tools"], record["messages"]
    if first_problem(tools, messages):
        return "schema"
    noise = source_noise(messages)
    if noise:
        return noise
    if wrong_argument_types(tools, messages):
        return "argument_type"
    if untraced_arguments(messages)[1]:
        return "untraced_argument"
    if unsupported_numbers(messages):
        return "unsupported_number"
    return None


def reject_reason(record, task, collisions):
    if record["id"] in collisions:
        return "overlaps_public"
    if task == "part2":
        return "buyer_check" if check_buyer.check(record)[0] else None
    return clean_part1(record)


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def dev_families(rng):
    """The families dev keeps to itself."""
    domains = sorted({record_family for record_family in read_families("chains_v1")})
    categories = sorted({record_family for record_family in read_families("buyer_v1")})
    return set(rng.sample(domains, DEV_DOMAINS)) | set(rng.sample(categories, DEV_CATEGORIES))


def read_families(name):
    with open(f"{PROCESSED}/{name}.jsonl", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)["family"]


def goes_to_dev(family, generated, held_out):
    if generated:
        return family in held_out
    return int(stable_hash(family), 16) % DEV_FAMILY_SHARE == 0


def take(records, quota, task, seen, counts, name, collisions):
    """Whole conversations until the quota of assistant steps is reached."""
    kept, steps = [], 0
    for record in records:
        if steps >= quota:
            break
        reason = reject_reason(record, task, collisions)
        if reason:
            counts[name][f"rejected_{reason}"] += 1
            continue
        key = stable_hash([record["tools"], record["messages"]])
        if key in seen:
            counts[name]["rejected_duplicate"] += 1
            continue
        seen.add(key)
        kinds = step_kinds(record["messages"])
        steps += len(kinds)
        kept.append({**record, "task": task})
    return kept, steps


def write(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def tally(records):
    table = collections.defaultdict(collections.Counter)
    for record in records:
        table[record["source"]]["conversations"] += 1
        table[record["source"]].update(step_kinds(record["messages"]))
    return {source: dict(sorted(count.items())) for source, count in sorted(table.items())}


def main():
    rng = random.Random(SEED)
    held_out = dev_families(rng)
    with open(COLLISIONS, encoding="utf-8") as f:
        collisions = set(json.load(f)["drop_ids"])
    seen, counts = set(), collections.defaultdict(collections.Counter)
    train, dev, inputs = [], [], {}
    for name, task, train_quota, dev_quota in SOURCES:
        path = f"{PROCESSED}/{name}.jsonl"
        inputs[name] = sha256(path)
        with open(path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        rng.shuffle(records)
        generated = name.endswith("_v1")
        pools = {True: [], False: []}
        for record in records:
            pools[goes_to_dev(record["family"], generated, held_out)].append(record)
        kept, steps = take(pools[False], train_quota, task, seen, counts, name, collisions)
        train.extend(kept)
        counts[name]["train_steps"] = steps
        kept, steps = take(pools[True], dev_quota, task, seen, counts, name, collisions)
        dev.extend(kept)
        counts[name]["dev_steps"] = steps
    rng.shuffle(train)
    rng.shuffle(dev)

    overlap = {record["family"] for record in train} & {record["family"] for record in dev}
    if overlap:
        sys.exit(f"families on both sides: {sorted(overlap)[:5]}")
    os.makedirs(OUT, exist_ok=True)
    write(f"{OUT}/train.jsonl", train)
    write(f"{OUT}/dev.jsonl", dev)
    manifest = {
        "seed": SEED, "inputs": inputs, "train_sha256": sha256(f"{OUT}/train.jsonl"), "dev_sha256": sha256(f"{OUT}/dev.jsonl"),
        "dev_held_out_families": sorted(held_out), "train": tally(train), "dev": tally(dev),
        "selection": {name: dict(sorted(count.items())) for name, count in counts.items()},
    }
    with open(f"{OUT}/manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print("held out:", sorted(held_out))
    for side in ("train", "dev"):
        print(side)
        for source, count in manifest[side].items():
            print(f"  {source:12s} {count}")
    for name, count in manifest["selection"].items():
        print(f"  {name:12s} {count}")


if __name__ == "__main__":
    main()

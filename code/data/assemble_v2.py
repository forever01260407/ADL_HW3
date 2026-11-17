"""Build and freeze data_v2.

    python code/data/assemble_v2.py

Part 1 keeps all of data_v1 and adds service conversations under long policies. Part 2 is new
buyer data whose first offers stand anywhere against the target. Dev stays comparable: dev.jsonl
is data_v1's dev, byte for byte, and dev_new.jsonl holds the new shapes from domains and
categories that training never sees. The split is by family and checked. Writes train.jsonl,
dev.jsonl, dev_new.jsonl and a manifest with hashes, counts and the reason for every rejection.
"""
import collections
import json
import os
import random
import shutil
import sys

import check_buyer
import check_policy
from assemble_v1 import clean_part1, sha256, tally, write
from common import stable_hash, step_kinds

SEED = 20261109
OUT = "data_work/frozen/data_v2"
V1 = "data_work/frozen/data_v1"
PROCESSED = "data_work/processed"
COLLISIONS = f"{PROCESSED}/collisions_v2.json"
#            source       task     train  dev_new
SOURCES = (("policy_v2", "part1", 11000, 500), ("buyer_v2", "part2", 8000, 400))
# The same domains and categories data_v1 holds out, so dev never meets them in training.
HELD_OUT = {"policy_gym", "policy_hr", "policy_realty", "buyer_Kitchen", "buyer_Outdoors"}


def reject_reason(record, task, collisions):
    if record["id"] in collisions:
        return "overlaps_public"
    if task == "part2":
        return "buyer_check" if check_buyer.check(record)[0] else None
    if check_policy.check(record):
        return "policy_check"
    return clean_part1(record)


def take(records, quota, task, seen, counts, name, collisions):
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
        steps += len(step_kinds(record["messages"]))
        kept.append({**record, "task": task})
    return kept, steps


def main():
    rng = random.Random(SEED)
    with open(COLLISIONS, encoding="utf-8") as f:
        collisions = set(json.load(f)["drop_ids"])
    with open(f"{V1}/train.jsonl", encoding="utf-8") as f:
        train = [record for record in map(json.loads, f) if record["task"] == "part1"]
    seen, counts, inputs, dev_new = set(), collections.defaultdict(collections.Counter), {}, []
    for name, task, train_quota, dev_quota in SOURCES:
        path = f"{PROCESSED}/{name}.jsonl"
        inputs[name] = sha256(path)
        with open(path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        rng.shuffle(records)
        kept, steps = take([r for r in records if r["family"] not in HELD_OUT], train_quota, task, seen, counts, name, collisions)
        train.extend(kept)
        counts[name]["train_steps"] = steps
        kept, steps = take([r for r in records if r["family"] in HELD_OUT], dev_quota, task, seen, counts, name, collisions)
        dev_new.extend(kept)
        counts[name]["dev_new_steps"] = steps
    rng.shuffle(train)
    rng.shuffle(dev_new)

    os.makedirs(OUT, exist_ok=True)
    shutil.copyfile(f"{V1}/dev.jsonl", f"{OUT}/dev.jsonl")
    with open(f"{OUT}/dev.jsonl", encoding="utf-8") as f:
        old_dev = {json.loads(line)["family"] for line in f}
    overlap = {record["family"] for record in train} & (old_dev | {record["family"] for record in dev_new})
    if overlap:
        sys.exit(f"families on both sides: {sorted(overlap)[:5]}")
    write(f"{OUT}/train.jsonl", train)
    write(f"{OUT}/dev_new.jsonl", dev_new)
    manifest = {
        "seed": SEED, "inputs": inputs, "data_v1_train_sha256": sha256(f"{V1}/train.jsonl"), "train_sha256": sha256(f"{OUT}/train.jsonl"), "dev_sha256": sha256(f"{OUT}/dev.jsonl"),
        "dev_new_sha256": sha256(f"{OUT}/dev_new.jsonl"), "held_out_families": sorted(HELD_OUT), "train": tally(train), "dev_new": tally(dev_new),
        "selection": {name: dict(sorted(count.items())) for name, count in counts.items()},
    }
    with open(f"{OUT}/manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    for side in ("train", "dev_new"):
        print(side)
        for source, count in manifest[side].items():
            print(f"  {source:12s} {count}")
    for name, count in manifest["selection"].items():
        print(f"  {name:12s} {count}")


if __name__ == "__main__":
    main()

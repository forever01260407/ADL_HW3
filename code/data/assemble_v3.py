"""Build and freeze data_v3: data_v1's Part 1 with data_v2's Part 2.

    python code/data/assemble_v3.py

E20 trained on data_v2, which changed both tasks at once. Part 2 improved and Part 1 fell
(records/e20_results.md), so this version keeps only the Part 2 change. Nothing is generated or
filtered again: Part 1 is every Part 1 record of data_v1's train.jsonl, Part 2 every Part 2
record of data_v2's train.jsonl, both already checked. dev.jsonl is data_v1's dev byte for byte
and dev_new.jsonl is data_v2's, so E10, E20 and E21 score on the same dev sets.
"""
import json
import os
import random
import shutil
import sys

from assemble_v1 import sha256, tally, write

SEED = 20261110
OUT = "data_work/frozen/data_v3"
V1 = "data_work/frozen/data_v1"
V2 = "data_work/frozen/data_v2"


def records_of(path, task):
    with open(path, encoding="utf-8") as f:
        return [record for record in map(json.loads, f) if record["task"] == task]


def main():
    train = records_of(f"{V1}/train.jsonl", "part1") + records_of(f"{V2}/train.jsonl", "part2")
    random.Random(SEED).shuffle(train)

    os.makedirs(OUT, exist_ok=True)
    shutil.copyfile(f"{V1}/dev.jsonl", f"{OUT}/dev.jsonl")
    shutil.copyfile(f"{V2}/dev_new.jsonl", f"{OUT}/dev_new.jsonl")
    dev_families = set()
    for name in ("dev.jsonl", "dev_new.jsonl"):
        with open(f"{OUT}/{name}", encoding="utf-8") as f:
            dev_families |= {json.loads(line)["family"] for line in f}
    overlap = {record["family"] for record in train} & dev_families
    if overlap:
        sys.exit(f"families on both sides: {sorted(overlap)[:5]}")
    write(f"{OUT}/train.jsonl", train)
    manifest = {
        "seed": SEED,
        "made_from": {"part1": f"{V1}/train.jsonl", "part2": f"{V2}/train.jsonl"},
        "data_v1_train_sha256": sha256(f"{V1}/train.jsonl"), "data_v2_train_sha256": sha256(f"{V2}/train.jsonl"),
        "train_sha256": sha256(f"{OUT}/train.jsonl"), "dev_sha256": sha256(f"{OUT}/dev.jsonl"), "dev_new_sha256": sha256(f"{OUT}/dev_new.jsonl"),
        "train": tally(train),
    }
    with open(f"{OUT}/manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    for source, count in manifest["train"].items():
        print(f"  {source:12s} {count}")
    print("train", manifest["train_sha256"][:16], "dev", manifest["dev_sha256"][:16], "dev_new", manifest["dev_new_sha256"][:16])


if __name__ == "__main__":
    main()

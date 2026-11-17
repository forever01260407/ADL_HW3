"""Build and freeze data_v4: data_v3 plus the BFCL train records.

    python code/data/assemble_v4.py

The only change from data_v3 is one new Part 1 source, BFCL multi-turn tasks run through BFCL's
own tools (code/data/build_bfcl_multiturn.py, code/data/prepare_bfcl_v4.py). Every data_v3 train
record is kept unchanged, so E22 against E21 tests that one addition. The dev files:

- dev.jsonl: data_v1's dev, byte for byte (old dev, same as E10, E20, E21).
- dev_new.jsonl: data_v2's dev, byte for byte. Its Part 1 is long-policy conversations from a
  generator neither E21 nor E22 trains on, in domains no training data uses.
- dev_bfcl.jsonl: BFCL tasks from families kept out of training, each with and without the
  session state in a system message.
"""
import json
import os
import random
import shutil
import sys

from assemble_v1 import sha256, tally, write

SEED = 20261111
OUT = "data_work/frozen/data_v4"
V3 = "data_work/frozen/data_v3"
BFCL_TRAIN = "data_work/processed/bfcl_v4_train.jsonl"
BFCL_DEV = "data_work/processed/bfcl_v4_dev.jsonl"


def records_of(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    base = records_of(f"{V3}/train.jsonl")
    bfcl = records_of(BFCL_TRAIN)
    train = base + bfcl
    random.Random(SEED).shuffle(train)

    os.makedirs(OUT, exist_ok=True)
    shutil.copyfile(f"{V3}/dev.jsonl", f"{OUT}/dev.jsonl")
    shutil.copyfile(f"{V3}/dev_new.jsonl", f"{OUT}/dev_new.jsonl")
    shutil.copyfile(BFCL_DEV, f"{OUT}/dev_bfcl.jsonl")
    dev_families = set()
    for name in ("dev.jsonl", "dev_new.jsonl", "dev_bfcl.jsonl"):
        dev_families |= {record["family"] for record in records_of(f"{OUT}/{name}")}
    overlap = {record["family"] for record in train} & dev_families
    if overlap:
        sys.exit(f"families on both sides: {sorted(overlap)[:5]}")
    write(f"{OUT}/train.jsonl", train)
    manifest = {
        "seed": SEED,
        "made_from": {"data_v3": f"{V3}/train.jsonl", "bfcl_train": BFCL_TRAIN, "bfcl_dev": BFCL_DEV},
        "data_v3_train_sha256": sha256(f"{V3}/train.jsonl"), "bfcl_train_sha256": sha256(BFCL_TRAIN), "bfcl_dev_sha256": sha256(BFCL_DEV),
        "bfcl_multi_sha256": sha256("data_work/processed/bfcl_multi.jsonl"),
        "train_sha256": sha256(f"{OUT}/train.jsonl"), "dev_sha256": sha256(f"{OUT}/dev.jsonl"),
        "dev_new_sha256": sha256(f"{OUT}/dev_new.jsonl"), "dev_bfcl_sha256": sha256(f"{OUT}/dev_bfcl.jsonl"),
        "train": tally(train),
    }
    with open(f"{OUT}/manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    for source, count in manifest["train"].items():
        print(f"  {source:12s} {count}")
    print("train", manifest["train_sha256"][:16], "dev_bfcl", manifest["dev_bfcl_sha256"][:16])


if __name__ == "__main__":
    main()

"""Build and freeze data_v5: data_v3 plus the cleaned BFCL train records.

    python code/data/assemble_v5.py

Same as assemble_v4.py, with the BFCL records from prepare_bfcl_v5.py (whole public-collision
families removed, asks the state already answers not used as targets, numbers matched by value).
Every data_v3 train record is kept unchanged. The dev files:

- dev.jsonl, dev_new.jsonl: data_v3's, byte for byte (the same as data_v4's).
- dev_bfcl.jsonl: the same 124 held-out BFCL conversations as data_v4's, with the v5 targets.
"""
import assemble_v4 as A

A.OUT = "data_work/frozen/data_v5"
A.BFCL_TRAIN = "data_work/processed/bfcl_v5_train.jsonl"
A.BFCL_DEV = "data_work/processed/bfcl_v5_dev.jsonl"

if __name__ == "__main__":
    A.main()

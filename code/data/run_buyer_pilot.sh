#!/usr/bin/env bash
# Build the buyer pilot and check it. Run through code/env/run.sh.
set -e
python code/data/extract_buyer_interface.py
python code/data/gen_buyer.py --negotiations 300 --seed 1 --out data_work/processed/buyer_v0.jsonl
python code/data/check_buyer.py --self-test data_work/processed/buyer_v0.jsonl

#!/usr/bin/env bash
# Rebuild data_v0 from the processed sources and run every automatic check. Run through code/env/run.sh.
set -e
python code/data/gen_chains.py --conversations 300 --seed 1 --out data_work/processed/chains_v0.jsonl
python code/data/extract_buyer_interface.py
python code/data/gen_buyer.py --negotiations 300 --seed 1 --out data_work/processed/buyer_v0.jsonl
python code/data/check_buyer.py --self-test data_work/processed/buyer_v0.jsonl
python code/data/assemble_v0.py

# Split so each part gets the checks written for it: a buyer's counter-offer is a new number
# on purpose, so "reply numbers with no support" only means something for Part 1.
python - <<'PY'
import json
parts = {name: open(f"data_work/processed/data_v0_{name}.jsonl", "w", encoding="utf-8") for name in ("part1", "part2")}
with open("data_work/processed/data_v0.jsonl", encoding="utf-8") as f:
    for line in f:
        parts[json.loads(line)["task"]].write(line)
for out in parts.values():
    out.close()
PY
python code/data/qa_conversations.py --tokens data_work/processed/data_v0_part1.jsonl data_work/processed/data_v0_part2.jsonl
python code/data/check_buyer.py data_work/processed/data_v0_part2.jsonl

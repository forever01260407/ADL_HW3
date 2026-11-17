#!/usr/bin/env bash
# Build data_v2 on top of data_v1, freeze it and run every automatic check.
# Run through code/env/run.sh, after code/data/run_data_v1.sh.
set -euo pipefail
F=data_work/frozen/data_v2

python code/data/gen_policy.py --conversations 1600 --seed 3 --out data_work/processed/policy_v2.jsonl
python code/data/check_policy.py --self-test data_work/processed/policy_v2.jsonl
python code/data/gen_buyer.py --negotiations 2400 --seed 3 --spread-offers --out data_work/processed/buyer_v2.jsonl
python code/data/check_buyer.py --self-test data_work/processed/buyer_v2.jsonl
python -I code/data/dedupe_against_public.py --out data_work/processed/collisions_v2.json \
    data_work/processed/policy_v2.jsonl data_work/processed/buyer_v2.jsonl
python code/data/assemble_v2.py

python - <<'PY'
import json
F = "data_work/frozen/data_v2"
for side in ("train", "dev_new"):
    outs = {name: open(f"{F}/{side}_{name}.jsonl", "w", encoding="utf-8") for name in ("part1", "part2")}
    with open(f"{F}/{side}.jsonl", encoding="utf-8") as f:
        for line in f:
            outs[json.loads(line)["task"]].write(line)
    for out in outs.values():
        out.close()
PY
python code/data/qa_conversations.py --tokens "$F/train_part1.jsonl" "$F/dev_new_part1.jsonl" "$F/train_part2.jsonl" "$F/dev_new_part2.jsonl"
python code/data/check_buyer.py "$F/train_part2.jsonl"
python code/data/check_buyer.py "$F/dev_new_part2.jsonl"

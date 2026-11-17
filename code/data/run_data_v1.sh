#!/usr/bin/env bash
# Rebuild data_v1 from the processed sources, freeze it and run every automatic check.
# Run through code/env/run.sh. The third-party files come from code/data/convert_sources.py.
set -euo pipefail
F=data_work/frozen/data_v1

python code/data/extract_buyer_interface.py
python code/data/gen_chains.py --conversations 1500 --seed 2 --out data_work/processed/chains_v1.jsonl
python code/data/gen_buyer.py --negotiations 1300 --seed 2 --out data_work/processed/buyer_v1.jsonl
python code/data/check_buyer.py --self-test data_work/processed/buyer_v1.jsonl
python -I code/data/dedupe_against_public.py --out data_work/processed/collisions_v1.json \
    data_work/processed/glaive.jsonl data_work/processed/toolace.jsonl data_work/processed/hermes.jsonl \
    data_work/processed/chains_v1.jsonl data_work/processed/buyer_v1.jsonl
python code/data/assemble_v1.py

python - <<'PY'
import json
F = "data_work/frozen/data_v1"
for side in ("train", "dev"):
    outs = {name: open(f"{F}/{side}_{name}.jsonl", "w", encoding="utf-8") for name in ("part1", "part2")}
    with open(f"{F}/{side}.jsonl", encoding="utf-8") as f:
        for line in f:
            outs[json.loads(line)["task"]].write(line)
    for out in outs.values():
        out.close()
PY
python code/data/qa_conversations.py --tokens "$F/train_part1.jsonl" "$F/dev_part1.jsonl" "$F/train_part2.jsonl" "$F/dev_part2.jsonl"
python code/data/check_buyer.py "$F/train_part2.jsonl"
python code/data/check_buyer.py "$F/dev_part2.jsonl"

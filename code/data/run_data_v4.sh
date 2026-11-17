#!/usr/bin/env bash
# Build data_v4 = data_v3 + BFCL train records, and check it.
# Run through code/env/run.sh, after code/data/run_data_v3.sh and code/data/build_bfcl_multiturn.py.
set -euo pipefail
export HF_HUB_OFFLINE=1
echo "bfcl_multi.jsonl $(sha256sum data_work/processed/bfcl_multi.jsonl | cut -c1-16)"
python code/data/prepare_bfcl_v4.py
python code/data/qa_conversations.py data_work/processed/bfcl_v4_train.jsonl data_work/processed/bfcl_v4_dev.jsonl --tokens
python code/data/dedupe_against_public.py --out data_work/processed/collisions_bfcl_v4.json data_work/processed/bfcl_v4_train.jsonl data_work/processed/bfcl_v4_dev.jsonl
python code/data/assemble_v4.py

python - <<'PY'
import collections
import json
import sys

sys.path.insert(0, "code")
sys.path.insert(0, "code/data")
import model_setup as M
from common import step_kinds

F = "data_work/frozen/data_v4"
with open(f"{F}/train.jsonl", encoding="utf-8") as f:
    v4 = sorted(line for line in f if not json.loads(line)["source"].startswith("bfcl_"))
with open("data_work/frozen/data_v3/train.jsonl", encoding="utf-8") as f:
    v3 = sorted(f)
assert v4 == v3, "the non-BFCL records differ from data_v3"
print("data_v4 without BFCL equals data_v3")

steps = M.load_steps(f"{F}/train.jsonl")
old = M.load_steps("data_work/frozen/data_v3/train.jsonl")
assert len(old) == 22003, f"data_v3 now expands to {len(old)} steps"
count = collections.Counter(step["task"] for step in steps)
kinds = collections.Counter((step["source"], step_kinds(step["messages"])[-1]) for step in steps if step["source"].startswith("bfcl_"))
print("train steps", dict(count), "total", len(steps))
print("bfcl target steps", sorted(kinds.items()))
share = count["part2"] / len(steps)
print(f"part2 share {share:.4f}; one pass over both parts at batch 16 = {len(steps) / 16:.1f} updates")
PY

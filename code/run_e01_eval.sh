#!/usr/bin/env bash
# Evaluate one E01 checkpoint on its own training steps, through the official merge and generate path.
# Usage: bash code/run_e01_eval.sh <update>. Run through code/env/run.sh.
set -e
UPDATE="$1"
OWN=runs/E01/own
mkdir -p "$OWN"
[ -f "$OWN/steps.jsonl" ] || python code/eval_own.py export --config configs/e01.json --out "$OWN/steps.jsonl"
python hw3/eval.py generate --data "$OWN/steps.jsonl" --adapter "runs/E01/checkpoints/step_$UPDATE" \
  --merge-dir "runs/E01/merged/step_$UPDATE" --out "$OWN/step_$UPDATE.jsonl" --backend vllm
python code/eval_own.py score --steps "$OWN/steps.jsonl" --pred "$OWN/step_$UPDATE.jsonl" --out "$OWN/step_$UPDATE.metrics.json"

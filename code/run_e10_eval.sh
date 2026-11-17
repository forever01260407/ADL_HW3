#!/usr/bin/env bash
# Evaluate one training checkpoint on our dev set and on the fixed public samples,
# through the official merge and generate path.
#
#   bash code/env/run.sh bash code/run_e10_eval.sh <update> [run]
#
# With EXTRA_DEV set to a conversation file, that file is evaluated the same way, as dev_new.
# Every checkpoint uses the same id lists, so the scores join into one curve. They are samples,
# not the full public test. The Part 2 response metric needs the judge and is not run here.
set -euo pipefail

UPDATE="$1"
RUN="${2:-runs/E10}"
EVAL="$RUN/eval"
ADAPTER="$RUN/checkpoints/step_$UPDATE"
MERGED="$RUN/merged/step_$UPDATE"
TOOLS=hw3/data/public_test.jsonl.gz
NEGOTIATION=hw3/data/public_test_negotiation.jsonl.gz
mkdir -p "$EVAL"

[ -f "$EVAL/dev_steps.jsonl" ] || python code/eval_own.py export --data data_work/frozen/data_v1/dev.jsonl --out "$EVAL/dev_steps.jsonl"

run_generate() {  # out, then the data and id arguments
    local out="$1"
    shift
    python hw3/eval.py generate "$@" --adapter "$ADAPTER" --merge-dir "$MERGED" --out "$out" --backend vllm > "$out.log" 2>&1
}

run_generate "$EVAL/dev_$UPDATE.jsonl" --data "$EVAL/dev_steps.jsonl"
if [ -n "${EXTRA_DEV:-}" ]; then
    [ -f "$EVAL/dev_new_steps.jsonl" ] || python code/eval_own.py export --data "$EXTRA_DEV" --out "$EVAL/dev_new_steps.jsonl"
    run_generate "$EVAL/dev_new_$UPDATE.jsonl" --data "$EVAL/dev_new_steps.jsonl"
fi
run_generate "$EVAL/tools_$UPDATE.jsonl" --data "$TOOLS" --ids-file data_work/frozen/eval_ids_tools.txt
run_generate "$EVAL/negotiation_$UPDATE.jsonl" --data "$NEGOTIATION" --ids-file data_work/frozen/eval_ids_negotiation.txt

echo "===== dev, step $UPDATE"
python code/eval_own.py score --steps "$EVAL/dev_steps.jsonl" --pred "$EVAL/dev_$UPDATE.jsonl" --out "$EVAL/dev_$UPDATE.metrics.json" --show 0
if [ -n "${EXTRA_DEV:-}" ]; then
    echo "===== dev_new, step $UPDATE"
    python code/eval_own.py score --steps "$EVAL/dev_new_steps.jsonl" --pred "$EVAL/dev_new_$UPDATE.jsonl" --out "$EVAL/dev_new_$UPDATE.metrics.json" --show 0
fi
echo "===== public sample, step $UPDATE"
python code/eval_breakdown.py --pred "$EVAL/tools_$UPDATE.jsonl" --only-predicted --out "$EVAL/tools_$UPDATE.breakdown.json"
python hw3/eval.py score --data "$NEGOTIATION" --pred "$EVAL/negotiation_$UPDATE.jsonl" --only-predicted --out "$EVAL/negotiation_$UPDATE.metrics.json" | grep -m4 -E "replies|order  |deal volume|deal price"

# The merged model is rebuilt from the adapter when needed, so it need not stay on disk.
rm -rf "$MERGED"

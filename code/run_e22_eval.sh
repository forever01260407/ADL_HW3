#!/usr/bin/env bash
# Evaluate one adapter on the three data_v4 dev sets and the fixed public samples,
# through the official merge and generate path.
#
#   bash code/env/run.sh bash code/run_e22_eval.sh <update> [run]
#   ADAPTER=runs/E21/checkpoints/step_1375 EVAL=runs/E22/baseline_E21 SKIP_PUBLIC=1 bash code/env/run.sh bash code/run_e22_eval.sh 1375
#
# dev: data_v1's dev. dev_new: long-policy conversations from a generator no E21/E22 data comes
# from. dev_bfcl: held-out BFCL families, with and without the session state. Scores break Part 1
# down by source, step kind and where the call's values come from (code/eval_own.py --by-source).
# GPU_UTIL lowers vLLM's share of the GPU when a training run shares it.
set -euo pipefail

UPDATE="$1"
RUN="${2:-runs/E22}"
EVAL="${EVAL:-$RUN/eval}"
ADAPTER="${ADAPTER:-$RUN/checkpoints/step_$UPDATE}"
MERGED="$EVAL/merged_$UPDATE"
DATA="${DATA:-data_work/frozen/data_v4}"  # E23: data_v5 (same dev and dev_new; dev_bfcl with the v5 targets)
TOOLS=hw3/data/public_test.jsonl.gz
NEGOTIATION=hw3/data/public_test_negotiation.jsonl.gz
UTIL=()
[ -n "${GPU_UTIL:-}" ] && UTIL=(--gpu-util "$GPU_UTIL")
mkdir -p "$EVAL"

run_generate() {  # out, then the data and id arguments
    local out="$1"
    shift
    [ -s "$out" ] && return 0  # the official generate fails when every id is already done
    python hw3/eval.py generate "$@" --adapter "$ADAPTER" --merge-dir "$MERGED" --out "$out" --backend vllm "${UTIL[@]}" > "$out.log" 2>&1
}

for name in dev dev_new dev_bfcl; do
    [ -f "$EVAL/${name}_steps.jsonl" ] || python code/eval_own.py export --data "$DATA/$name.jsonl" --out "$EVAL/${name}_steps.jsonl"
    run_generate "$EVAL/${name}_$UPDATE.jsonl" --data "$EVAL/${name}_steps.jsonl"
done
if [ -z "${SKIP_PUBLIC:-}" ]; then
    run_generate "$EVAL/tools_$UPDATE.jsonl" --data "$TOOLS" --ids-file data_work/frozen/eval_ids_tools.txt
    run_generate "$EVAL/negotiation_$UPDATE.jsonl" --data "$NEGOTIATION" --ids-file data_work/frozen/eval_ids_negotiation.txt
fi
rm -rf "$MERGED"

for name in dev dev_new dev_bfcl; do
    echo "===== $name, step $UPDATE"
    python code/eval_own.py score --steps "$EVAL/${name}_steps.jsonl" --pred "$EVAL/${name}_$UPDATE.jsonl" --by-source --out "$EVAL/${name}_$UPDATE.metrics.json" --show 0
done
if [ -z "${SKIP_PUBLIC:-}" ]; then
    echo "===== public sample, step $UPDATE"
    python code/eval_breakdown.py --pred "$EVAL/tools_$UPDATE.jsonl" --only-predicted --out "$EVAL/tools_$UPDATE.breakdown.json"
    python hw3/eval.py score --data "$NEGOTIATION" --pred "$EVAL/negotiation_$UPDATE.jsonl" --only-predicted --out "$EVAL/negotiation_$UPDATE.metrics.json" | grep -m4 -E "replies|order  |deal volume|deal price"
fi

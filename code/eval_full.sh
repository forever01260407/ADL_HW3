#!/usr/bin/env bash
# Full official evaluation of one model on both public test sets.
#
#   bash code/env/run.sh bash code/eval_full.sh runs/E02
#   bash code/env/run.sh bash code/eval_full.sh runs/E10/ckpt-1000 runs/E10/checkpoints/ckpt-1000
#
# Every model needs its own output folder: generation skips ids that are already
# in the output file, so a shared folder would mix replies from different models.
# Rerunning with the same model and folder resumes where it stopped.
set -euo pipefail

RUN="$1"
ADAPTER="${2:-}"
TOOLS=hw3/data/public_test.jsonl.gz
NEGOTIATION=hw3/data/public_test_negotiation.jsonl.gz
TOOLS_PRED="$RUN/predictions/tools.jsonl"
NEGOTIATION_PRED="$RUN/predictions/negotiation.jsonl"
MERGED="$RUN/merged"

mkdir -p "$RUN/predictions" "$RUN/metrics" "$RUN/judge" "$RUN/logs"
touch "$RUN/timings.txt"

ADAPTER_ARGS=()
if [ -n "$ADAPTER" ]; then
    ADAPTER_ARGS=(--adapter "$ADAPTER" --merge-dir "$MERGED")
fi

timed() {
    local label="$1"
    shift
    local start=$SECONDS
    "$@" >> "$RUN/logs/$label.log" 2>&1
    echo "$label $((SECONDS - start))s" | tee -a "$RUN/timings.txt"
}

timed generate_tools python hw3/eval.py generate --data "$TOOLS" --out "$TOOLS_PRED" --backend vllm "${ADAPTER_ARGS[@]}"
python hw3/eval.py score --data "$TOOLS" --pred "$TOOLS_PRED" --out "$RUN/metrics/tools.json"
python code/eval_breakdown.py --pred "$TOOLS_PRED" --out "$RUN/metrics/tools_breakdown.json"

timed generate_negotiation python hw3/eval.py generate --data "$NEGOTIATION" --out "$NEGOTIATION_PRED" --backend vllm "${ADAPTER_ARGS[@]}"
python hw3/eval.py score --data "$NEGOTIATION" --pred "$NEGOTIATION_PRED" --out "$RUN/metrics/negotiation_no_judge.json"

python hw3/eval.py judge build --data "$NEGOTIATION" --pred "$NEGOTIATION_PRED" --out "$RUN/judge/requests.jsonl"
if [ -s "$RUN/judge/requests.jsonl" ]; then
    timed judge python hw3/eval.py judge run --requests "$RUN/judge/requests.jsonl" --out "$RUN/judge/judgments.jsonl" --gpu 0
    python hw3/eval.py score --data "$NEGOTIATION" --pred "$NEGOTIATION_PRED" \
        --judgments "$RUN/judge/judgments.jsonl" --out "$RUN/metrics/negotiation.json"
else
    echo "no reply has text to judge; response stays not computed"
fi

rm -rf "$MERGED"

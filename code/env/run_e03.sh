#!/usr/bin/env bash
# E03: does vLLM's V2 runner work under WSL, and does it agree with V1?
#
#   bash code/env/run.sh bash code/env/run_e03.sh [limit]
#
# Same slice and same model as E00 (no adapter), so the only intended change is the
# runner. E00 against E02 is the control: both ran on V1, so any difference between
# them comes from something other than the runner. The judge is not covered here.
set -uo pipefail

export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_WSL2_ENABLE_PIN_MEMORY=1

LIMIT="${1:-100}"
RUN=runs/E03
TOOLS=hw3/data/public_test.jsonl.gz
NEGOTIATION=hw3/data/public_test_negotiation.jsonl.gz

mkdir -p "$RUN/predictions" "$RUN/metrics" "$RUN/logs"
: > "$RUN/timings.txt"

generate() {
    local label="$1" data="$2"
    local start=$SECONDS
    python hw3/eval.py generate --data "$data" --out "$RUN/predictions/$label.jsonl" --backend vllm --limit "$LIMIT" \
        > "$RUN/logs/generate_$label.log" 2>&1
    local status=$?
    echo "generate_$label $((SECONDS - start))s exit=$status" | tee -a "$RUN/timings.txt"
    return $status
}

if ! generate tools "$TOOLS"; then
    echo "V2 failed on Part 1, last lines of the log:"
    tail -n 25 "$RUN/logs/generate_tools.log"
    exit 1
fi
generate negotiation "$NEGOTIATION" || tail -n 25 "$RUN/logs/generate_negotiation.log"

echo; echo "===== V2 scores"
python hw3/eval.py score --data "$TOOLS" --pred "$RUN/predictions/tools.jsonl" --only-predicted --out "$RUN/metrics/tools.json"
python code/eval_breakdown.py --pred "$RUN/predictions/tools.jsonl" --only-predicted

echo; echo "===== control: V1 against V1"
python code/env/compare_predictions.py runs/E00/predictions/tools_smoke.jsonl runs/E02/predictions/tools.jsonl
python code/env/compare_predictions.py runs/E00/predictions/negotiation_smoke.jsonl runs/E02/predictions/negotiation.jsonl

echo; echo "===== V1 against V2"
python code/env/compare_predictions.py runs/E00/predictions/tools_smoke.jsonl "$RUN/predictions/tools.jsonl"
python code/env/compare_predictions.py runs/E00/predictions/negotiation_smoke.jsonl "$RUN/predictions/negotiation.jsonl"
python code/env/compare_predictions.py runs/E02/predictions/tools.jsonl "$RUN/predictions/tools.jsonl"
python code/env/compare_predictions.py runs/E02/predictions/negotiation.jsonl "$RUN/predictions/negotiation.jsonl"

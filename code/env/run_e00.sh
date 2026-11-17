#!/usr/bin/env bash
# E00: run the whole official chain on a small slice, with no adapter.
#
#   bash code/env/run.sh bash code/env/run_e00.sh [limit]
#
# The scores are a smoke test of the chain, not a model result. --limit can cut a
# negotiation in half, so deal price and deal volume here mean nothing.
set -euo pipefail

LIMIT="${1:-100}"
RUN=runs/E00
TOOLS=hw3/data/public_test.jsonl.gz
NEGOTIATION=hw3/data/public_test_negotiation.jsonl.gz
TOOLS_PRED="$RUN/predictions/tools_smoke.jsonl"
NEGOTIATION_PRED="$RUN/predictions/negotiation_smoke.jsonl"

mkdir -p "$RUN/predictions" "$RUN/metrics" "$RUN/judge" "$RUN/logs"
: > "$RUN/timings.txt"

nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 > "$RUN/gpu_memory_mib.log" &
MONITOR=$!
trap 'kill "$MONITOR" 2>/dev/null || true' EXIT

timed() {
    local label="$1"
    shift
    local start=$SECONDS
    "$@" > "$RUN/logs/$label.log" 2>&1
    echo "$label $((SECONDS - start))s" | tee -a "$RUN/timings.txt"
}

timed generate_tools python hw3/eval.py generate --data "$TOOLS" --out "$TOOLS_PRED" --backend vllm --limit "$LIMIT"
timed generate_negotiation python hw3/eval.py generate --data "$NEGOTIATION" --out "$NEGOTIATION_PRED" --backend vllm --limit "$LIMIT"

python hw3/eval.py score --data "$TOOLS" --pred "$TOOLS_PRED" --only-predicted --out "$RUN/metrics/tools_smoke.json"
python hw3/eval.py score --data "$NEGOTIATION" --pred "$NEGOTIATION_PRED" --only-predicted --out "$RUN/metrics/negotiation_smoke_no_judge.json"

python - "$TOOLS_PRED" "$NEGOTIATION_PRED" <<'EOF'
import collections
import json
import sys

for path in sys.argv[1:]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    reasons = collections.Counter(row["finish_reason"] for row in rows)
    print(path, dict(reasons))
EOF

timed judge_build python hw3/eval.py judge build --data "$NEGOTIATION" --pred "$NEGOTIATION_PRED" --out "$RUN/judge/requests.jsonl"
cat "$RUN/logs/judge_build.log"

REQUESTS=$(wc -l < "$RUN/judge/requests.jsonl")
echo "judge requests: $REQUESTS"
if [ "$REQUESTS" -gt 0 ]; then
    timed judge_run python hw3/eval.py judge run --requests "$RUN/judge/requests.jsonl" --out "$RUN/judge/judgments.jsonl" --gpu 0
    python hw3/eval.py score --data "$NEGOTIATION" --pred "$NEGOTIATION_PRED" --only-predicted \
        --judgments "$RUN/judge/judgments.jsonl" --out "$RUN/metrics/negotiation_smoke.json"
fi

echo "peak gpu memory: $(sort -n "$RUN/gpu_memory_mib.log" | tail -1) MiB"

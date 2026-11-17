#!/usr/bin/env bash
# STEP 07: walk the E01 adapter through the grader's path and measure what the runner changes.
#
#   bash code/env/run.sh bash code/run_e01_compat.sh
#
# The scores here are from an adapter trained on 56 examples, for checking the path only.
set -euo pipefail

RUN=runs/E01
OUT=$RUN/compat
ADAPTER=$RUN/checkpoints/step_200
MERGED=$RUN/merged/step_200
TOOLS=hw3/data/public_test.jsonl.gz
NEGOTIATION=hw3/data/public_test_negotiation.jsonl.gz
TOOL_IDS=data_work/frozen/eval_ids_tools.txt
NEGOTIATION_IDS=data_work/frozen/eval_ids_negotiation.txt
mkdir -p "$OUT"

generate() {  # data ids out [extra args]
    local data="$1" ids="$2" out="$3"
    shift 3
    python hw3/eval.py generate --data "$data" --ids-file "$ids" --adapter "$ADAPTER" --merge-dir "$MERGED" \
        --out "$out" --backend vllm "$@" > "$out.log" 2>&1
}

echo "===== adapter check"
python code/check_adapter.py "$ADAPTER" "$MERGED" 2>/dev/null

echo "===== V1 on the fixed samples"
time generate "$TOOLS" "$TOOL_IDS" "$OUT/tools_v1.jsonl"
time generate "$NEGOTIATION" "$NEGOTIATION_IDS" "$OUT/negotiation_v1.jsonl"
python hw3/eval.py score --data "$TOOLS" --pred "$OUT/tools_v1.jsonl" --only-predicted --out "$OUT/tools_v1.metrics.json"
python code/eval_breakdown.py --pred "$OUT/tools_v1.jsonl" --only-predicted
python hw3/eval.py score --data "$NEGOTIATION" --pred "$OUT/negotiation_v1.jsonl" --only-predicted --out "$OUT/negotiation_v1.metrics.json"

echo "===== 8 GB memory budget on the longest prompts"
python - "$TOOLS" "$NEGOTIATION" "$OUT/longest_ids.txt" <<'EOF'
import gzip, json, sys
rows = []
for path in sys.argv[1:3]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows.append((len(json.dumps(row["messages"])) + len(json.dumps(row["tools"])), row["id"]))
with open(sys.argv[3], "w") as f:
    for _, row_id in sorted(rows, reverse=True)[:64]:
        f.write(row_id + "\n")
EOF
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 > "$OUT/budget_gpu.txt" &
WATCH=$!
time python hw3/eval.py generate --data "$TOOLS" "$NEGOTIATION" --ids-file "$OUT/longest_ids.txt" --adapter "$ADAPTER" --merge-dir "$MERGED" \
    --out "$OUT/budget.jsonl" --backend vllm --gpu-util 0.28 > "$OUT/budget.log" 2>&1
kill "$WATCH"
echo "peak GPU MiB during the budget test: $(sort -n "$OUT/budget_gpu.txt" | tail -n 1)"
grep -m1 "prompt tokens" "$OUT/budget.log"
grep -m1 "GPU KV cache size" "$OUT/budget.log" || true
python - "$OUT/budget.jsonl" <<'EOF'
import collections, json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    print("budget test finish reasons", collections.Counter(json.loads(line)["finish_reason"] for line in f))
EOF

echo "===== V2, twice"
(
    export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_WSL2_ENABLE_PIN_MEMORY=1
    for round in 1 2; do
        generate "$TOOLS" "$TOOL_IDS" "$OUT/tools_v2_$round.jsonl"
        generate "$NEGOTIATION" "$NEGOTIATION_IDS" "$OUT/negotiation_v2_$round.jsonl"
    done
)
grep -m1 -c "Using V2 Model Runner" "$OUT/tools_v2_1.jsonl.log" || true

echo "===== outputs compared"
for name in tools negotiation; do
    echo "--- $name: V1 against V2 first run"
    python code/env/compare_predictions.py "$OUT/${name}_v1.jsonl" "$OUT/${name}_v2_1.jsonl"
    echo "--- $name: V2 first run against V2 second run"
    python code/env/compare_predictions.py "$OUT/${name}_v2_1.jsonl" "$OUT/${name}_v2_2.jsonl"
done

echo "===== scores"
for pred in tools_v1 tools_v2_1 tools_v2_2; do
    echo "--- $pred"
    python code/eval_breakdown.py --pred "$OUT/$pred.jsonl" --only-predicted | head -n 1
done
for pred in negotiation_v1 negotiation_v2_1 negotiation_v2_2; do
    echo "--- $pred"
    python hw3/eval.py score --data "$NEGOTIATION" --pred "$OUT/$pred.jsonl" --only-predicted | grep -E "order  |deal volume|deal price|replies"
done

#!/usr/bin/env bash
# STEP 13: freeze the final adapter and run every final check on the frozen copy.
#
#   bash code/env/run.sh bash code/run_final.sh runs/E21/checkpoints/step_1375
#
# The copy holds only the two files the submission needs. Every check below reads the copy,
# never the training checkpoint: the official merge, the adapter check, the 8 GB budget test,
# the full public run with the V1 runner and the judge, and a full generation with V2.
set -euo pipefail

SOURCE="$1"
FINAL=release/final/adapter_checkpoint
OUT=runs/FINAL
TOOLS=hw3/data/public_test.jsonl.gz
NEGOTIATION=hw3/data/public_test_negotiation.jsonl.gz
if [ -e "$FINAL" ]; then
    echo "$FINAL already exists; move it away before freezing another adapter" >&2
    exit 1
fi
mkdir -p "$FINAL" "$OUT"

echo "===== freeze"
cp "$SOURCE/adapter_config.json" "$SOURCE/adapter_model.safetensors" "$FINAL/"
echo "source $SOURCE" > "$OUT/sha256.txt"
sha256sum "$SOURCE/adapter_config.json" "$SOURCE/adapter_model.safetensors" "$FINAL/adapter_config.json" "$FINAL/adapter_model.safetensors" | tee -a "$OUT/sha256.txt"
ls -l "$FINAL"

echo "===== 8 GB memory budget on the longest prompts, through the official merge"
cp runs/E01/compat/longest_ids.txt "$OUT/longest_ids.txt"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 > "$OUT/budget_gpu.txt" &
WATCH=$!
python hw3/eval.py generate --data "$TOOLS" "$NEGOTIATION" --ids-file "$OUT/longest_ids.txt" --adapter "$FINAL" --merge-dir "$OUT/merged" \
    --out "$OUT/budget.jsonl" --backend vllm --gpu-util 0.28 > "$OUT/budget.log" 2>&1
kill "$WATCH"
echo "idle GPU MiB before the test: $(head -n 1 "$OUT/budget_gpu.txt"); peak: $(sort -n "$OUT/budget_gpu.txt" | tail -n 1)"
grep -m1 "prompt tokens" "$OUT/budget.log" || true
python - "$OUT/budget.jsonl" <<'EOF'
import collections, json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    print("budget test finish reasons", collections.Counter(json.loads(line)["finish_reason"] for line in f))
EOF

echo "===== adapter check on the frozen copy and its official merge"
python code/check_adapter.py "$FINAL" "$OUT/merged" 2>/dev/null
rm -rf "$OUT/merged"

echo "===== full public, V1 runner, with the judge"
bash code/eval_full.sh "$OUT/v1" "$FINAL"
grep -h -E "HEADLINE weighted|missing predictions|order  |deal volume|deal price|response " "$OUT/v1/logs/"*.log 2>/dev/null | sort -u || true

echo "===== full public generation, V2 runner"
(
    export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_WSL2_ENABLE_PIN_MEMORY=1
    mkdir -p "$OUT/v2"
    python hw3/eval.py generate --data "$TOOLS" --out "$OUT/v2/tools.jsonl" --adapter "$FINAL" --merge-dir "$OUT/v2/merged" --backend vllm > "$OUT/v2/tools.log" 2>&1
    python hw3/eval.py generate --data "$NEGOTIATION" --out "$OUT/v2/negotiation.jsonl" --adapter "$FINAL" --merge-dir "$OUT/v2/merged" --backend vllm > "$OUT/v2/negotiation.log" 2>&1
    grep -m1 -c "Using V2 Model Runner" "$OUT/v2/tools.log" || true
    rm -rf "$OUT/v2/merged"
)
python hw3/eval.py score --data "$TOOLS" --pred "$OUT/v2/tools.jsonl" --out "$OUT/v2/tools.metrics.json" | grep -E "HEADLINE weighted|missing predictions" | sort -u
python hw3/eval.py score --data "$NEGOTIATION" --pred "$OUT/v2/negotiation.jsonl" --out "$OUT/v2/negotiation.metrics.json" | grep -E "order  |deal volume|deal price|missing predictions" | sort -u

echo "===== V1 against V2"
python code/env/compare_predictions.py "$OUT/v1/predictions/tools.jsonl" "$OUT/v2/tools.jsonl"
python code/env/compare_predictions.py "$OUT/v1/predictions/negotiation.jsonl" "$OUT/v2/negotiation.jsonl"
echo FINAL_DONE

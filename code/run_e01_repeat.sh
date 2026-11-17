#!/usr/bin/env bash
# Repeat the V1 run of E01 on the same ids and show the GPU memory baseline of the budget test.
# Usage: bash code/env/run.sh bash code/run_e01_repeat.sh
set -euo pipefail

OUT=runs/E01/compat
for name in tools negotiation; do
    if [ "$name" = tools ]; then
        DATA=hw3/data/public_test.jsonl.gz; IDS=data_work/frozen/eval_ids_tools.txt
    else
        DATA=hw3/data/public_test_negotiation.jsonl.gz; IDS=data_work/frozen/eval_ids_negotiation.txt
    fi
    python hw3/eval.py generate --data "$DATA" --ids-file "$IDS" --adapter runs/E01/checkpoints/step_200 \
        --merge-dir runs/E01/merged/step_200 --out "$OUT/${name}_v1_repeat.jsonl" --backend vllm > "$OUT/${name}_v1_repeat.log" 2>&1
    python code/env/compare_predictions.py "$OUT/${name}_v1.jsonl" "$OUT/${name}_v1_repeat.jsonl"
done

python - "$OUT/budget_gpu.txt" <<'EOF'
import sys
with open(sys.argv[1]) as f:
    values = [int(line) for line in f if line.strip()]
print(f"budget test GPU MiB: first {values[0]}, min {min(values)}, max {max(values)}, samples {len(values)}")
EOF
echo "GPU now, idle: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"

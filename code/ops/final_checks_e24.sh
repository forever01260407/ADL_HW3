#!/usr/bin/env bash
# E24 STEP 10 for the selected candidate (E23 copy): adapter check, one more full public Part 1
# generation (same model, same settings, new folder), 8 GB budget approximation on the longest
# prompts with the official merge. Every GPU step runs under run_supervised.py. Stops at the
# first failure. Run from the project root through code/env/run.sh.
#
#   bash code/env/run.sh bash code/ops/final_checks_e24.sh runs/E24_session_X release/candidates/X/adapter_checkpoint
set -euo pipefail
S="$1"
CAND="$2"
SUP="python code/ops/run_supervised.py --session $S --poll 15"
TOOLS=hw3/data/public_test.jsonl.gz
NEG=hw3/data/public_test_negotiation.jsonl.gz
mkdir -p "$S/final_repeat" "$S/compat"

echo "== adapter check (fresh Base)"
python code/check_adapter.py "$CAND" > "$S/compat/check_adapter.txt" 2>&1
tail -3 "$S/compat/check_adapter.txt"

echo "== full public Part 1, second generation of this model"
$SUP --name final_repeat_tools --gpu --progress "lines:$S/final_repeat/tools.jsonl" --init-grace 900 --stall 600 --max-seconds 2400 --expect-seconds 900 \
    --done-check "[\"python\", \"code/audit/check_predictions.py\", \"--data\", \"$TOOLS\", \"--pred\", \"$S/final_repeat/tools.jsonl\"]" \
    -- python hw3/eval.py generate --data "$TOOLS" --adapter "$CAND" --merge-dir "$S/final_repeat/merged" --out "$S/final_repeat/tools.jsonl" --backend vllm
python hw3/eval.py score --data "$TOOLS" --pred "$S/final_repeat/tools.jsonl" --out "$S/final_repeat/tools.metrics.json" > "$S/final_repeat/score.txt" 2>&1
grep -m2 "HEADLINE" "$S/final_repeat/score.txt" || true
python code/check_adapter.py "$CAND" "$S/final_repeat/merged" > "$S/compat/check_adapter_merged.txt" 2>&1
tail -3 "$S/compat/check_adapter_merged.txt"

echo "== 8 GB budget approximation (longest 64 prompts of both public files, gpu-util 0.28)"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 > "$S/compat/budget_gpu.txt" &
WATCH=$!
set +e
$SUP --name budget_test --gpu --progress "lines:$S/compat/budget.jsonl" --init-grace 900 --stall 600 --max-seconds 1500 --expect-seconds 600 \
    --done-check "[\"python\", \"code/audit/check_predictions.py\", \"--data\", \"$S/compat/longest_rows.jsonl\", \"--pred\", \"$S/compat/budget.jsonl\"]" \
    -- python hw3/eval.py generate --data "$TOOLS" "$NEG" --ids-file runs/E01/compat/longest_ids.txt --adapter "$CAND" --merge-dir "$S/final_repeat/merged" \
    --out "$S/compat/budget.jsonl" --backend vllm --gpu-util 0.28
CODE=$?
set -e
kill "$WATCH"
echo "idle GPU MiB: $(head -n 1 "$S/compat/budget_gpu.txt"); peak: $(sort -n "$S/compat/budget_gpu.txt" | tail -n 1)"
python - "$S/compat/budget.jsonl" <<'EOF'
import collections, json, sys
print("budget finish reasons", collections.Counter(json.loads(l)["finish_reason"] for l in open(sys.argv[1])))
EOF
case "$(realpath "$S/final_repeat/merged")" in "$(realpath "$S")"/*) rm -rf "$S/final_repeat/merged" ;; esac
exit $CODE

#!/usr/bin/env bash
# E30 fast path: train from E28, then the official generation and scoring on both public files (no judge).
#
#   bash code/env/run.sh bash code/ops/e30_run.sh runs/E30_session_X
#
# Every GPU step runs under code/ops/run_supervised.py (session + global GPU lock, deadline D_gpu, low-disk stop).
# Public numbers are used only to let the user choose between the finished E28 and E30 adapters.
set -euo pipefail
S="$1"
BASE=runs/E28_session_20261109_182859/E27/checkpoints/step_250
U=$(python -c "import json;print(json.load(open('$S/E30_config.json'))['max_updates'])")
SUP=(python code/ops/run_supervised.py --session "$S" --gpu --deadline-key D_gpu --poll 15 --done-timeout 120)

"${SUP[@]}" --name train_E30 --progress "update:$S/E30/train_log.jsonl" --max-seconds 1500 --expect-seconds 900 --stall 600 --init-grace 600 \
  -- python code/train.py --config "$S/E30_config.json" --run "$S/E30" --init-adapter "$BASE"
CK="$S/E30/checkpoints/step_$U"
test -f "$CK/done.json"

O="$S/public"
mkdir -p "$O/predictions" "$O/metrics" "$O/logs"
"${SUP[@]}" --name public_tools --progress "lines:$O/predictions/tools.jsonl" --max-seconds 1200 --expect-seconds 480 --stall 300 --init-grace 600 \
  --done-check "[\"python\", \"code/audit/check_predictions.py\", \"--data\", \"hw3/data/public_test.jsonl.gz\", \"--pred\", \"$O/predictions/tools.jsonl\"]" \
  -- python hw3/eval.py generate --data hw3/data/public_test.jsonl.gz --out "$O/predictions/tools.jsonl" --backend vllm --adapter "$CK" --merge-dir "$O/merged_tools"
python hw3/eval.py score --data hw3/data/public_test.jsonl.gz --pred "$O/predictions/tools.jsonl" --out "$O/metrics/tools.json" > "$O/logs/score_tools.log" 2>&1
python -c "import json;t=json.load(open('$O/metrics/tools.json'))['headline'];print('E30 PUBLIC P1', round(100*t['with_first_step_rule']['all']['weighted_step_acc'],2))" | tee "$O/p1.txt"
rm -rf "$O/merged_tools"

"${SUP[@]}" --name public_negotiation --progress "lines:$O/predictions/negotiation.jsonl" --max-seconds 1200 --expect-seconds 420 --stall 300 --init-grace 600 \
  --done-check "[\"python\", \"code/audit/check_predictions.py\", \"--data\", \"hw3/data/public_test_negotiation.jsonl.gz\", \"--pred\", \"$O/predictions/negotiation.jsonl\"]" \
  -- python hw3/eval.py generate --data hw3/data/public_test_negotiation.jsonl.gz --out "$O/predictions/negotiation.jsonl" --backend vllm --adapter "$CK" --merge-dir "$O/merged_neg"
python hw3/eval.py score --data hw3/data/public_test_negotiation.jsonl.gz --pred "$O/predictions/negotiation.jsonl" --out "$O/metrics/negotiation_no_judge.json" > "$O/logs/score_neg.log" 2>&1
rm -rf "$O/merged_neg"
echo "E30 DONE"

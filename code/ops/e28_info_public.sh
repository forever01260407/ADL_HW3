#!/bin/bash
# E28 information-only full public evaluation (a deviation from the frozen E28 plan; see records/e28_*_results.md).
# Why: the frozen selection compares buyer fill on our own buyer_select_new set, which E27 showed does not track
# public (E23 fills 48% there and 92% on public). E28 fills 43% there, so the controller keeps E23 without a public run.
# This script waits for the controller's handoff.json, then runs the same full public evaluation under the supervisor
# (session + global GPU locks, D_gpu 20:00). No adoption is automatic: the user decides from the numbers.
#   bash code/ops/e28_info_public.sh runs/E28_session_X
set -u
S="$1"
cd /home/hw3/HW3_ADL
until [ -f "$S/handoff.json" ]; do sleep 5; done
O="$S/info_public"
CHECK="python code/audit/check_predictions.py --data hw3/data/public_test.jsonl.gz --pred $O/predictions/tools.jsonl && python code/audit/check_predictions.py --data hw3/data/public_test_negotiation.jsonl.gz --pred $O/predictions/negotiation.jsonl && test -f $O/metrics/tools.json && test -f $O/metrics/negotiation.json"
python code/ops/run_supervised.py --session "$S" --name info_public --gpu --progress none --deadline-key D_gpu \
  --max-seconds 3000 --expect-seconds 1900 --stall 600 --init-grace 900 --poll 15 --done-timeout 120 \
  --done-check "$(python -c 'import json,sys; print(json.dumps(["bash","-c",sys.argv[1]]))' "$CHECK")" \
  -- bash code/eval_full.sh "$O" "$S/E27/checkpoints/step_250"

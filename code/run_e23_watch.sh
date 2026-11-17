#!/usr/bin/env bash
# Evaluate every checkpoint of the given runs as soon as training saves it, one at a time, on the
# three data_v5 dev sets and the fixed public samples (code/run_e22_eval.sh).
#
#   RUNS="runs/E23" HOURS=3 bash code/env/run.sh bash code/run_e23_watch.sh
#
# A node is done when its dev_bfcl metrics exist, so the script can be restarted. Run it next to
# ONE training at most: two trainings at once filled the 24 GB GPU and both stalled (2026-11-08).
#
# Stop conditions, so it never waits forever:
# - a checkpoint is missing and no training process for that run is alive: exit 1;
# - the deadline (HOURS from start, default 3) passes: exit 2;
# - an evaluation fails: exit 3.
set -euo pipefail
export DATA=data_work/frozen/data_v5 GPU_UTIL="${GPU_UTIL:-0.4}"
RUNS="${RUNS:-runs/E23 runs/E23b}"
DEADLINE=$(( $(date +%s) + ${HOURS:-3} * 3600 ))

check_deadline() {
    if [ "$(date +%s)" -gt "$DEADLINE" ]; then echo "$(date +%T) STOP: deadline passed ($1)"; exit 2; fi
}

# An evaluation started by an earlier copy of this script may still be running.
while pgrep -f "^bash code/run_e22_eval.sh" > /dev/null; do check_deadline "earlier evaluation still running"; sleep 30; done

for update in 148 369 738 1106 1475; do
    for run in $RUNS; do
        while [ ! -f "$run/checkpoints/step_$update/adapter_model.safetensors" ]; do
            if ! pgrep -f "^python code/train.py .*--run $run( |\$)" > /dev/null; then
                echo "$(date +%T) STOP: $run has no step_$update and its training is not running"; exit 1
            fi
            check_deadline "waiting for $run step_$update"
            sleep 60
        done
        sleep 30  # let the save finish
        [ -f "$run/eval/dev_bfcl_$update.metrics.json" ] && continue
        echo "$(date +%T) eval $run $update"
        if ! bash code/run_e22_eval.sh "$update" "$run" > "$run/eval_$update.log" 2>&1; then
            echo "$(date +%T) STOP: evaluation failed for $run $update"; exit 3
        fi
        echo "$(date +%T) done $run $update"
    done
done
echo "$(date +%T) all nodes evaluated"

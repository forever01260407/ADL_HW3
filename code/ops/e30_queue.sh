#!/usr/bin/env bash
# Start code/ops/e30_run.sh for session $2 only after the e30_run.sh of session $1 has exited and the GPU lock is free.
#   bash code/env/run.sh bash code/ops/e30_queue.sh runs/E30_session_A runs/E30_session_B
set -u
PREV="$1"
NEXT="$2"
while ps -eo args | grep -v grep | grep -q "e30_run.sh $PREV"; do sleep 10; done
until flock -n runs/.gpu_global.lock true; do sleep 10; done
echo "queue: $PREV finished, GPU free $(date +%T), starting $NEXT"
exec bash code/ops/e30_run.sh "$NEXT"

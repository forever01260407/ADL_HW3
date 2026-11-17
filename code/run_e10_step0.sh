#!/usr/bin/env bash
# Score the unmodified Base on the same fixed public samples as the E10 checkpoints.
#
#   bash code/env/run.sh bash code/run_e10_step0.sh [run]
#
# This reuses the full E02 predictions, so step 0 of the curve needs no new generation.
set -euo pipefail

RUN="${1:-runs/E10}"
EVAL="$RUN/eval"
TOOLS=hw3/data/public_test.jsonl.gz
NEGOTIATION=hw3/data/public_test_negotiation.jsonl.gz
mkdir -p "$EVAL"

keep_ids() {  # predictions, id list, output
    python - "$1" "$2" "$3" <<'PY'
import json
import sys

predictions, id_list, out = sys.argv[1:]
with open(id_list, encoding="utf-8") as f:
    wanted = {line.strip() for line in f if line.strip()}
written = 0
with open(predictions, encoding="utf-8") as f, open(out, "w", encoding="utf-8") as g:
    for line in f:
        if json.loads(line)["id"] in wanted:
            g.write(line)
            written += 1
if written != len(wanted):
    sys.exit(f"{out}: {written} predictions for {len(wanted)} ids")
print(f"{written} predictions -> {out}")
PY
}

keep_ids runs/E02/predictions/tools.jsonl data_work/frozen/eval_ids_tools.txt "$EVAL/tools_0.jsonl"
keep_ids runs/E02/predictions/negotiation.jsonl data_work/frozen/eval_ids_negotiation.txt "$EVAL/negotiation_0.jsonl"

echo "===== public sample, step 0"
python code/eval_breakdown.py --pred "$EVAL/tools_0.jsonl" --only-predicted --out "$EVAL/tools_0.breakdown.json"
python hw3/eval.py score --data "$NEGOTIATION" --pred "$EVAL/negotiation_0.jsonl" --only-predicted --out "$EVAL/negotiation_0.metrics.json" | grep -m4 -E "replies|order  |deal volume|deal price"

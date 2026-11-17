#!/usr/bin/env bash
# E00, judge part: run the official judge, then measure its speed.
#
#   bash code/env/run.sh bash code/env/run_e00_judge.sh
#
# The Base model gives too few text replies to measure speed, so the second half
# judges fixed placeholder sentences. Those labels are for timing only and are
# kept in their own folder; they must never be read as a response score.
set -euo pipefail

RUN=runs/E00
NEGOTIATION=hw3/data/public_test_negotiation.jsonl.gz
TIMING="$RUN/judge_timing"
TIMING_PRED="$TIMING/placeholder_replies.jsonl"
mkdir -p "$TIMING"

stamp() {
    while IFS= read -r line; do
        printf '%s %s\n' "$(date +%T)" "$line"
    done
}

echo "=== official judge on the E00 replies"
python hw3/eval.py judge run --requests "$RUN/judge/requests.jsonl" --out "$RUN/judge/judgments.jsonl" --gpu 0 \
    2>&1 | stamp > "$RUN/logs/judge_run.log"
grep -E "judged |RuntimeError" "$RUN/logs/judge_run.log" | tail -5
python hw3/eval.py score --data "$NEGOTIATION" --pred "$RUN/predictions/negotiation_smoke.jsonl" --only-predicted \
    --judgments "$RUN/judge/judgments.jsonl" --out "$RUN/metrics/negotiation_smoke.json"

echo "=== judge speed on placeholder replies"
python - "$NEGOTIATION" "$TIMING_PRED" <<'EOF'
import gzip
import json
import sys

SENTENCES = [
    "That is a bit more than I hoped to spend. Would you take a little less?",
    "Thanks for the details. Can you do better on the price if I decide today?",
    "I like it, but money is tight this month. Is there any room to move?",
    "Could you tell me a little more about its condition before I decide?",
]
with gzip.open(sys.argv[1], "rt", encoding="utf-8") as src, open(sys.argv[2], "w", encoding="utf-8") as out:
    for i, line in enumerate(src):
        if i == 128:
            break
        row = {"id": json.loads(line)["id"], "tool_calls": [], "text": f"{SENTENCES[i % 4]} (note {i})"}
        out.write(json.dumps(row) + "\n")
EOF
python hw3/eval.py judge build --data "$NEGOTIATION" --pred "$TIMING_PRED" --out "$TIMING/requests.jsonl"
python hw3/eval.py judge run --requests "$TIMING/requests.jsonl" --out "$TIMING/judgments.jsonl" --gpu 0 --chunk 64 \
    2>&1 | stamp > "$TIMING/run.log"
grep -E "requests |judged |RuntimeError" "$TIMING/run.log" | cut -c1-120

python - "$TIMING/judgments.jsonl" <<'EOF'
import collections
import json
import statistics
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]
tokens = [row["n_tokens"] for row in rows]
print("judged", len(rows), "| tokens mean", round(statistics.mean(tokens)), "max", max(tokens), "| total", sum(tokens))
print("finish", dict(collections.Counter(row["finish_reason"] for row in rows)))
print("unparsed", sum(row["label"] == "unparsed" for row in rows))
EOF

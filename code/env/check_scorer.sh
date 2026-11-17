#!/usr/bin/env bash
# Scorer self-check: feeding the gold answers back must give a perfect score.
#
#   bash code/env/run.sh bash code/env/check_scorer.sh
set -euo pipefail

mkdir -p runs/E00/metrics
python hw3/eval.py score --data hw3/data/public_test.jsonl.gz --gold-echo \
    --out runs/E00/metrics/gold_echo_tools.json
python hw3/eval.py score --data hw3/data/public_test_negotiation.jsonl.gz --gold-echo \
    --out runs/E00/metrics/gold_echo_negotiation.json

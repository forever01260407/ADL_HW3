#!/usr/bin/env bash
# E03 control: repeat the same 100-step slice on each runner.
#
#   bash code/env/run.sh bash code/env/run_e03_repeat.sh [limit]
#
# E00 against E02 differed although both used V1, but E02 also sent a much larger
# set of prompts. Repeating the identical slice separates "the batch contents change
# the output" from "every run differs".
set -euo pipefail

LIMIT="${1:-100}"
RUN=runs/E03
TOOLS=hw3/data/public_test.jsonl.gz
NEGOTIATION=hw3/data/public_test_negotiation.jsonl.gz

generate() {
    local dir="$1"
    mkdir -p "$RUN/$dir"
    python hw3/eval.py generate --data "$TOOLS" --out "$RUN/$dir/tools.jsonl" --backend vllm --limit "$LIMIT" > "$RUN/$dir/tools.log" 2>&1
    python hw3/eval.py generate --data "$NEGOTIATION" --out "$RUN/$dir/negotiation.jsonl" --backend vllm --limit "$LIMIT" > "$RUN/$dir/negotiation.log" 2>&1
}

generate repeat_v1
(
    export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_WSL2_ENABLE_PIN_MEMORY=1
    generate repeat_v2
)

echo "===== same runner, same slice"
python code/env/compare_predictions.py runs/E00/predictions/tools_smoke.jsonl "$RUN/repeat_v1/tools.jsonl"
python code/env/compare_predictions.py runs/E00/predictions/negotiation_smoke.jsonl "$RUN/repeat_v1/negotiation.jsonl"
python code/env/compare_predictions.py "$RUN/predictions/tools.jsonl" "$RUN/repeat_v2/tools.jsonl"
python code/env/compare_predictions.py "$RUN/predictions/negotiation.jsonl" "$RUN/repeat_v2/negotiation.jsonl"

echo "===== every run of the slice through the official scorer"
mkdir -p "$RUN/control"
python - <<'EOF'
import json

for name, smoke in (("tools", "tools_smoke"), ("negotiation", "negotiation_smoke")):
    with open(f"runs/E00/predictions/{smoke}.jsonl", encoding="utf-8") as f:
        ids = {json.loads(line)["id"] for line in f}
    with open(f"runs/E02/predictions/{name}.jsonl", encoding="utf-8") as f:
        rows = [line for line in f if json.loads(line)["id"] in ids]
    with open(f"runs/E03/control/e02_{name}.jsonl", "w", encoding="utf-8") as f:
        f.writelines(rows)
EOF

for pred in runs/E00/predictions/tools_smoke.jsonl "$RUN/repeat_v1/tools.jsonl" "$RUN/control/e02_tools.jsonl" \
            "$RUN/predictions/tools.jsonl" "$RUN/repeat_v2/tools.jsonl"; do
    echo "--- $pred"
    python code/eval_breakdown.py --pred "$pred" --only-predicted | head -n 1
    python hw3/eval.py score --data "$TOOLS" --pred "$pred" --only-predicted | grep -m1 "capped"
done

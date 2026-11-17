#!/usr/bin/env bash
# Build data_v3 from the frozen data_v1 and data_v2, and confirm it holds exactly their records.
# Run through code/env/run.sh, after code/data/run_data_v2.sh.
set -euo pipefail
python code/data/assemble_v3.py

python - <<'PY'
import json
F = "data_work/frozen/data_v3"
outs = {name: open(f"{F}/train_{name}.jsonl", "w", encoding="utf-8") for name in ("part1", "part2")}
with open(f"{F}/train.jsonl", encoding="utf-8") as f:
    for line in f:
        outs[json.loads(line)["task"]].write(line)
for out in outs.values():
    out.close()


def lines(path, task=None):
    with open(path, encoding="utf-8") as f:
        return sorted(line for line in f if task is None or json.loads(line)["task"] == task)


assert lines(f"{F}/train_part1.jsonl") == lines("data_work/frozen/data_v1/train.jsonl", "part1"), "Part 1 differs from data_v1"
assert lines(f"{F}/train_part2.jsonl") == lines("data_work/frozen/data_v2/train.jsonl", "part2"), "Part 2 differs from data_v2"
print("data_v3 Part 1 equals data_v1 Part 1; Part 2 equals data_v2 Part 2")
PY
python code/data/check_buyer.py data_work/frozen/data_v3/train_part2.jsonl

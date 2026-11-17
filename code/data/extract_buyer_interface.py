"""Take the fixed parts of the Part 2 prompt from the public file.

    python code/data/extract_buyer_interface.py

Writes data_work/processed/buyer_interface.json with the tool list and the lines of the
system message that are identical in every negotiation. A line is kept only when all
negotiations share it at the same position, so nothing specific to one negotiation can
reach the output. Personas, catalogs, prices and dialogue are written by gen_buyer.py.
"""
import gzip
import json
import sys

PUBLIC = "hw3/data/public_test_negotiation.jsonl.gz"
OUT = "data_work/processed/buyer_interface.json"
PERSONA_OPEN = "<<<PERSONA_SECTION>>>"
SITUATION = "YOUR SITUATION (private to you, fixed for this conversation):"


def main():
    systems, tools = {}, set()
    with gzip.open(PUBLIC, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            tools.add(json.dumps(row["tools"]))
            systems.setdefault(row["negotiation"], row["messages"][0]["content"].split("\n"))
    if len(tools) != 1:
        sys.exit(f"expected one tool list, found {len(tools)}")
    if len({len(lines) for lines in systems.values()}) != 1:
        sys.exit("system messages differ in line count")

    all_lines = list(systems.values())
    fixed = [len({lines[i] for lines in all_lines}) == 1 for i in range(len(all_lines[0]))]
    template = all_lines[0]
    open_at = template.index(PERSONA_OPEN)
    situation_at = template.index(SITUATION)
    tail_at = next(i for i in range(situation_at, open_at, -1) if not fixed[i - 1] and template[i - 1]) if not all(fixed[open_at:situation_at]) else open_at + 1
    if not all(fixed[:open_at + 1]) or not all(fixed[tail_at:situation_at + 1]):
        sys.exit("the kept ranges are not fixed")

    interface = {
        "system_head": "\n".join(template[:open_at + 1]),
        "system_tail": "\n".join(template[tail_at:situation_at + 1]),
        "tools": json.loads(tools.pop()),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(interface, f, ensure_ascii=False, indent=1)
    print(f"negotiations {len(systems)}, tools {len(interface['tools'])}")
    print(f"kept {open_at + 1 + situation_at + 1 - tail_at} fixed lines; left out {tail_at - open_at - 1} persona lines and the situation JSON")


if __name__ == "__main__":
    main()

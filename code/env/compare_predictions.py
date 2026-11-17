"""Compare two prediction files on the ids they share.

    python code/env/compare_predictions.py A.jsonl B.jsonl

Prints counts only, never the model text.
"""
import json
import os
import sys


def load(path):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return {row["id"]: row for row in rows}


def common_prefix(a, b):
    return len(os.path.commonprefix((a, b)))


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    shared = sorted(set(a) & set(b))
    same = [i for i in shared if a[i]["output"] == b[i]["output"]]
    different = [i for i in shared if a[i]["output"] != b[i]["output"]]
    finish = sum(a[i].get("finish_reason") != b[i].get("finish_reason") for i in shared)
    print(f"{sys.argv[1]} vs {sys.argv[2]}")
    print(f"  shared ids {len(shared)} (only in first {len(a) - len(shared)}, only in second {len(b) - len(shared)})")
    print(f"  identical output {len(same)} / {len(shared)}; finish_reason differs {finish}")
    if different:
        prefixes = sorted(common_prefix(a[i]["output"], b[i]["output"]) for i in different)
        lengths = sorted(max(len(a[i]["output"]), len(b[i]["output"])) for i in different)
        print(f"  different outputs: shared prefix chars min {prefixes[0]}, median {prefixes[len(prefixes) // 2]}, "
              f"max {prefixes[-1]}; median length {lengths[len(lengths) // 2]}")


if __name__ == "__main__":
    main()

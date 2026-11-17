"""Split a Part 1 score into the step types that carry its weight.

    python code/eval_breakdown.py --pred runs/E02/predictions/tools.jsonl --out runs/E02/metrics/tools_breakdown.json

Every verdict comes from the official scorer; this file only groups them. It prints
counts, never public content. Run it from the project root.
"""
import argparse
import collections
import json
import sys

sys.path.insert(0, "hw3")

from eval import format as F  # noqa: E402
from eval import score_tools as T  # noqa: E402


def step_type(step):
    if T.must_call(step):
        return "must_call"
    return "first_call" if step["is_call"] else "text"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=F.DATA_TOOLS)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--parser", default="lenient", choices=("lenient", "strict"))
    ap.add_argument("--only-predicted", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()

    rows = F.read_jsonl(args.data)
    preds, _ = F.load_predictions([args.pred])
    only = set(preds) if args.only_predicted else None
    result, steps = T.score_tools(rows, preds, args.parser, only)
    previous = {row["id"]: row["messages"][-1]["role"] for row in rows}

    groups = collections.defaultdict(collections.Counter)
    for step in steps:
        group = groups[f"{step_type(step)} after {previous[step['id']]}"]
        group["n"] += 1
        group["right"] += step["step"]
        group["right_tool_name"] += step["tool"]
        group["invalid"] += step["invalid"]
        group["capped"] += step["capped"]

    weight = F.settings()["must_call_weight"]
    total = earned = 0
    report = {}
    for name in sorted(groups):
        group = groups[name]
        scale = weight if name.startswith("must_call") else 1
        total += scale * group["n"]
        earned += scale * group["right"]
        report[name] = {**group, "weight": scale * group["n"], "accuracy": group["right"] / group["n"]}

    # The groups must add back up to the official number, or the grouping is wrong.
    official = result["with_first_step_rule"]["all"]["weighted_step_acc"]
    if abs(earned / total - official) > 1e-12:
        sys.exit(f"breakdown {earned / total} does not match the official score {official}")

    print(f"{args.parser} weighted step accuracy {100 * official:.2f}")
    print(f"{'step type':26s} {'n':>6s} {'weight':>7s} {'share':>6s} {'right':>6s} {'acc':>6s} {'capped':>6s}")
    for name, group in report.items():
        print(f"{name:26s} {group['n']:6d} {group['weight']:7d} {100 * group['weight'] / total:5.1f}% "
              f"{group['right']:6d} {100 * group['accuracy']:5.1f}% {group['capped']:6d}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"parser": args.parser, "weighted_step_acc": official, "groups": report}, f, indent=2)


if __name__ == "__main__":
    main()

"""Print the E22 comparison table from saved metric files, one row per evaluated adapter.

    python code/summarize_e22.py [EVAL_DIR:UPDATE ...]

Default rows: E10 and E21 baselines, then every E22 node. Each cell is exact (name and
arguments right), or for "name" columns only the tool name, as a percentage of the group's
steps. Part 2 columns are the share of steps where ordering or not matches the gold. Public
columns come from the fixed sample and the official scorer; they are reported, not used to pick.
Run from the project root.
"""
import json
import os
import sys

# The baselines skip the public sample; the same adapters' sample results from their own runs are used.
PUBLIC_FROM = {"runs/E22/baseline_E10": "runs/E10/eval", "runs/E22/baseline_E21": "runs/E21/eval",
               "runs/E23/baseline_E21": "runs/E21/eval", "runs/E23/baseline_E22": "runs/E22/eval"}  # E23: v5 dev_bfcl
DEFAULT = ["runs/E22/baseline_E10:1250", "runs/E22/baseline_E21:1375"] + [f"runs/E22/eval:{n}" for n in (148, 369, 738, 1106, 1475)]
COLUMNS = (  # header, file, group, measure
    ("old dev P1", "dev", "part1", "exact"),
    ("rules P1", "dev_new", "part1", "exact"),
    ("rules call after tool", "dev_new", "part1 policy call after tool", "exact"),
    ("rules values from system", "dev_new", "part1 policy call, values from system", "exact"),
    ("bfcl P1", "dev_bfcl", "part1", "exact"),
    ("bfcl call after tool: name", "dev_bfcl", "call after tool", "right_action"),
    ("bfcl call after tool", "dev_bfcl", "call after tool", "exact"),
    ("bfcl values from tool", "dev_bfcl", "values from tool result", "exact"),
    ("bfcl values from system", "dev_bfcl", "values from system", "exact"),
    ("bfcl text after user", "dev_bfcl", "text after user", "exact"),
    ("old dev P2", "dev", "part2", "order_decision_right"),
    ("rules-dev P2", "dev_new", "part2", "order_decision_right"),
)


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cell(totals, group, measure):
    if group in totals:
        counts = [totals[group]]
    else:  # a group name shared by both BFCL variants: add them up
        counts = [count for name, count in totals.items() if name.startswith("part1 bfcl_") and name.endswith(group)]
    steps = sum(count["steps"] for count in counts)
    return f"{100 * sum(count.get(measure, 0) for count in counts) / steps:.1f}" if steps else "-"


def main():
    rows = sys.argv[1:] or DEFAULT
    print("| adapter | " + " | ".join(header for header, *_ in COLUMNS) + " | public P1 | public must-call | order | volume | price |")
    print("|---" * (len(COLUMNS) + 6) + "|")
    for row in rows:
        folder, update = row.rsplit(":", 1)
        cells = []
        for _, name, group, measure in COLUMNS:
            metrics = load(f"{folder}/{name}_{update}.metrics.json")
            cells.append(cell(metrics["totals"], group, measure) if metrics else "未測")
        public = PUBLIC_FROM.get(folder, folder)
        tools = load(f"{public}/tools_{update}.breakdown.json")
        negotiation = load(f"{public}/negotiation_{update}.metrics.json")
        if tools:
            must = tools["groups"]["must_call after tool"]
            cells += [f"{100 * tools['weighted_step_acc']:.2f}", f"{100 * must['accuracy']:.1f}"]
        else:
            cells += ["未測", "未測"]
        if negotiation:
            m = negotiation["metrics"]
            cells += [f"{100 * m['order_accuracy']:.1f}" if isinstance(m["order_accuracy"], float) else "n/a", f"{100 * m['fill_rate']:.1f}",
                      f"{m['relative_price']:.3f}" if isinstance(m["relative_price"], float) else "n/a"]
        else:
            cells += ["未測"] * 3
        print(f"| {folder.split('/')[-1]} {update} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()

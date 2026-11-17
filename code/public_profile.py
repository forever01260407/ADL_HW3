"""Profile the public Part 1 steps by whether a long system message is present.

    python code/public_profile.py [--pred FILE]

Prints aggregate counts only, never public content. It reads no gold text into any file and
feeds nothing into training data; it only tells us which structure our data lacks. With --pred,
verdicts come from the official scorer. Run from the project root.
"""
import argparse
import collections
import hashlib
import statistics
import sys

sys.path.insert(0, "hw3")

from eval import format as F  # noqa: E402
from eval import score_tools as T  # noqa: E402

LONG_SYSTEM = 2000


def group_of(row):
    first = row["messages"][0]
    if first["role"] != "system":
        return "no system"
    return "long system" if len(first.get("content") or "") >= LONG_SYSTEM else "short system"


def step_kind(row):
    after = row["messages"][-1]["role"]
    if row["acceptable_next"]:
        return "call after user" if row.get("first_step_eligible") else f"must-call after {after}"
    return f"text after {after}"


def fingerprint(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def quantiles(values):
    ordered = sorted(values)
    return f"min {ordered[0]} median {statistics.median(ordered):.0f} max {ordered[-1]}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred")
    args = ap.parse_args()

    rows = F.read_jsonl(F.DATA_TOOLS)
    verdicts = {}
    if args.pred:
        predictions, _ = F.load_predictions([args.pred])
        _, steps = T.score_tools(rows, predictions, "lenient", set(predictions))
        verdicts = {step["id"]: step for step in steps}

    weight_of = F.settings()["must_call_weight"]
    total_weight = 0
    table = collections.defaultdict(collections.Counter)
    systems = collections.defaultdict(collections.Counter)
    tool_lists = collections.defaultdict(set)
    tool_counts = collections.defaultdict(list)
    episodes = collections.defaultdict(set)
    system_lengths = collections.defaultdict(list)
    result_lengths = collections.defaultdict(list)
    user_turns = collections.defaultdict(dict)
    for row in rows:
        group, kind = group_of(row), step_kind(row)
        must = kind.startswith("must-call")
        weight = weight_of if must else 1
        total_weight += weight
        entry = table[(group, kind)]
        entry["n"] += 1
        entry["weight"] += weight
        step = verdicts.get(row["id"])
        if step:
            entry["scored"] += 1
            entry["right"] += step["step"]
        episodes[group].add(row["episode"])
        tool_lists[group].add(fingerprint(sorted(tool["function"]["name"] if "function" in tool else tool["name"] for tool in row["tools"])))
        tool_counts[group].append(len(row["tools"]))
        if row["messages"][0]["role"] == "system":
            systems[group][fingerprint(row["messages"][0].get("content"))] += 1
            system_lengths[group].append(len(row["messages"][0].get("content") or ""))
        if row["messages"][-1]["role"] == "tool":
            result_lengths[group].append(len(str(row["messages"][-1].get("content"))))
        turns = sum(message["role"] == "user" for message in row["messages"])
        user_turns[group][row["episode"]] = max(turns, user_turns[group].get(row["episode"], 0))

    print(f"{'group':12s} {'step':22s} {'n':>6s} {'weight':>7s} {'share':>6s} {'right':>7s}")
    for (group, kind), entry in sorted(table.items()):
        rate = f"{100 * entry['right'] / entry['scored']:.1f}%" if entry["scored"] else "-"
        print(f"{group:12s} {kind:22s} {entry['n']:6d} {entry['weight']:7d} {100 * entry['weight'] / total_weight:5.1f}% {rate:>7s}")

    print()
    for group in sorted(episodes):
        counter = systems[group]
        print(f"{group}: {len(episodes[group])} conversations, {len(tool_lists[group])} distinct tool lists, tools per step {quantiles(tool_counts[group])}")
        print(f"  user turns per conversation {quantiles(list(user_turns[group].values()))}")
        if result_lengths[group]:
            print(f"  tool result length {quantiles(result_lengths[group])}")
        if counter:
            top = [count for _, count in counter.most_common(5)]
            print(f"  {len(counter)} distinct system messages, length {quantiles(system_lengths[group])}; steps under the five most common: {top}")


if __name__ == "__main__":
    main()

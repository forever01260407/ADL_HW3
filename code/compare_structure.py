"""Compare the shape of must-call steps in the public test with our training data.

    python code/compare_structure.py --pred FILE [--train data_work/frozen/data_v1/train.jsonl]

For each must-call step this asks where the next call's argument values appear, whether the next
tool repeats the previous one, and how the model did on that kind of step. It prints aggregate
counts only, never public content, and nothing it reads goes into training data. Verdicts come
from the official scorer. Run from the project root.
"""
import argparse
import collections
import json
import sys

sys.path.insert(0, "code/data")
sys.path.insert(0, "hw3")

from common import expand_steps  # noqa: E402
from eval import format as F  # noqa: E402
from eval import score_tools as T  # noqa: E402


def as_dict(arguments):
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except ValueError:
            return {}
    return arguments if isinstance(arguments, dict) else {}


def leaves(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from leaves(item)
    elif isinstance(value, list):
        for item in value:
            yield from leaves(item)
    elif value is not None and not isinstance(value, bool):
        yield str(value)


def text_of(message):
    content = message.get("content")
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)


def value_source(value, messages):
    """The nearest place a value appears word for word. Values under 3 characters match by chance."""
    if len(value) < 3:
        return "short value"
    results = [text_of(m) for m in messages if m["role"] == "tool"]
    if results and value in results[-1]:
        return "latest tool result"
    if any(value in result for result in results[:-1]):
        return "earlier tool result"
    if any(m["role"] == "user" and value in text_of(m) for m in messages):
        return "user message"
    if any(m["role"] == "system" and value in text_of(m) for m in messages):
        return "system message"
    return "nowhere word for word"


def calls_so_far(messages):
    return [call["function"] for m in messages if m["role"] == "assistant" for call in m.get("tool_calls") or []]


def results_since_user(messages):
    count = 0
    for message in reversed(messages):
        if message["role"] == "user":
            break
        count += message["role"] == "tool"
    return min(count, 4)


def system_bucket(length):
    for limit in (200, 500, 1000, 2000, 4000):
        if length < limit:
            return f"under {limit:5d}"
    return "4000 or more"


def shape_of(messages, call, n_tools):
    """Facts about one must-call step, as (name, value) pairs."""
    earlier = calls_so_far(messages)
    arguments = as_dict(call["arguments"])
    previous = earlier[-1]["name"] if earlier else None
    yield "tools offered", "up to 15" if n_tools <= 15 else "16-30" if n_tools <= 30 else "more than 30"
    has_system = messages[0]["role"] == "system"
    tools_bucket = "up to 15" if n_tools <= 15 else "more than 15"
    yield "system message", has_system
    yield "system and tools", f"{'system' if has_system else 'no system'}, {tools_bucket} tools"
    if has_system:
        yield "system length, characters", system_bucket(len(text_of(messages[0])))
    yield "results since user", results_since_user(messages)
    yield "next tool", "same as previous" if call["name"] == previous else "used earlier" if call["name"] in [c["name"] for c in earlier] else "new in this conversation"
    yield "arguments", min(len(arguments), 4)
    sources = {value_source(value, messages) for value in leaves(arguments)}
    for source in ("latest tool result", "earlier tool result", "user message", "system message", "nowhere word for word"):
        yield f"has a value from: {source}", source in sources


def public_steps(pred):
    rows = F.read_jsonl(F.DATA_TOOLS)
    predictions, _ = F.load_predictions([pred])
    _, steps = T.score_tools(rows, predictions, "lenient", set(predictions))
    by_id = {row["id"]: row for row in rows}
    for step in steps:
        if T.must_call(step):
            row = by_id[step["id"]]
            yield row["messages"], row["acceptable_next"][0], len(row["tools"]), step["step"]


def train_steps(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("task") == "part2":
                continue
            for step in expand_steps(record):
                gold = step["messages"][-1]
                before = step["messages"][:-1]
                if gold.get("tool_calls") and before[-1]["role"] == "tool":
                    yield before, gold["tool_calls"][0]["function"], len(step["tools"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--train", default="data_work/frozen/data_v1/train.jsonl")
    args = ap.parse_args()

    public = collections.Counter()
    right = collections.Counter()
    train = collections.Counter()
    n_public = n_train = 0
    for messages, call, n_tools, correct in public_steps(args.pred):
        n_public += 1
        for fact in shape_of(messages, call, n_tools):
            public[fact] += 1
            right[fact] += correct
    for messages, call, n_tools in train_steps(args.train):
        n_train += 1
        for fact in shape_of(messages, call, n_tools):
            train[fact] += 1

    print(f"must-call steps: public {n_public}, train {n_train}")
    print(f"{'fact':52s} {'public':>7s} {'train':>7s} {'model right':>12s}")
    for fact in sorted(set(public) | set(train), key=lambda item: (item[0], str(item[1]))):
        label = f"{fact[0]} = {fact[1]}"
        rate = f"{100 * right[fact] / public[fact]:.1f}%" if public[fact] else "-"
        print(f"{label:52s} {100 * public[fact] / n_public:6.1f}% {100 * train[fact] / n_train:6.1f}% {rate:>12s}")


if __name__ == "__main__":
    main()

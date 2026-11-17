"""Turn the BFCL multi-turn candidates into train and dev records for data_v4.

    python code/data/prepare_bfcl_v4.py

Reads data_work/processed/bfcl_multi.jsonl (built by build_bfcl_multiturn.py) and the BFCL
task files. Two things change:

1. Session state. BFCL loads each task's starting state into its tools, so in the original
   task the model never sees it, yet the reference calls use it (a contact's user id, a file
   name, a password). In the "state" variant the state the tools started from is written
   into a system message: the class description BFCL ships with each tool class, then the
   state. The next call can then depend on reading a long system text. The "plain" variant
   has no system message, as in BFCL.
2. Unsupported targets. A call whose argument value cannot be found in anything the model
   has seen (system, user, earlier calls and results, or the tool's own documentation) is
   not used as a target. It stays in the history, so later steps keep their context. The
   message indices go to `no_target`, which common.expand_steps skips.

Families (one BFCL task number) are split once: a quarter go to dev, the rest to train.
Train gets one variant per conversation, chosen at random; dev gets both, so dev can show
whether reading the state helps. Nothing here reads the public test.
"""
import collections
import json
import random
import re
import sys

sys.path.insert(0, "code/data")

import build_bfcl_multiturn as B  # noqa: E402
from qa_conversations import FREE_TEXT, leaves  # noqa: E402

SOURCE = "data_work/processed/bfcl_multi.jsonl"
TRAIN_OUT = "data_work/processed/bfcl_v4_train.jsonl"
DEV_OUT = "data_work/processed/bfcl_v4_dev.jsonl"
SEED = 20261108
DEV_SHARE = 0.25
STATE_SHARE = 0.5
HIDDEN = {"generated_ids", "long_context", "tweet_counter", "order_counter", "message_count", "ticket_counter", "user_count"}
HEADERS = (
    "You are an assistant working inside the user's session. The tools below act on that session. "
    "This is the state of the session when the conversation starts. Use it when it answers what you need, "
    "and use the tools to change anything or to check something the state does not show.",
    "You operate the user's apps through the tools provided. Below is everything the session held before "
    "the user's first message. Values the user refers to, such as names, ids, files or settings, can be read "
    "from here; actions still have to be carried out with the tools.",
    "Session context for this conversation. The following state was loaded before the user started. "
    "Rely on it for ids, names and settings, call the tools for every action, and do not assume a change "
    "happened unless a tool result shows it.",
)


def tree_lines(name, node, depth):
    if node.get("type") == "file":
        return [f"{'  ' * depth}{name}"]
    lines = [f"{'  ' * depth}{name}/"]
    for child, item in (node.get("contents") or {}).items():
        lines.extend(tree_lines(child, item, depth + 1))
    return lines


def file_system_state(config):
    root = config.get("root") or {}
    lines = []
    for name, node in root.items():
        lines.append(f"Current directory: {name}")
        lines.extend(tree_lines(name, node, 0))
    return "\n".join(lines)


def class_state(name, instance, config):
    if name == "GorillaFileSystem":
        return file_system_state(config)
    state = {key: value for key, value in vars(instance).items() if not key.startswith("_") and key not in HIDDEN}
    return json.dumps(state, ensure_ascii=False, indent=1, default=str)


def system_text(row, classes, rng):
    instances = B.start_instances(row, classes)
    parts = [rng.choice(HEADERS)]
    for name, instance in instances.items():
        description = (getattr(instance, "_api_description", "") or "").strip()
        state = class_state(name, instance, row["initial_config"].get(name, {}))
        parts.append(f"## {name}\n{description}\n\nState:\n{state}")
    return "\n\n".join(parts)


def norm(text):
    return re.sub(r"[\s_\-]+", " ", str(text).lower()).strip()


def supported(leaf, seen, doc):
    text = str(leaf)
    if isinstance(leaf, bool):
        return True
    if text in seen or json.dumps(leaf) in seen:
        return True
    if norm(text) and norm(text) in norm(seen):
        return True
    return not isinstance(leaf, (int, float)) and text in doc


def unsupported_targets(tools, messages):
    """Indices of call messages with an argument nothing visible supports."""
    docs = {tool["function"]["name"]: json.dumps(tool["function"], ensure_ascii=False) for tool in tools}
    seen, bad = "", []
    for index, message in enumerate(messages):
        calls = message.get("tool_calls") or []
        if any(not supported(leaf, seen, docs[call["function"]["name"]])
               for call in calls for name, value in call["function"]["arguments"].items() if name not in FREE_TEXT
               for leaf in leaves(value)):
            bad.append(index)
        seen += "\n" + str(message.get("content") or "")
        for call in calls:
            seen += "\n" + json.dumps(call["function"]["arguments"], ensure_ascii=False)
    return bad


def variant(record, row, classes, rng, kind):
    messages = [dict(message) for message in record["messages"]]
    if kind == "state":
        messages = [{"role": "system", "content": system_text(row, classes, rng)}] + messages
    out = {**record, "id": f"{record['id']}:{kind}", "source": f"bfcl_{kind}", "task": "part1", "messages": messages}
    out["no_target"] = unsupported_targets(record["tools"], messages)
    return out


def main():
    classes = B.load_classes()
    rows = {}
    for subset in B.SUBSETS:
        rows.update({row["id"]: row for row in B.read_lines(f"{B.RAW}/BFCL_v3_multi_turn_{subset}.json")})
    records = B.read_lines(SOURCE)
    rng = random.Random(SEED)
    families = sorted({record["family"] for record in records})
    rng.shuffle(families)
    dev_families = set(families[:round(DEV_SHARE * len(families))])

    tally = collections.Counter()
    with open(TRAIN_OUT, "w", encoding="utf-8") as train, open(DEV_OUT, "w", encoding="utf-8") as dev:
        for record in records:
            row = rows[record["id"]]
            side = "dev" if record["family"] in dev_families else "train"
            kinds = ("plain", "state") if side == "dev" else ("state" if rng.random() < STATE_SHARE else "plain",)
            for kind in kinds:
                out = variant(record, row, classes, rng, kind)
                (dev if side == "dev" else train).write(json.dumps(out, ensure_ascii=False) + "\n")
                calls = sum(bool(message.get("tool_calls")) for message in out["messages"])
                tally[f"{side} {kind} conversations"] += 1
                tally[f"{side} {kind} call steps"] += calls
                tally[f"{side} {kind} call steps without support (not targets)"] += len(out["no_target"])
                if kind == "state":
                    tally[f"{side} state system chars"] += len(out["messages"][0]["content"])
    tally["dev families"] = len(dev_families)
    tally["train families"] = len(families) - len(dev_families)
    for key, value in sorted(tally.items()):
        print(f"{key:60s} {value}")


if __name__ == "__main__":
    main()

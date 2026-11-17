"""Turn the BFCL multi-turn candidates into train and dev records for data_v5.

    python code/data/prepare_bfcl_v5.py

Same as prepare_bfcl_v4.py (same seed, same family split, same plain/state choice per
conversation, same system headers), with three fixes found by the E22 audit
(records/bfcl_independent_audit_20261108.md):

1. Whole families. A public collision in any variant removes the whole task family
   (bfcl_family_guard.excluded_families), not only the colliding record. The records are
   still drawn in the same order, so every other record comes out as in v4.
2. Asks the state answers. In the state variant a miss_param conversation can ask the user
   for a value the system message already shows. An ask (a text reply after a user message,
   followed by another user message) is not a target when every argument value of the next
   call that came from the follow-up message is a string the state shows as the value of a
   single field (`"access_token": "abc123xyz"`). A value that is one of several listed items
   (a directory, a stock symbol, a hashtag), a number that happens to match, or a follow-up
   that also supplies something the state lacks keeps the ask as a target. The reply stays
   in the history either way.
3. Numbers by value. A numeric argument is supported when a number with the same value
   appears in what the model has seen ("150" supports 150.0, "20,000" supports 20000; "31.0"
   no longer supports 1.0), or a number word from zero to twenty does. Other values are
   checked as in v4.

Nothing here reads the public test; the family list comes from the isolated collision report.
"""
import collections
import json
import random
import re
import sys

sys.path.insert(0, "code/data")

import build_bfcl_multiturn as B  # noqa: E402
import prepare_bfcl_v4 as V4  # noqa: E402
from bfcl_family_guard import excluded_families  # noqa: E402
from qa_conversations import FREE_TEXT, leaves  # noqa: E402

COLLISIONS = "data_work/processed/collisions_bfcl.json"
TRAIN_OUT = "data_work/processed/bfcl_v5_train.jsonl"
DEV_OUT = "data_work/processed/bfcl_v5_dev.jsonl"
WORDS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
         "sixteen seventeen eighteen nineteen twenty").split()


def numbers_in(text):
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)
    found = {float(match) for match in re.findall(r"(?<![\d.])-?\d+(?:\.\d+)?(?![\d])", text)}
    words = set(re.findall(r"[a-z]+", text.lower()))
    return found | {float(value) for value, word in enumerate(WORDS) if word in words}


def supported(leaf, seen, doc):
    if isinstance(leaf, bool):
        return True
    if isinstance(leaf, (int, float)):
        return float(leaf) in numbers_in(seen)
    return V4.supported(leaf, seen, doc)


def unsupported_targets(tools, messages):
    """Indices of call messages with an argument nothing visible supports (v4 with numbers by value)."""
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


def found_in(leaf, text):
    if isinstance(leaf, bool):
        return False
    if isinstance(leaf, (int, float)):
        return float(leaf) in numbers_in(text)
    return len(str(leaf)) > 1 and (str(leaf) in text or V4.norm(leaf) in V4.norm(text))


def asks_the_state_answers(messages):
    """Ask replies whose missing value the system message already shows."""
    if not messages or messages[0]["role"] != "system":
        return []
    system, bad = messages[0]["content"], []
    for index in range(1, len(messages) - 2):
        ask, follow = messages[index], messages[index + 1]
        if not (ask["role"] == "assistant" and not ask.get("tool_calls") and messages[index - 1]["role"] == "user"
                and follow["role"] == "user"):
            continue
        call = next((m for m in messages[index + 2:] if m["role"] == "assistant"), None)
        if not call or not call.get("tool_calls"):
            continue
        given = [leaf for item in call["tool_calls"] for leaf in leaves(item["function"]["arguments"])
                 if found_in(leaf, follow["content"] or "")]
        if given and all(isinstance(leaf, str) and f": {json.dumps(leaf, ensure_ascii=False)}" in system for leaf in given):
            bad.append(index)
    return bad


def variant(record, row, classes, rng, kind):
    messages = [dict(message) for message in record["messages"]]
    if kind == "state":
        messages = [{"role": "system", "content": V4.system_text(row, classes, rng)}] + messages
    out = {**record, "id": f"{record['id']}:{kind}", "source": f"bfcl_{kind}", "task": "part1", "messages": messages}
    asks = asks_the_state_answers(messages)
    out["no_target"] = sorted(set(unsupported_targets(record["tools"], messages)) | set(asks))
    return out, asks


def main():
    with open(COLLISIONS, encoding="utf-8") as f:
        blocked = excluded_families(json.load(f))
    classes = B.load_classes()
    rows = {}
    for subset in B.SUBSETS:
        rows.update({row["id"]: row for row in B.read_lines(f"{B.RAW}/BFCL_v3_multi_turn_{subset}.json")})
    records = B.read_lines(V4.SOURCE)
    rng = random.Random(V4.SEED)
    families = sorted({record["family"] for record in records})
    rng.shuffle(families)
    dev_families = set(families[:round(V4.DEV_SHARE * len(families))])

    tally = collections.Counter()
    with open(TRAIN_OUT, "w", encoding="utf-8") as train, open(DEV_OUT, "w", encoding="utf-8") as dev:
        for record in records:
            row = rows[record["id"]]
            side = "dev" if record["family"] in dev_families else "train"
            kinds = ("plain", "state") if side == "dev" else ("state" if rng.random() < V4.STATE_SHARE else "plain",)
            for kind in kinds:
                out, asks = variant(record, row, classes, rng, kind)  # drawn even when dropped, so the rest match v4
                if record["family"] in blocked:
                    tally[f"{side} dropped: family collides with public"] += 1
                    continue
                (dev if side == "dev" else train).write(json.dumps(out, ensure_ascii=False) + "\n")
                tally[f"{side} {kind} conversations"] += 1
                tally[f"{side} {kind} call steps"] += sum(bool(message.get("tool_calls")) for message in out["messages"])
                tally[f"{side} {kind} not targets"] += len(out["no_target"])
                tally[f"{side} {kind} asks the state answers (not targets)"] += len(asks)
    tally["dev families"] = len(dev_families - blocked)
    tally["train families"] = len(set(families) - dev_families - blocked)
    tally["blocked families"] = len(blocked)
    for key, value in sorted(tally.items()):
        print(f"{key:60s} {value}")


if __name__ == "__main__":
    main()

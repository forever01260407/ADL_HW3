"""Evaluate a model on our own data through the official generation path.

    python code/eval_own.py export --out runs/E01/own/steps.jsonl (--config configs/e01.json | --data FILE)
    python hw3/eval.py generate --data runs/E01/own/steps.jsonl --adapter CHECKPOINT --out runs/E01/own/step_200.jsonl --backend vllm
    python code/eval_own.py score --steps runs/E01/own/steps.jsonl --pred runs/E01/own/step_200.jsonl

`export` writes each step as an input the official `generate` can read, with the gold reply
beside it. `score` reads replies with the official reader and compares calls with the
official comparison. Orders go through the official order_ok_hw. Run from the project root.
"""
import argparse
import collections
import json
import sys

sys.path.insert(0, "code/data")
sys.path.insert(0, "hw3")

import check_buyer  # noqa: E402
from common import expand_steps  # noqa: E402
from eval import format as F  # noqa: E402
from eval import score_negotiation as N  # noqa: E402
from eval import score_tools as T  # noqa: E402


def records_of(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def buyer_facts(record):
    return {step["index"]: {"case": check_buyer.case_of(step), "step_meta": check_buyer.meta_of(step)} for step in check_buyer.steps(record)}


def gold_of(message):
    return {"text": (message.get("content") or "").strip(),
            "calls": [{"name": call["function"]["name"], "arguments": call["function"]["arguments"]} for call in message.get("tool_calls") or []]}


def argument_origin(messages, gold):
    """Where the first gold call's values were last visible before the step, for the breakdown.

    'tool result' if some value is found only in a tool message, else 'system' if only in the
    system message, else 'user or earlier call'. Free-text values are left out, as in
    qa_conversations.py. Steps without a call get None.
    """
    from qa_conversations import FREE_TEXT, leaves

    if not gold["calls"]:
        return None
    texts = collections.defaultdict(str)
    for message in messages:
        texts[message["role"]] += "\n" + str(message.get("content") or "")
        for call in message.get("tool_calls") or []:
            texts["assistant"] += "\n" + json.dumps(call["function"]["arguments"], ensure_ascii=False)
    arguments = gold["calls"][0]["arguments"] if isinstance(gold["calls"][0]["arguments"], dict) else {}
    found = set()
    for name, value in arguments.items():
        if name in FREE_TEXT:
            continue
        for leaf in leaves(value):
            if isinstance(leaf, bool):
                continue
            where = {role for role, text in texts.items() if str(leaf) in text or json.dumps(leaf) in text}
            if where and where <= {"tool"}:
                found.add("tool result")
            elif where and where <= {"system"}:
                found.add("system")
    return "tool result" if "tool result" in found else "system" if "system" in found else "user or earlier call"


def export(args):
    wanted = None
    if args.config:
        import train

        with open(args.config, encoding="utf-8") as f:
            config = json.load(f)
        wanted = {step["id"] for step in train.select_steps(config)}
        path = config["train_file"]
    else:
        path = args.data
    written = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for record in records_of(path):
            facts = buyer_facts(record) if record.get("task") == "part2" else {}
            for step in expand_steps(record):
                if wanted is not None and step["id"] not in wanted:
                    continue
                index = len(step["messages"]) - 1
                gold = gold_of(step["messages"][-1])
                out.write(json.dumps({
                    "id": step["id"], "task": step["task"], "source": step["source"], "after": step["messages"][-2]["role"],
                    "origin": argument_origin(step["messages"][:-1], gold), "tools": step["tools"], "messages": step["messages"][:-1],
                    "gold": gold, **facts.get(index, {}),
                }, ensure_ascii=False) + "\n")
                written += 1
    print(f"{written} steps -> {args.out}")


def same_calls(gold, predicted):
    return bool(gold) and bool(predicted) and T._same_call(T._canon(gold[0], {}), T._canon(predicted[0], {}))


def kind_of(row):
    gold = row["gold"]
    if row["task"] == "part2":
        return row["case"]
    return "call" if gold["calls"] else "text"


def score(args):
    rows = list(F.iter_jsonl(args.steps))
    predictions, _ = F.load_predictions([args.pred])
    totals = collections.defaultdict(collections.Counter)
    errors = []
    for row in rows:
        prediction = predictions.get(row["id"])
        reply = F.read_reply(prediction, "lenient")
        gold, kind = row["gold"], kind_of(row)
        names = [row["task"], f"{row['task']} {kind}"]
        if args.by_source and row["task"] == "part1" and "after" in row:
            source = f"{row['task']} {row.get('source')}"
            names += [source, f"{source} {kind} after {row['after']}"]
            if row.get("origin"):
                names.append(f"{source} call, values from {row['origin']}")
        for group in names:
            count = totals[group]
            count["steps"] += 1
            count["stopped"] += bool(prediction) and prediction.get("finish_reason") == "stop"
            usable = reply is not None and not reply["invalid"]
            count["parsed"] += usable
            calls = reply["calls"] if usable else []
            names_right = [call["name"] for call in calls[:1]] == [call["name"] for call in gold["calls"][:1]]
            exact = same_calls(gold["calls"], calls) if gold["calls"] else (usable and not calls and bool(reply["text"].strip()))
            count["right_action"] += usable and names_right
            count["exact"] += usable and exact
            if row["task"] == "part2":
                orders = calls[:1] if calls and calls[0]["name"] == "order" else []
                count["model_orders"] += len(orders)
                for call in orders:
                    arguments = call["arguments"] if isinstance(call["arguments"], dict) else {}
                    count["valid_orders"] += bool(N.order_ok_hw(row["step_meta"], arguments, require_citation=False))
                wanted_order = any(call["name"] == "order" for call in gold["calls"])
                count["order_decision_right"] += usable and bool(orders) == wanted_order
        if not (reply is not None and not reply["invalid"] and exact):
            errors.append({"id": row["id"], "kind": kind, "gold": gold, "finish": prediction and prediction.get("finish_reason"), "output": (prediction or {}).get("output", "")[:300]})
    print(f"{len(rows)} steps, {len(predictions)} predictions")
    for group in sorted(totals):
        count = totals[group]
        n = count["steps"]
        line = f"{group:32s} n={n:4d} stop {count['stopped'] / n:6.1%} parsed {count['parsed'] / n:6.1%} action {count['right_action'] / n:6.1%} exact {count['exact'] / n:6.1%}"
        if "order_decision_right" in count or group.startswith("part2"):
            line += f" order-or-not {count['order_decision_right'] / n:6.1%} orders {count['valid_orders']}/{count['model_orders']} valid"
        print(line)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"totals": {group: dict(count) for group, count in totals.items()}, "errors": errors}, f, indent=1, ensure_ascii=False)
    for error in errors[:args.show]:
        print(f"\n--- {error['id']} [{error['kind']}] finish={error['finish']}")
        print(f"  gold : {json.dumps(error['gold'], ensure_ascii=False)[:300]}")
        print(f"  model: {error['output']!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = ap.add_subparsers(dest="command", required=True)
    one = commands.add_parser("export")
    one.add_argument("--config")
    one.add_argument("--data")
    one.add_argument("--out", required=True)
    two = commands.add_parser("score")
    two.add_argument("--steps", required=True)
    two.add_argument("--pred", required=True)
    two.add_argument("--out")
    two.add_argument("--show", type=int, default=12)
    two.add_argument("--by-source", action="store_true", help="also group Part 1 by source, step kind and where the call's values come from")
    args = ap.parse_args()
    (export if args.command == "export" else score)(args)


if __name__ == "__main__":
    main()

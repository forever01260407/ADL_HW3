"""Run BFCL's multi-turn tasks through its own tools to get real tool results.

    python code/data/build_bfcl_multiturn.py

Writes data_work/processed/bfcl_multi.jsonl. Every call in a task's ground truth is
executed in order against the task's initial state, so each tool message is what the
tool really returned. Calls are parsed with `ast`, never evaluated. BFCL has no
assistant text, so the reply that ends a turn is a short template built only from the
last tool result. The tool code comes from code/data/fetch_bfcl_tools.py.
"""
import ast
import collections
import copy
import importlib.util
import inspect
import json
import random
import sys
import types

from common import call_message, make_tool, step_kinds, tool_message

RAW = "data_work/raw/bfcl"
TOOL_CODE = "data_work/raw/bfcl_tools"
OUT = "data_work/processed/bfcl_multi.jsonl"
COLLISIONS = "data_work/processed/collisions_bfcl.json"
SEED = 20261107

CLASSES = {
    "GorillaFileSystem": "gorilla_file_system", "MathAPI": "math_api", "MessageAPI": "message_api", "TwitterAPI": "posting_api",
    "TicketAPI": "ticket_api", "TradingBot": "trading_bot", "TravelAPI": "travel_booking", "VehicleControlAPI": "vehicle_control",
}
# Subsets whose tool list stays fixed through the conversation. composite and miss_func
# add a tool in the middle, which never happens at a graded step.
SUBSETS = ("base", "miss_param")
LONG_MODULE = "bfcl_eval.eval_checker.multi_turn_eval.func_source_code.long_context"
ASK = (
    "I need a bit more detail before I can do that. Could you tell me exactly which one you mean?",
    "Could you give me the missing detail first? I can't complete this without it.",
    "Which one do you mean? Please share the specifics so I can continue.",
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_classes():
    # The tool files import long_context through their package path.
    parts = LONG_MODULE.split(".")
    for depth in range(1, len(parts)):
        sys.modules.setdefault(".".join(parts[:depth]), types.ModuleType(parts[depth - 1]))
    load_module(LONG_MODULE, f"{TOOL_CODE}/long_context.py")
    return {name: getattr(load_module(f"bfcl_{stem}", f"{TOOL_CODE}/{stem}.py"), name) for name, stem in CLASSES.items()}


def load_docs():
    docs = {}
    for name, stem in CLASSES.items():
        with open(f"{RAW}/multi_turn_func_doc/{stem}.json", encoding="utf-8") as f:
            docs[name] = [make_tool(json.loads(line)) for line in f]
    return docs


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def parse_call(text):
    """'cd(folder="a")' -> (name, positional values, keyword values)."""
    node = ast.parse(text, mode="eval").body
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ValueError("not a plain call")
    return node.func.id, [ast.literal_eval(a) for a in node.args], {k.arg: ast.literal_eval(k.value) for k in node.keywords}


def start_instances(row, classes):
    instances = {}
    for name in row["involved_classes"]:
        instance = classes[name]()
        if hasattr(instance, "_load_scenario"):
            instance._load_scenario(copy.deepcopy(row["initial_config"].get(name, {})))
        instances[name] = instance
    return instances


def find_method(instances, name):
    for instance in instances.values():
        method = getattr(instance, name, None)
        if method is not None and not name.startswith("_"):
            return method
    raise ValueError(f"no tool {name}")


def run_call(instances, text):
    name, positional, keywords = parse_call(text)
    method = find_method(instances, name)
    arguments = dict(inspect.signature(method).bind(*positional, **keywords).arguments)
    return name, arguments, method(**arguments)


def closing_text(rng, name, result):
    if isinstance(result, dict):
        if "error" in result:
            return f"That did not work: {str(result['error'])[:160]}"
        facts = [f"{key}: {str(value)[:80]}" for key, value in list(result.items())[:2]]
        if facts:
            return rng.choice(("Done. ", "All set. ", "Finished. ")) + "; ".join(facts) + "."
    return rng.choice(("Done.", "All set.", "That's finished."))


def build_messages(row, truth, classes, rng):
    instances = start_instances(row, classes)
    messages = []
    failed_calls = 0
    for turn, calls in zip(row["question"], truth):
        messages.extend({"role": m["role"], "content": m["content"]} for m in turn if m["role"] == "user")
        if not calls:
            messages.append({"role": "assistant", "content": rng.choice(ASK)})
            continue
        name, result = None, None
        for text in calls:
            name, arguments, result = run_call(instances, text)
            failed_calls += isinstance(result, dict) and "error" in result
            messages.append(call_message([(name, arguments)]))
            messages.append(tool_message(name, result))
        messages.append({"role": "assistant", "content": closing_text(rng, name, result)})
    return messages, failed_calls


def main():
    classes = load_classes()
    docs = load_docs()
    with open(COLLISIONS, encoding="utf-8") as f:
        drop = set(json.load(f)["drop_ids"])
    rng = random.Random(SEED)
    outcomes, kinds = collections.Counter(), collections.Counter()
    errors = collections.Counter()
    with open(OUT, "w", encoding="utf-8") as f:
        for subset in SUBSETS:
            rows = read_lines(f"{RAW}/BFCL_v3_multi_turn_{subset}.json")
            answers = {a["id"]: a["ground_truth"] for a in read_lines(f"{RAW}/possible_answer/BFCL_v3_multi_turn_{subset}.json")}
            for row in rows:
                if row["id"] in drop:
                    outcomes[f"{subset}: dropped, overlaps public"] += 1
                    continue
                try:
                    messages, failed = build_messages(row, answers[row["id"]], classes, rng)
                except Exception as error:  # noqa: BLE001 - any failure means this task's results are unusable
                    outcomes[f"{subset}: dropped, could not execute"] += 1
                    errors[f"{type(error).__name__}: {str(error)[:70]}"] += 1
                    continue
                if failed:
                    outcomes[f"{subset}: dropped, a ground-truth call returned an error"] += 1
                    continue
                tools = [tool for name in row["involved_classes"] for tool in docs[name]]
                rng.shuffle(tools)
                number = row["id"].rsplit("_", 1)[-1]
                f.write(json.dumps({"id": row["id"], "source": "bfcl_multi", "family": f"bfcl_task_{number}", "tools": tools, "messages": messages}, ensure_ascii=False) + "\n")
                outcomes[f"{subset}: kept"] += 1
                kinds.update(step_kinds(messages))
    for key, value in sorted(outcomes.items()):
        print(f"{key:58s} {value:5d}")
    for key, value in sorted(kinds.items()):
        print(f"{key:22s} {value:6d}")
    for key, value in errors.most_common(10):
        print(f"  {value:4d}  {key}")


if __name__ == "__main__":
    main()

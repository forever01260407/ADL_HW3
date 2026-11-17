"""Shared helpers for conversations in the Qwen3 messages + tools format.

One conversation is one JSON line: {"id", "source", "family", "tools", "messages"}.
`family` groups records that must stay on the same side of a train/dev split.
"""
import hashlib
import json

TYPE_ALIASES = {
    "dict": "object", "float": "number", "double": "number", "int": "integer", "bool": "boolean", "str": "string",
    "list": "array", "tuple": "array", "List": "array", "Dict": "object", "String": "string", "Boolean": "boolean",
    "Integer": "integer", "Number": "number", "Array": "array", "Object": "object",
}
JSON_TYPES = {"string", "number", "integer", "boolean", "array", "object", "null"}


def clean_schema(node):
    """Turn a loose parameter schema into plain JSON schema."""
    if not isinstance(node, dict):
        return node
    out = {}
    for key, value in node.items():
        if key == "type" and isinstance(value, str):
            value = TYPE_ALIASES.get(value, value)
            if value not in JSON_TYPES:
                continue
        elif key == "properties" and isinstance(value, dict):
            value = {name: clean_schema(item) for name, item in value.items()}
        elif key == "items":
            value = clean_schema(value)
        elif key == "required" and not isinstance(value, list):
            continue
        out[key] = value
    return out


def make_tool(doc):
    function = doc.get("function", doc)
    parameters = clean_schema(function.get("parameters") or {})
    parameters.setdefault("type", "object")
    parameters.setdefault("properties", {})
    parameters.setdefault("required", [])
    return {"type": "function", "function": {"name": function["name"], "description": function.get("description") or "", "parameters": parameters}}


def call_message(calls):
    """calls: [(name, arguments dict)]."""
    return {"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": name, "arguments": arguments}} for name, arguments in calls]}


def tool_message(name, content):
    return {"role": "tool", "name": name, "content": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)}


def call_problem(function, arguments):
    """Why a call does not fit its tool, or None."""
    if not isinstance(arguments, dict):
        return "arguments_not_object"
    schema = function["parameters"]
    properties = schema["properties"]
    for name in schema["required"]:
        if name not in arguments:
            return "missing_required"
    if properties and not schema.get("additionalProperties"):
        if any(name not in properties for name in arguments):
            return "unknown_argument"
    return None


def wrong_type(function, arguments):
    """Name of the first argument whose value does not fit its declared JSON type, or None."""
    checks = {
        "string": lambda v: isinstance(v, str), "boolean": lambda v: isinstance(v, bool), "array": lambda v: isinstance(v, list),
        "object": lambda v: isinstance(v, dict), "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    }
    properties = function["parameters"]["properties"]
    for name, value in arguments.items():
        declared = (properties.get(name) or {}).get("type")
        allowed = [checks[kind] for kind in (declared if isinstance(declared, list) else [declared]) if kind in checks]
        if allowed and not any(check(value) for check in allowed):
            return name
    return None


def first_problem(tools, messages):
    """Index of the first assistant step that cannot be trained on, with the reason."""
    functions = {tool["function"]["name"]: tool["function"] for tool in tools}
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        if index and messages[index - 1]["role"] == "assistant":
            return index, "assistant_after_assistant"
        calls = message.get("tool_calls") or []
        if not calls and not (message.get("content") or "").strip():
            return index, "empty_text"
        for call in calls:
            function = functions.get(call["function"]["name"])
            if function is None:
                return index, "unknown_tool"
            problem = call_problem(function, call["function"]["arguments"])
            if problem:
                return index, problem
        results = 0
        for later in messages[index + 1:]:
            if later["role"] != "tool":
                break
            results += 1
        if calls and index + 1 < len(messages) and results != len(calls):
            return index, "result_count_mismatch"
    return None


def keep_valid_prefix(tools, messages):
    """Cut before the first bad step. Returns (messages, reason or None)."""
    found = first_problem(tools, messages)
    kept, reason = messages, None
    if found:
        kept, reason = messages[:found[0]], found[1]
    while kept and kept[-1]["role"] != "assistant":
        kept = kept[:-1]
    # A trailing call is a usable target even though its result is unknown.
    return kept, reason


def step_kinds(messages):
    """One label per assistant step, e.g. call_after_tool."""
    kinds = []
    for index, message in enumerate(messages):
        if message["role"] == "assistant" and index:
            action = "call" if message.get("tool_calls") else "text"
            kinds.append(f"{action}_after_{messages[index - 1]['role']}")
    return kinds


def expand_steps(record):
    """One training example per assistant message, keeping the ids needed to split by group.

    Indices listed in the record's optional `no_target` stay in the history of later steps
    but are not examples themselves (code/data/prepare_bfcl_v4.py). Older records have none.
    """
    skip = set(record.get("no_target") or ())
    for index, message in enumerate(record["messages"]):
        if message["role"] == "assistant" and index and index not in skip:
            yield {
                "id": f"{record['id']}#{index}", "conversation": record["id"], "source": record["source"], "family": record["family"],
                "task": record.get("task"), "tools": record["tools"], "messages": record["messages"][:index + 1],
            }


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]

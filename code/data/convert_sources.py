"""Convert the downloaded tool-calling datasets to the common conversation format.

    python code/data/convert_sources.py

Writes data_work/processed/<source>.jsonl and prints what was kept and why the rest
was not. A conversation is cut before its first unusable step, so the earlier steps
stay available. Records are not mixed or sampled here.
"""
import ast
import collections
import json
import keyword
import os
import re

from common import call_message, keep_valid_prefix, make_tool, stable_hash, step_kinds, tool_message

RAW = "data_work/raw"
OUT = "data_work/processed"


class Skip(Exception):
    """This record cannot be read; the message says why."""


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---- Glaive: plain text with USER / ASSISTANT / FUNCTION RESPONSE markers
GLAIVE_SPLIT = re.compile(r"(?:^|\n\n\n)(USER|ASSISTANT|FUNCTION RESPONSE): ")
GLAIVE_CALL = re.compile(r"""^<functioncall>\s*\{"name":\s*"(?P<name>[^"]+)",\s*"arguments":\s*(?P<quote>'?)(?P<args>.*)(?P=quote)\s*\}$""", re.S)


def glaive_tools(system):
    decoder = json.JSONDecoder()
    text = system.split("Use them if required -", 1)
    if len(text) < 2:
        return []
    text, tools, position = text[1], [], 0
    while True:
        position = text.find("{", position)
        if position < 0:
            return tools
        try:
            doc, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError as error:
            raise Skip("bad_tool_json") from error
        tools.append(make_tool(doc))


def glaive_conversation(row):
    tools = glaive_tools(row["system"])
    if not tools:
        raise Skip("no_tools")
    parts = GLAIVE_SPLIT.split(row["chat"])
    messages, pending = [], None
    for role, text in zip(parts[1::2], parts[2::2]):
        text = text.replace("<|endoftext|>", "").strip()
        if role == "USER":
            messages.append({"role": "user", "content": text})
        elif role == "FUNCTION RESPONSE":
            if pending is None:
                raise Skip("result_without_call")
            messages.append(tool_message(pending, text))
            pending = None
        elif text.startswith("<functioncall>"):
            match = GLAIVE_CALL.match(text)
            if not match:
                raise Skip("bad_call_text")
            try:
                arguments = json.loads(match["args"])
            except json.JSONDecodeError as error:
                raise Skip("bad_arguments_json") from error
            call = call_message([(match["name"], arguments)])
            if messages and messages[-1]["role"] == "assistant" and not messages[-1].get("tool_calls"):
                call["content"] = messages.pop()["content"]  # text and call split over two segments
            messages.append(call)
            pending = match["name"]
        else:
            messages.append({"role": "assistant", "content": text})
    return tools, messages


# ---- Hermes: ShareGPT turns with <tool_call> and <tool_response> blocks
HERMES_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
HERMES_RESULT = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.S)


def loose_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError) as error:
            raise Skip("bad_block") from error


def hermes_conversation(row):
    declared = row["tools"]
    tools = [make_tool(doc) for doc in (json.loads(declared) if isinstance(declared, str) else declared)]
    messages, names = [], []
    for turn in row["conversations"]:
        role, text = turn["from"], turn["value"]
        if role == "human":
            messages.append({"role": "user", "content": text.strip()})
        elif role == "gpt":
            blocks = HERMES_CALL.findall(text)
            if blocks:
                calls = [loose_json(block) for block in blocks]
                names = [call["name"] for call in calls]
                messages.append(call_message([(call["name"], call.get("arguments", {})) for call in calls]))
            else:
                messages.append({"role": "assistant", "content": text.strip()})
        elif role == "tool":
            results = HERMES_RESULT.findall(text)
            if len(results) != len(names):
                raise Skip("result_count_mismatch")
            for name, block in zip(names, results):
                result = loose_json(block)
                messages.append(tool_message(name, result.get("content", result) if isinstance(result, dict) else result))
    return tools, messages


# ---- ToolACE: calls written as [name(a=1, b="x"), other()]
def split_calls(text):
    """Split the inside of [...] on top-level commas."""
    pieces, depth, quote, start = [], 0, None, 0
    for index, char in enumerate(text):
        if quote:
            if char == quote and text[index - 1] != "\\":
                quote = None
        elif char in "\"'":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            pieces.append(text[start:index])
            start = index + 1
    pieces.append(text[start:])
    return [piece.strip() for piece in pieces if piece.strip()]


def toolace_call(piece, names):
    """names: tool names, longest first. They may hold spaces and brackets, so a regex cannot find them."""
    name = next((n for n in names if piece.startswith(n) and piece[len(n):].lstrip().startswith("(")), None)
    if name is None:
        raise Skip("unknown_tool")
    inner = piece[len(name):].strip()[1:-1]
    # Parameter names such as `from` are Python keywords.
    inner = re.sub(r"(?<![\w\"'])([a-z]+)\s*=(?!=)", lambda m: f"{m[1]}_=" if keyword.iskeyword(m[1]) else m[0], inner)
    try:
        call = ast.parse(f"f({inner})", mode="eval").body
        if not isinstance(call, ast.Call) or call.args or any(item.arg is None for item in call.keywords):
            raise Skip("positional_arguments")
        return name, {item.arg.removesuffix("_") if keyword.iskeyword(item.arg.removesuffix("_")) else item.arg: ast.literal_eval(item.value)
                      for item in call.keywords}
    except (ValueError, SyntaxError, TypeError) as error:
        raise Skip("bad_arguments") from error


def toolace_tools(system):
    start = system.find("[{")
    if start < 0:
        raise Skip("no_tools")
    try:
        docs, _ = json.JSONDecoder().raw_decode(system, start)
    except json.JSONDecodeError as error:
        raise Skip("bad_tool_json") from error
    return [make_tool(doc) for doc in docs]


def toolace_conversation(row):
    tools = toolace_tools(row["system"])
    known = sorted((tool["function"]["name"] for tool in tools), key=len, reverse=True)
    messages, names = [], []
    turns = row["conversations"]
    for index, turn in enumerate(turns):
        role, text = turn["from"], turn["value"].strip()
        if role == "user":
            messages.append({"role": "user", "content": text})
        elif role == "assistant":
            if text.startswith("[") and text.endswith("]") and "(" in text:
                calls = [toolace_call(piece, known) for piece in split_calls(text[1:-1])]
                names = [name for name, _ in calls]
                messages.append(call_message(calls))
                if index + 1 >= len(turns) or turns[index + 1]["from"] != "tool":
                    break  # no result was recorded, so nothing after this call can be trusted
            else:
                messages.append({"role": "assistant", "content": text})
        elif role == "tool":
            try:
                results = json.loads(text)
            except json.JSONDecodeError as error:
                raise Skip("bad_result_json") from error
            if not isinstance(results, list) or len(results) != len(names):
                raise Skip("result_count_mismatch")
            for name, result in zip(names, results):
                messages.append(tool_message(name, result.get("results", result) if isinstance(result, dict) else result))
    return tools, messages


def read_all(rows, read_one):
    """Yield (tools, messages) per row, or a Skip for a row that cannot be read."""
    for row in rows:
        try:
            yield read_one(row)
        except Skip as skip:
            yield skip
        except (KeyError, TypeError, AttributeError, ValueError) as error:
            yield Skip(type(error).__name__)


def convert(name, stream):
    reasons, kinds = collections.Counter(), collections.Counter()
    seen = set()
    kept = 0
    with open(f"{OUT}/{name}.jsonl", "w", encoding="utf-8") as f:
        for index, item in enumerate(stream):
            if isinstance(item, Skip):
                reasons[f"dropped: {item}"] += 1
                continue
            tools, messages = item
            messages, reason = keep_valid_prefix(tools, messages)
            if reason:
                reasons[f"cut: {reason}"] += 1
            if not messages:
                reasons["dropped: nothing usable"] += 1
                continue
            digest = stable_hash([tools, messages])
            if digest in seen:
                reasons["dropped: exact duplicate"] += 1
                continue
            seen.add(digest)
            kept += 1
            kinds.update(step_kinds(messages))
            family = stable_hash(sorted(tool["function"]["name"] for tool in tools))
            record = {"id": f"{name}_{index}", "source": name, "family": f"{name}_{family}", "tools": tools, "messages": messages}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\n{name}: kept {kept} conversations")
    for key, value in sorted(kinds.items()):
        print(f"  {key:22s} {value:7d}")
    for key, value in reasons.most_common():
        print(f"  {key:34s} {value:6d}")


def main():
    os.makedirs(OUT, exist_ok=True)
    convert("glaive", read_all(read_json(f"{RAW}/glaive/glaive-function-calling-v2.json"), glaive_conversation))
    convert("hermes", read_all(read_json(f"{RAW}/hermes/func-calling.json"), hermes_conversation))
    convert("toolace", read_all(read_json(f"{RAW}/toolace/data.json"), toolace_conversation))


if __name__ == "__main__":
    main()

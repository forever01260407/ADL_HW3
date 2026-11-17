"""Independent check of gen_focus.py output: re-solve every target step from the rendered messages.

    python code/data/check_focus.py FILE [FILE ...] --out REPORT.json

Shares no code with the generator. Tools are recognised by their descriptions (names change per
conversation). For each assistant step the checker works out, from what is visible before it,
what the next action must be (which tool with which arguments, or a text reply) and compares it
with the target. It also checks the schema (required, unknown, types, enums), that every
non-free-text argument value was visible before the step, and that the facts in closing
replies (ids, dates, dollar amounts) appear in earlier tool results. Exit 1 on any failure.
"""
import argparse
import collections
import json
import re
import sys

EMAIL = re.compile(r"[\w.]+@[\w.]+\.test")
ROLE_BY_DESCRIPTION = (("find", r"^Find a "), ("whoami", r"signed in to this session"), ("list", r"^List every "), ("get", r"^Full details of one"),
                       ("approve", r"^Ask a supervisor"), ("notify", r"^Send a message to"), ("commit", r"^Write all pending changes"),
                       ("act", r"^\w+ one \w+\. Only for amounts"))


def roles(tools):
    out = {}
    for tool in tools:
        f = tool["function"]
        for role, pattern in ROLE_BY_DESCRIPTION:
            if re.search(pattern, f["description"]):
                out[role] = f
    return out


def schema_problem(function, args):
    props, required = function["parameters"]["properties"], function["parameters"]["required"]
    if any(k not in args for k in required):
        return "missing_required"
    if any(k not in props for k in args):
        return "unknown_argument"
    for k, v in args.items():
        t = props[k].get("type")
        if t == "string" and not isinstance(v, str) or t == "number" and (isinstance(v, bool) or not isinstance(v, (int, float))):
            return f"wrong_type:{k}"
        if "enum" in props[k] and v not in props[k]["enum"]:
            return f"not_in_enum:{k}"
    return None


def visible(value, seen):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return any(abs(float(x) - float(value)) < 1e-9 for x in re.findall(r"-?\d+(?:\.\d+)?", seen))
    return str(value) in seen


def param(function, kind):
    """Name of the parameter of a tool that holds an id, an email, an amount."""
    for name, spec in function["parameters"]["properties"].items():
        d = spec.get("description", "")
        if kind == "email" and "email" in d:
            return name
        if kind == "owner" and d.startswith("Id of the") and name not in ("subject_ref",) and "item" not in name and not d.endswith(("tray.", "vehicle.", "session.", "sample.", "shipment.", "seat_hold.", "berth_booking.", "desk_booking.")):
            return name
        if kind == "item" and d.startswith("Id of the") and d.endswith(("tray.", "vehicle.", "session.", "sample.", "shipment.", "seat_hold.", "berth_booking.", "desk_booking.")):
            return name
    return None


def expected(messages, tools):
    """('call', name, args) or ('text', why) for the step after `messages`."""
    r = roles(tools)
    users = [m["content"] for m in messages if m["role"] == "user"]
    goal = users[0]
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    results = [(m["name"], json.loads(m["content"])) for m in messages if m["role"] == "tool"]
    by_role = {}
    names = {f["name"]: role for role, f in r.items()}
    for name, content in results:
        by_role.setdefault(names.get(name, name), content)
    if "find" not in by_role and "whoami" not in by_role:
        emails = EMAIL.findall(" ".join(users))
        if emails:
            last_user_with_email = [u for u in users if EMAIL.search(u)][-1]
            f = r["find"]
            return "call", f["name"], {param(f, "email"): EMAIL.findall(last_user_with_email)[-1]}
        if "whoami" in r:
            return "call", r["whoami"]["name"], {}
        return "text", "ask: no email and no session user"
    ident = by_role.get("find") or by_role.get("whoami")
    owner_id = next(v for k, v in ident.items() if k != "name")
    if re.search(r"What is my [\w ]+ id\?", goal):
        return "text", "finish: identity answers"
    f = r["list"]
    if "list" not in by_role:
        return "call", f["name"], {param(f, "owner"): owner_id}
    items = by_role["list"]["items"]
    if not items:
        return "text", "ask: nothing listed"
    m = re.search(r"with the highest (.+?)(?: last updated|[;.?])", goal)
    s = re.search(r" that is (.+?)(?: last updated|[;.?])", goal)
    if m:
        key = m[1].replace(" ", "_")
        target = max(items, key=lambda x: x[key])
    elif s:
        hits = [x for x in items if x["status"].replace("_", " ") == s[1]]
        if len(hits) != 1:
            return ("text", "ask: nothing meets the condition") if not hits else ("unresolved", "several meet the condition", None)
        target = hits[0]
    else:
        hits = [x for x in items if any(isinstance(v, str) and k not in ("id", "status") and (f"my {v} " in goal or f" {v} " in goal or goal.find(f"{k} {v}") >= 0) for k, v in x.items())]
        if len(hits) != 1:
            return "unresolved", f"{len(hits)} items match the description", None
        target = hits[0]
    g = r["get"]
    if "get" not in by_role:
        return "call", g["name"], {param(g, "item"): target["id"]}
    detail = by_role["get"]
    amount = next(v for k, v in detail.items() if isinstance(v, (int, float)) and not isinstance(v, bool))
    wants = []
    a_word = r["act"]["description"].split()[0].lower()
    if re.search(rf"[Pp]lease {a_word} ", goal):
        wants.append("policy")
    if "send me a note" in goal or "message me about it" in goal:
        wants.append("notify")
    if "save the changes" in goal:
        wants.append("commit")
    done = set(by_role)
    for want in wants:
        if want == "policy" and not ({"act", "approve"} & done):
            limit = float(re.search(r"is above \$(\d+(?:\.\d+)?)", system)[1])
            if amount > limit:
                return "call", r["approve"]["name"], {param(r["approve"], "item"): target["id"], "amount": amount}
            return "call", r["act"]["name"], {param(r["act"], "item"): target["id"]}
        if want == "notify" and "notify" not in done:
            n = r["notify"]
            enum = n["parameters"]["properties"]["channel"]["enum"]
            asked = re.search(r"by (\w+), not the usual channel", goal)
            channel = asked[1] if asked else re.search(r"go out by (\w+) unless", system)[1]
            if channel not in enum:
                return "unresolved", "channel not in enum", None
            note = re.search(r'saying "([^"]+)"', goal)[1]
            return "call", n["name"], {param(n, "owner"): owner_id, "channel": channel, "subject_ref": target["id"], "message": note}
        if want == "commit" and "commit" not in done:
            return "call", r["commit"]["name"], {}
    return "text", "finish: request complete"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    report = {"files": {}, "failures": []}
    for path in a.files:
        tally = collections.Counter()
        for line in open(path, encoding="utf-8"):
            rec = json.loads(line)
            msgs, tools = rec["messages"], rec["tools"]
            fn = {t["function"]["name"]: t["function"] for t in tools}
            for i, msg in enumerate(msgs):
                if msg["role"] != "assistant" or i == 0:
                    continue
                tally["steps"] += 1
                seen = "\n".join(str(m.get("content") or "") for m in msgs[:i])
                exp = expected(msgs[:i], tools)
                calls = msg.get("tool_calls") or []
                fail = None
                if exp[0] == "unresolved":
                    fail = f"unresolved: {exp[1]}"
                elif calls:
                    c = calls[0]["function"]
                    tally["call_steps"] += 1
                    tally[f"call_after_{msgs[i - 1]['role']}"] += 1
                    problem = schema_problem(fn[c["name"]], c["arguments"]) if c["name"] in fn else "unknown_tool"
                    hidden = [k for k, v in c["arguments"].items() if k != "message" and not visible(v, seen)]
                    if problem:
                        fail = problem
                    elif hidden:
                        fail = f"value not visible before the step: {hidden}"
                    elif exp[0] != "call" or exp[1] != c["name"] or exp[2] != c["arguments"]:
                        fail = f"expected {exp}, target {c['name']} {c['arguments']}"
                else:
                    tally["text_steps"] += 1
                    tally[f"text_after_{msgs[i - 1]['role']}"] += 1
                    text = msg["content"]
                    facts = re.findall(r"\b[A-Z]{2}-[A-Z0-9]{6}\b|\bU-[A-Z0-9]{6}\b|\d{4}-\d{2}-\d{2}|\$\d+\.\d{2}", text)
                    results = "\n".join(m["content"] for m in msgs[:i] if m["role"] in ("tool", "user"))
                    missing = [x for x in facts if x.lstrip("$") not in results and x not in results and not visible(float(x.lstrip("$")), results) if x.startswith("$")]
                    missing += [x for x in facts if not x.startswith("$") and x not in results]
                    if exp[0] != "text":
                        fail = f"expected {exp}, target is text"
                    elif not text.strip():
                        fail = "empty text"
                    elif missing:
                        fail = f"reply states facts not in results: {missing}"
                    elif exp[1].startswith("ask") != text.rstrip().endswith("?"):
                        fail = f"expected {exp[1]}, text {'asks' if text.rstrip().endswith('?') else 'does not ask'}"
                if fail:
                    tally["failures"] += 1
                    report["failures"].append({"id": f"{rec['id']}#{i}", "family": rec["family"], "why": fail})
            tally["conversations"] += 1
            tally[f"family:{rec['family']}"] += 1
        report["files"][path] = dict(tally)
    report["passed"] = not report["failures"]
    with open(a.out, "x", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    for path, t in report["files"].items():
        fams = sum(1 for k in t if k.startswith("family:"))
        print(path, {k: v for k, v in t.items() if not k.startswith("family:")}, "families", fams)
    for x in report["failures"][:15]:
        print("FAIL", x)
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()

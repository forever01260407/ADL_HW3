"""Part 1 focus conversations for E24 (runbook STEP 04): workflow families built from small state machines.

    python code/data/gen_focus.py --split train --seed 24101 --per-family 16 --out FILE --sidecar FILE

A workflow family is a decision structure: how the conversation is entered (value given, zero-
argument lookup, value missing so the assistant asks, value corrected by the user), what the
tool results force next (follow an id, pick the one item a stated condition selects, assemble a
call from several sources, let a written policy choose the tool, take a value the user gave
earlier although the latest result shows a similar one, call a no-argument tool), and how it ends
(answer from the result, stop because the first result already answers, ask because nothing was
found). FAMILIES fixes which compositions belong to train, dev_select and dev_confirm; no
composition is in two splits. Domains, names and values are drawn per conversation and are not
what makes a family.

The worlds are computed: every tool result comes from the conversation's own state, every
argument comes from a message the model can see before the step, and every closing reply states
facts from the results. The sidecar (never shown to the model) records per target step the
expected action, the family, and where each argument comes from; code/data/check_focus.py checks
the conversations from the rendered messages alone and shares no code with this file.
Domains differ from code/data/gen_chains.py and nothing here reads the public test.
"""
import argparse
import json
import random

FAMILIES = {  # (entry, middle actions, end). `follow_id` means list -> get only, which every non-ending family does.
    "train": [
        ("direct", ["follow_id"], "answer"), ("zero_arg", ["pick_by_condition"], "answer"), ("ask_first", ["follow_id"], "answer"),
        ("direct", ["many_sources"], "answer"), ("direct", ["policy_choice"], "answer"), ("corrected", ["follow_id"], "answer"),
        ("direct", ["no_arg_tool"], "answer"), ("zero_arg", ["earlier_value"], "answer"), ("direct", [], "already_answered"),
        ("direct", ["follow_id"], "nothing_found"), ("ask_first", ["pick_by_condition", "many_sources"], "answer"), ("corrected", ["policy_choice"], "answer"),
    ],
    "select": [
        ("zero_arg", ["follow_id"], "answer"), ("direct", ["pick_by_condition"], "answer"), ("ask_first", ["many_sources"], "answer"),
        ("corrected", ["pick_by_condition"], "answer"), ("zero_arg", ["policy_choice"], "answer"), ("direct", ["earlier_value"], "answer"),
        ("ask_first", ["no_arg_tool"], "answer"), ("zero_arg", [], "already_answered"), ("direct", ["pick_by_condition"], "nothing_found"),
        ("direct", ["many_sources", "no_arg_tool"], "answer"), ("corrected", ["earlier_value"], "answer"), ("zero_arg", ["policy_choice", "many_sources"], "answer"),
    ],
    "confirm": [
        ("ask_first", ["policy_choice"], "answer"), ("corrected", ["many_sources"], "answer"), ("ask_first", ["policy_choice", "no_arg_tool"], "answer"),
        ("zero_arg", ["pick_by_condition", "policy_choice"], "answer"), ("ask_first", ["earlier_value"], "answer"), ("corrected", [], "already_answered"),
    ],
}

DOMAINS = [
    {"key": "greenhouse", "topic": "a greenhouse operations system", "owner": "grower", "item": "tray", "action": "transplant", "amount": "cost",
     "statuses": ("seeded", "rooted", "ready", "shipped"), "channel": ("sms", "email"), "detail": "variety", "values": ("basil", "tomato", "kale", "pepper", "mint")},
    {"key": "fleet", "topic": "a delivery fleet manager", "owner": "driver", "item": "vehicle", "action": "service", "amount": "repair_quote",
     "statuses": ("active", "parked", "due_service", "retired"), "channel": ("radio", "sms"), "detail": "model", "values": ("van", "truck", "cargo bike", "pickup")},
    {"key": "studio", "topic": "a recording studio scheduler", "owner": "artist", "item": "session", "action": "reschedule", "amount": "fee",
     "statuses": ("booked", "tentative", "completed", "cancelled"), "channel": ("email", "app"), "detail": "room", "values": ("A", "B", "Live room", "Vocal booth")},
    {"key": "lab", "topic": "a research lab sample tracker", "owner": "researcher", "item": "sample", "action": "archive", "amount": "storage_cost",
     "statuses": ("received", "testing", "verified", "flagged"), "channel": ("email", "pager"), "detail": "assay", "values": ("PCR", "ELISA", "titration", "microscopy")},
    {"key": "warehouse", "topic": "a warehouse inbound desk", "owner": "supplier", "item": "shipment", "action": "release", "amount": "duty",
     "statuses": ("in_transit", "at_dock", "held", "released"), "channel": ("email", "edi"), "detail": "dock", "values": ("north", "south", "cold store", "yard")},
    {"key": "theater", "topic": "a community theater box office", "owner": "patron", "item": "seat_hold", "action": "upgrade", "amount": "price",
     "statuses": ("held", "paid", "expired", "refunded"), "channel": ("email", "sms"), "detail": "section", "values": ("stalls", "circle", "balcony", "box")},
    {"key": "marina", "topic": "a marina berth office", "owner": "boat_owner", "item": "berth_booking", "action": "extend", "amount": "berth_fee",
     "statuses": ("reserved", "occupied", "vacated", "waitlisted"), "channel": ("vhf", "email"), "detail": "pier", "values": ("pier 1", "pier 2", "outer pontoon", "dry dock")},
    {"key": "coworking", "topic": "a coworking space portal", "owner": "member", "item": "desk_booking", "action": "transfer", "amount": "charge",
     "statuses": ("confirmed", "pending", "checked_in", "void"), "channel": ("app", "email"), "detail": "floor", "values": ("2", "3", "rooftop", "basement")},
]
FIRST = "Aiko Bruno Carmen Dmitri Esme Femi Goran Hira Ivo Jana Kemal Lucia Mads Nia Oskar Pita Quinn Rosa Sven Tala Umar Vera Wen Yara".split()
LAST = "Abara Brandt Cruz Duarte Eklund Fofana Grieg Hosseini Ilic Juarez Kaur Lindqvist Mbeki Nakano Ortiz Petrov Quist Rahman Sato Toure".split()
FREE_TEXT_NOTE = "Short message for the {owner}."


def code(rng, prefix, n=6):
    return prefix + "".join(rng.choice("ABCDEFGHJKMNPQRSTUVWXYZ23456789") for _ in range(n))


def dollars(cents):
    return round(cents / 100, 2)


class Conv:
    def __init__(self, rng, domain, family, number, split):
        self.rng, self.d, self.family, self.split = rng, domain, family, split
        self.id = f"focus_{split}_{number}"
        self.messages, self.side = [], []
        d = domain
        o, i = d["owner"], d["item"]
        self.names = {  # tool and parameter names vary per conversation so names alone do not pick the tool
            "find": rng.choice((f"find_{o}", f"lookup_{o}", f"search_{o}s")),
            "whoami": rng.choice(("get_current_user", "whoami", f"current_{o}")),
            "list": rng.choice((f"list_{i}s", f"get_{i}s_for_{o}", f"{i}s_of_{o}")),
            "get": rng.choice((f"get_{i}", f"{i}_details", f"fetch_{i}")),
            "act": f"{d['action']}_{i}", "approve": f"request_{d['action']}_approval",
            "notify": rng.choice((f"notify_{o}", f"send_{o}_message", f"message_{o}")),
            "commit": rng.choice(("commit_changes", "sync_ledger", "finalize_queue")),
            "history": f"get_{i}_history", "bulk": f"{d['action']}_all_{i}s", "export": f"export_{o}_report",
        }
        self.oid_key, self.iid_key = rng.choice((f"{o}_id", "owner_id")), rng.choice((f"{i}_id", "item_id", "ref"))
        self.email_key = rng.choice(("email", "contact_email"))
        self.owner = {"name": f"{rng.choice(FIRST)} {rng.choice(LAST)}", "id": code(rng, "U-"), "email": None}
        self.owner["email"] = self.owner["name"].lower().replace(" ", ".") + rng.choice(("@mail.test", "@post.test", "@inbox.test"))
        self.limit = rng.choice((150, 200, 250, 300, 400)) * 100
        self.items = []
        for _ in range(rng.randint(3, 5)):
            self.items.append({"id": code(rng, d["key"][:2].upper() + "-"), "status": rng.choice(d["statuses"]), d["detail"]: rng.choice(d["values"]),
                               "amount": rng.randrange(40, 900) * 100 + rng.choice((0, 50, 25, 99))})
        self.system = None

    # ---- rendering helpers
    def tool_doc(self, key, description, props, required):
        return {"type": "function", "function": {"name": self.names[key], "description": description,
                                                 "parameters": {"type": "object", "properties": props, "required": required}}}

    def tools(self):
        d, o, i = self.d, self.d["owner"], self.d["item"]
        s = lambda desc: {"type": "string", "description": desc}  # noqa: E731
        docs = [
            self.tool_doc("find", f"Find a {o} by {self.email_key.replace('_', ' ')}. Returns the {o}'s id and name.", {self.email_key: s(f"The {o}'s email address.")}, [self.email_key]),
            self.tool_doc("whoami", f"Return the {o} who is signed in to this session.", {}, []),
            self.tool_doc("list", f"List every {i} of a {o} with its status, {d['detail']} and {d['amount']}.", {self.oid_key: s(f"Id of the {o}.")}, [self.oid_key]),
            self.tool_doc("get", f"Full details of one {i}.", {self.iid_key: s(f"Id of the {i}.")}, [self.iid_key]),
            self.tool_doc("act", f"{d['action'].capitalize()} one {i}. Only for amounts the policy allows without approval.",
                          {self.iid_key: s(f"Id of the {i}."), "reason": s("Why, in a few words.")}, [self.iid_key]),
            self.tool_doc("approve", f"Ask a supervisor to approve a {d['action']} that needs approval.",
                          {self.iid_key: s(f"Id of the {i}."), "amount": {"type": "number", "description": f"The {d['amount']} in dollars."}}, [self.iid_key, "amount"]),
            self.tool_doc("notify", f"Send a message to a {o}.", {self.oid_key: s(f"Id of the {o}."), "channel": {"type": "string", "enum": list(d["channel"])},
                                                                 "subject_ref": s(f"Id of the {i} the message is about."), "message": s("The text to send.")},
                          [self.oid_key, "channel", "subject_ref", "message"]),
            self.tool_doc("commit", "Write all pending changes of this session. Takes no arguments.", {}, []),
            self.tool_doc("history", f"Past status changes of one {i}.", {self.iid_key: s(f"Id of the {i}.")}, [self.iid_key]),
            self.tool_doc("bulk", f"{d['action'].capitalize()} every {i} of a {o} at once.", {self.oid_key: s(f"Id of the {o}.")}, [self.oid_key]),
            self.tool_doc("export", f"Export a PDF report of a {o}'s account.", {self.oid_key: s(f"Id of the {o}.")}, [self.oid_key]),
        ]
        if self.family[0] != "zero_arg":  # no signed-in session: the owner has to be found by email
            docs = [doc for doc in docs if doc["function"]["name"] != self.names["whoami"]]
        self.rng.shuffle(docs)
        return docs

    def say(self, role, content):
        self.messages.append({"role": role, "content": content})

    def call(self, key, arguments, sources, kind="call"):
        self.messages.append({"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": self.names[key], "arguments": arguments}}]})
        self.side.append({"step_id": f"{self.id}#{len(self.messages) - 1}", "split": self.split, "workflow_family": "|".join(self.fam_key()),
                          "expected_action": kind, "tool": self.names[key], "arguments": sources})

    def result(self, key, content):
        self.messages.append({"role": "tool", "name": self.names[key], "content": json.dumps(content, ensure_ascii=False)})

    def text(self, content, kind, reason):
        self.messages.append({"role": "assistant", "content": content})
        self.side.append({"step_id": f"{self.id}#{len(self.messages) - 1}", "split": self.split, "workflow_family": "|".join(self.fam_key()),
                          "expected_action": kind, "reason": reason})

    def fam_key(self):
        entry, middle, end = self.family
        return [entry, "+".join(middle) or "none", end]

    def view(self, item):
        return {"id": item["id"], "status": item["status"], self.d["detail"]: item[self.d["detail"]], self.d["amount"]: dollars(item["amount"])}

    # ---- the workflow
    # After the owner is known: list (shows id, detail, status, and the amount only when the
    # condition is about the amount) -> pick the one item the user described -> get (amount and
    # last update are only here) -> middle actions -> closing reply. Each call is needed: the user
    # never gives the item id, and the facts the reply or the policy needs are only in `get`.
    def build(self):
        rng, d = self.rng, self.d
        entry, middle, end = self.family
        o_word, i_word = d["owner"].replace("_", " "), d["item"].replace("_", " ")
        self.channel = rng.choice(d["channel"])
        self.default_channel = [c for c in d["channel"] if c != self.channel][0]
        self.note = rng.choice(("Your booking was checked.", "We have an update for you.", "Please see the latest status."))
        self.choose_target(middle, end)
        lines = [f"You are the assistant of {d['topic']}. Use the tools to look things up and to make changes; never invent ids."]
        if "policy_choice" in middle:
            lines.append(f"Policy: a {d['action']} whose {d['amount'].replace('_', ' ')} is above ${dollars(self.limit):.2f} must not be done directly; "
                         f"call {self.names['approve']} with that amount instead. At or below it, call {self.names['act']}.")
        if "many_sources" in middle:
            lines.append(f"Messages to a {o_word} go out by {self.channel} unless the {o_word} asks for another channel.")
        if rng.random() < 0.6 or len(lines) > 1:
            lines.append(f"Keep replies short. The desk is open {rng.choice(('8-16', '9-17', '10-18'))}.")
            self.say("system", "\n".join(lines))
        goal = self.goal_text(middle, end, i_word, o_word)

        if entry == "zero_arg":
            self.say("user", f"Hi, it's me. {goal}")
            self.call("whoami", {}, {})
        elif entry == "direct":
            self.say("user", f"I'm {self.owner['name']}, {self.owner['email']}. {goal}")
            self.call("find", {self.email_key: self.owner["email"]}, {self.email_key: {"from": "user", "kind": "direct"}})
        elif entry == "ask_first":
            self.say("user", f"Hello. {goal}")
            self.text(f"Sure. What is the email address on your {o_word} account?", "ask", f"{self.names['find']} needs {self.email_key}; no message gives it")
            self.say("user", f"It's {self.owner['email']}.")
            self.call("find", {self.email_key: self.owner["email"]}, {self.email_key: {"from": "user", "kind": "direct"}})
        elif entry == "corrected":
            wrong = self.owner["email"].replace("@", "1@")
            self.say("user", f"My email is {wrong} - sorry, no, it's {self.owner['email']}. {goal}")
            self.call("find", {self.email_key: self.owner["email"]}, {self.email_key: {"from": "user", "kind": "corrected", "not": wrong}})
        key = "whoami" if entry == "zero_arg" else "find"
        self.result(key, {self.oid_key: self.owner["id"], "name": self.owner["name"]})
        if end == "already_answered":
            self.text(f"Your {o_word} id is {self.owner['id']}, registered to {self.owner['name']}.", "finish", "the first result answers the question")
            return

        self.call("list", {self.oid_key: self.owner["id"]}, {self.oid_key: {"from": "tool", "kind": "direct"}})
        if end == "nothing_found":
            if "pick_by_condition" in middle:
                self.result("list", {"items": [self.row(x, with_amount=False) for x in self.items]})
                self.text(f"None of your {i_word}s is {self.wanted.replace('_', ' ')} right now. Which {i_word} did you mean?", "ask", "no listed item meets the stated condition")
            else:
                self.result("list", {"items": []})
                self.text(f"I don't see any {i_word}s on your account. Is there another email I should check?", "ask", "the list is empty")
            return
        by_amount = self.condition == "max_amount"
        self.result("list", {"items": [self.row(x, with_amount=by_amount) for x in self.items]})
        t = self.target
        kind = "condition" if "pick_by_condition" in middle else "described"
        self.call("get", {self.iid_key: t["id"]}, {self.iid_key: {"from": "tool", "kind": kind}})
        detail = {**self.row(t, with_amount=True), "updated": self.updated}
        if "earlier_value" in middle:
            detail["default_channel"] = self.default_channel
        self.result("get", detail)
        for step in middle:
            if step == "policy_choice":
                if t["amount"] > self.limit:
                    self.call("approve", {self.iid_key: t["id"], "amount": dollars(t["amount"])},
                              {self.iid_key: {"from": "tool", "kind": "direct"}, "amount": {"from": "tool", "kind": "direct"}})
                    self.result("approve", {"request_id": code(rng, "AP-"), "status": "pending_approval"})
                    self.outcome = "approval"
                else:
                    self.call("act", {self.iid_key: t["id"]}, {self.iid_key: {"from": "tool", "kind": "direct"}})
                    self.result("act", {"id": t["id"], "status": "done"})
                    self.outcome = "done"
            elif step in ("many_sources", "earlier_value"):
                channel = self.channel
                origin = {"from": "system", "kind": "direct"} if step == "many_sources" else {"from": "user", "kind": "earlier", "not": self.default_channel}
                self.call("notify", {self.oid_key: self.owner["id"], "channel": channel, "subject_ref": t["id"], "message": self.note},
                          {self.oid_key: {"from": "tool", "kind": "direct"}, "channel": origin, "subject_ref": {"from": "tool", "kind": "direct"},
                           "message": {"from": "user", "kind": "free_text"}})
                self.result("notify", {"sent": True, "channel": channel})
            elif step == "no_arg_tool":
                self.call("commit", {}, {})
                self.result("commit", {"committed": True})
        self.text(self.answer(middle, i_word), "finish", "the last result completes the request")

    def row(self, item, with_amount):
        r = {"id": item["id"], self.d["detail"]: item[self.d["detail"]], "status": item["status"]}
        if with_amount:
            r[self.d["amount"]] = dollars(item["amount"])
        return r

    def choose_target(self, middle, end):
        rng, d = self.rng, self.d
        self.updated = f"2026-{rng.randint(1, 10):02d}-{rng.randint(1, 28):02d}"
        values = rng.sample(d["values"], len(self.items)) if len(self.items) <= len(d["values"]) else None
        if values is None:
            self.items = self.items[:len(d["values"])]
            values = rng.sample(d["values"], len(self.items))
        for item, value in zip(self.items, values):  # the detail tells items apart
            item[d["detail"]] = value
        amounts = sorted({x["amount"] for x in self.items})
        if len(amounts) < len(self.items):
            for k, item in enumerate(self.items):
                item["amount"] += k * 7
        self.condition = None
        if "pick_by_condition" in middle:
            self.condition = "status" if end == "nothing_found" or rng.random() < 0.5 else "max_amount"
        self.target = rng.choice(self.items)
        if self.condition == "max_amount":
            self.target = max(self.items, key=lambda x: x["amount"])
        elif self.condition == "status":
            self.wanted = self.target["status"]
            for x in self.items:
                if x is not self.target and x["status"] == self.wanted:
                    x["status"] = rng.choice([s for s in d["statuses"] if s != self.wanted])
            if end == "nothing_found":
                self.wanted = rng.choice([s for s in d["statuses"] if all(x["status"] != s for x in self.items)] or ["archived"])
        if "policy_choice" in middle and rng.random() < 0.5:  # both sides of the limit, about evenly
            self.target["amount"] = self.limit + rng.randrange(1, 300) * 100 if self.target["amount"] <= self.limit else self.limit - rng.randrange(1, 100) * 100
        if self.condition == "max_amount":  # the change above must not take the target off the top
            for k, x in enumerate(self.items):
                if x is not self.target and x["amount"] >= self.target["amount"]:
                    x["amount"] = max(100 + k, self.target["amount"] - rng.randrange(1, 40) * 100 - k)

    def goal_text(self, middle, end, i_word, o_word):
        d, t = self.d, self.target
        if end == "already_answered":
            return f"What is my {o_word} id?"
        if self.condition == "max_amount":
            ref = f"my {i_word} with the highest {d['amount'].replace('_', ' ')}"
        elif self.condition == "status":
            ref = f"my {i_word} that is {self.wanted.replace('_', ' ')}"
        else:
            ref = f"my {t[d['detail']]} {i_word}" if d["detail"] not in ("room", "floor", "pier", "dock", "section") else f"my {i_word} in {d['detail']} {t[d['detail']]}"
        if end == "nothing_found" and "pick_by_condition" not in middle:
            return f"Can you tell me about {ref}?"
        wants = []
        if "policy_choice" in middle:
            wants.append(f"please {d['action'].replace('_', ' ')} {ref}")
        if "many_sources" in middle:
            wants.append(f"send me a note about it saying \"{self.note}\"")
        if "earlier_value" in middle:
            wants.append(f"message me about it by {self.channel}, not the usual channel, saying \"{self.note}\"")
        if "no_arg_tool" in middle:
            wants.append("save the changes when you're done")
        if not wants or wants[0].startswith(("send", "message", "save")):
            wants.insert(0, f"when was {ref} last updated")
        text = "; ".join(wants)
        return text[0].upper() + text[1:] + ("?" if text.startswith("when") and len(wants) == 1 else ".")

    def answer(self, middle, i_word):
        t = self.target
        amount = f"${dollars(t['amount']):.2f}"
        parts = []
        if "policy_choice" in middle:
            parts.append(f"{i_word.capitalize()} {t['id']} is {amount}, " + ("within the limit, so it is done." if self.outcome == "done" else "above the limit, so I sent it for approval."))
        else:
            parts.append(f"{i_word.capitalize()} {t['id']} was last updated on {self.updated}.")
        if "many_sources" in middle or "earlier_value" in middle:
            parts.append(f"I sent you the note by {self.channel}.")
        if "no_arg_tool" in middle:
            parts.append("The changes are saved.")
        return " ".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=sorted(FAMILIES), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--per-family", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sidecar", required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    n = 0
    with open(a.out, "x", encoding="utf-8") as out, open(a.sidecar, "x", encoding="utf-8") as side:
        for family in FAMILIES[a.split]:
            for _ in range(a.per_family):
                domain = rng.choice(DOMAINS)
                c = Conv(rng, domain, family, n, a.split)
                c.build()
                fam = "|".join(c.fam_key())
                out.write(json.dumps({"id": c.id, "source": "focus", "family": f"focus_{fam}", "task": "part1", "tools": c.tools(), "messages": c.messages}, ensure_ascii=False) + "\n")
                for s in c.side:
                    side.write(json.dumps(s, ensure_ascii=False) + "\n")
                n += 1
    print(f"{a.split}: {n} conversations, {len(FAMILIES[a.split])} families -> {a.out}")


if __name__ == "__main__":
    main()

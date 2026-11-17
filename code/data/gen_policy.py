"""Generate multi-turn service conversations that run under a long written policy.

    python code/data/gen_policy.py --conversations 3000 --seed 3 --out data_work/processed/policy_v2.jsonl

Each conversation gets its own operating policy, written out in the system message, a small
tool set and a small in-memory world. The rules change per conversation: how a person is
verified, what must happen before and after a change, whether to confirm first, when a case
goes to a supervisor or a person. The same request therefore ends differently under different
policies, and the next step can only be found by reading the policy. Tools really run on the
world, so every argument follows from the policy, the user or an earlier result. All policy
wording, tools and worlds are written here; nothing is read from the public test. `rules` keeps
the values the policy was written from, for code/data/check_policy.py, and never enters a prompt.
"""
import argparse
import collections
import json
import random

from common import first_problem, step_kinds
from gen_chains import DOMAINS, Conversation, Tool, World, a, code, date, plural, spoken

COMPANIES = ("Bluecrest", "Tallgrass", "Redwood Lane", "Cedar Point", "Silverleaf", "Brightwater", "Kestrel", "Maple Row", "Ironbridge", "Sunfield", "Larkspur", "Stonegate",
             "Harrowgate", "Palmetto", "Quillfeather", "Westmere", "Copperline", "Fernhill", "Oakhaven", "Marigold")
CATEGORIES = ("billing", "service quality", "safety", "technical problem", "staff conduct", "accessibility")
LEVELS = (("P1", "P2", "P3"), ("urgent", "high", "normal"), ("critical", "major", "minor"), ("A", "B", "C"))
SITUATIONS = {"charged twice": "I was charged twice", "not delivered": "I never received what I paid for", "poor quality": "it was really poor quality",
              "changed plans": "my plans changed", "price error": "the price was wrong"}
METHODS = ("original_payment", "store_credit", "bank_transfer", "account_balance")
WINDOWS = ("morning", "afternoon", "evening")
CHANNELS = ("email", "sms", "phone", "app")
TEMPLATES = ("change_confirmation", "account_update", "service_notice", "request_received")
CAPABILITIES = ("change", "refund", "move", "contact", "case")
DOW = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


class Rules:
    """The values one policy is written from."""

    def __init__(self, world, rng):
        self.capabilities = rng.sample(CAPABILITIES, rng.randint(2, 5))
        self.identity = rng.choice(("email", "name_zip", "member"))
        self.open_check = rng.choice((None, None, "clock", "status"))
        self.precheck = rng.random() < 0.5
        self.confirm = rng.random() < 0.55
        self.note_after = rng.random() < 0.45
        self.notify = rng.choice((None, None, *CHANNELS))
        self.template = rng.choice(TEMPLATES)
        self.credit_limit = rng.choice((50, 75, 100, 150, 200, 250, 300, 400, 500))
        self.credit_method = rng.choice((None, *METHODS))
        self.move_channel = rng.choice((None, *CHANNELS))
        self.case_before_transfer = rng.random() < 0.35
        situations = rng.sample(sorted(SITUATIONS), rng.randint(3, 5))
        self.reason_codes = {situation: f"{rng.choice(('RC', 'R-', 'REF', 'C'))}{rng.randint(10, 99)}" for situation in situations}
        levels = rng.choice(LEVELS)
        categories = rng.sample(CATEGORIES, rng.randint(3, 5))
        self.priorities = {category: rng.choice(levels) for category in categories}
        self.opens, self.closes = rng.choice((("08:00", "18:00"), ("09:00", "17:30"), ("07:30", "20:00"), ("10:00", "16:00")))
        self.days = rng.choice(("Monday to Friday", "Monday to Saturday", "every day"))
        self.hotline = f"+1-800-555-{rng.randint(1000, 9999)}"
        self.company = f"{rng.choice(COMPANIES)} {world.d.key.replace('_', ' ').title()}"

    def saved(self, tools):
        return {
            "capabilities": self.capabilities, "identity": self.identity, "open_check": self.open_check, "precheck": self.precheck, "confirm": self.confirm,
            "note_after": self.note_after, "notify": self.notify, "template": self.template, "credit_limit": self.credit_limit, "credit_method": self.credit_method,
            "move_channel": self.move_channel, "case_before_transfer": self.case_before_transfer, "reason_codes": self.reason_codes, "priorities": self.priorities,
            "tool_names": {key: tool.name for key, tool in tools.items()},
        }


class ServiceWorld(World):
    def __init__(self, domain, rng):
        super().__init__(domain, rng)
        self.rules = Rules(self, rng)
        self.today = date(rng)

    def make_top(self, name):
        top = super().make_top(name)
        rng = self.rng
        top.update(zip_code=f"{rng.randint(10000, 99999)}", member_code=code(rng, 7), birth_date=f"19{rng.randint(50, 99)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}", phone=f"+1-555-{rng.randint(1000, 9999)}")
        return top

    def open_status(self):
        return self.d.statuses[0]

    def make_tools(self):
        self.auth = False
        tools = super().make_tools()
        d, s, rng = self.d, self.style, self.rng
        id_kind = "integer" if s.int_ids else "string"
        top_id = ("top_id", s.id_field(d.top), id_kind, f"Identifier of the {spoken(d.top)}.", True)
        mid_id = ("mid_id", s.id_field(d.mid), id_kind, f"Identifier of the {spoken(d.mid)}.", True)

        def missing(entity):
            return {"error": f"No {spoken(entity)} with that identifier."}

        def find_person(name, zip_code, **_):
            return {"results": [self.show_top(t) for t in self.tops if t["name"].lower() == name.lower() and t["zip_code"] == zip_code]}

        def verify_member(member_code, birth_date, **_):
            top = next((t for t in self.tops if t["member_code"] == member_code and t["birth_date"] == birth_date), None)
            return {"verified": True, **self.show_top(top)} if top else {"verified": False}

        def check(mid_id, **_):
            mid = self.by_id(self.mids(), mid_id)
            if not mid:
                return missing(d.mid)
            if mid["status"] == d.statuses[0]:
                return {"eligible": True, "status": mid["status"]}
            return {"eligible": False, "status": mid["status"], "reason": f"Only {a(d.statuses[0])} {spoken(d.mid)} qualifies."}

        def quote(mid_id, **_):
            mid = self.by_id(self.mids(), mid_id)
            if not mid:
                return missing(d.mid)
            amount = 0.0 if mid["status"] == d.statuses[2] else mid[d.amount]
            return {"refundable_amount": amount, "currency": "USD", "status": mid["status"]}

        def credit(mid_id, amount, method, reason_code, **_):
            return {"refund_id": code(rng, 9), "amount": amount, "method": method, "state": "issued"}

        def move(mid_id, new_date, time_window, contact_channel, **_):
            mid = self.by_id(self.mids(), mid_id)
            if not mid:
                return missing(d.mid)
            mid["date"] = new_date
            return {"success": True, "new_date": new_date, "time_window": time_window}

        def update_contact(top_id, field, value, **_):
            top = self.by_id(self.tops, top_id)
            if not top:
                return missing(d.top)
            top[field] = value
            return {"success": True, "updated": field}

        def open_case(top_id, category, priority, summary, **_):
            return {"case_id": f"CASE-{rng.randint(10000, 99999)}", "state": "open", "priority": priority}

        def note(top_id, note, **_):
            return {"saved": True}

        def notify(top_id, channel, template, **_):
            return {"queued": True, "channel": channel}

        def escalate(top_id, mid_id, amount, reason, **_):
            return {"ticket": f"SUP-{rng.randint(1000, 9999)}", "state": "awaiting supervisor"}

        def transfer(summary, **_):
            return {"transferred": True, "queue_position": rng.randint(1, 9)}

        def clock():
            return {"date": self.today}

        def status():
            return {"state": "operational", "maintenance": False}

        def fees():
            return {"late_fee": round(rng.uniform(2, 30), 2), "change_fee": round(rng.uniform(5, 45), 2), "currency": "USD"}

        verb = f"{d.verb}_{d.mid}"
        tools.update({
            "find_person": Tool(s.name(s.find.format(f"{d.top}_by_name_and_zip")), f"Find {a(d.top)} by full name and zip code.",
                                [("name", "full_name", "string", "Full name.", True), ("zip_code", "zip_code", "string", "Five-digit zip code.", True)], find_person),
            "verify_member": Tool(s.name(rng.choice(("verify_member", "check_member_code", "authenticate_member"))), "Verify a member code against a date of birth.",
                                  [("member_code", "member_code", "string", "Member code.", True), ("birth_date", "date_of_birth", "string", "Date of birth, YYYY-MM-DD.", True)], verify_member),
            "check": Tool(s.name(rng.choice((f"check_{verb}_eligibility", f"can_{verb}", f"{d.mid}_eligibility"))), f"Check whether {a(d.mid)} may be changed.", [mid_id], check),
            "quote": Tool(s.name(rng.choice((f"quote_{d.mid}_refund", "get_refund_quote", "calculate_refund"))), f"Work out what can be refunded on {a(d.mid)}.", [mid_id], quote),
            "credit": Tool(s.name(rng.choice(("issue_refund", "create_refund", "process_refund"))), f"Refund {a(d.mid)}.",
                           [mid_id, ("amount", "amount", "number", "Amount in USD.", True), ("method", "refund_method", "string", "How the money is returned.", True),
                            ("reason_code", "reason_code", "string", "Reason code from the policy.", True)], credit),
            "move": Tool(s.name(rng.choice((f"reschedule_{d.mid}", f"move_{d.mid}", f"change_{d.mid}_date"))), f"Give {a(d.mid)} a new date.",
                         [mid_id, ("new_date", "new_date", "string", "New date, YYYY-MM-DD.", True), ("time_window", "time_window", "string", "morning, afternoon or evening.", True),
                          ("contact_channel", "contact_channel", "string", "How to reach the person about the change.", True)], move),
            "update_contact": Tool(s.name(rng.choice(("update_contact_details", "change_contact_info", "set_contact_field"))), f"Change a contact field on {a(d.top)}.",
                                   [top_id, ("field", "field", "string", "phone or email.", True), ("value", "value", "string", "New value.", True)], update_contact),
            "open_case": Tool(s.name(rng.choice(("open_case", "create_case", "file_complaint"))), "Open a case for follow-up.",
                              [top_id, ("category", "category", "string", "Case category.", True), ("priority", "priority", "string", "Priority from the policy.", True),
                               ("summary", "summary", "string", "One line about the problem.", True)], open_case),
            "note": Tool(s.name(rng.choice(("add_account_note", "log_note", "write_case_note"))), "Add a note to an account.", [top_id, ("note", "note", "string", "Note text.", True)], note),
            "notify": Tool(s.name(rng.choice(("send_notification", "queue_notice", "notify_customer"))), "Send a notification.",
                           [top_id, ("channel", "channel", "string", "Delivery channel.", True), ("template", "template", "string", "Template name.", True)], notify),
            "escalate": Tool(s.name(rng.choice(("escalate_to_supervisor", "request_supervisor_review", "raise_to_manager"))), "Send a request to a supervisor.",
                             [top_id, mid_id, ("amount", "amount", "number", "Amount in USD.", True), ("reason", "reason", "string", "Why it is escalated.", True)], escalate),
            "transfer": Tool(s.name(rng.choice(("transfer_to_human", "handoff_to_agent", "connect_live_agent"))), "Hand the conversation to a person.",
                             [("summary", "summary", "string", "What the person should know.", True)], transfer),
            "clock": Tool(s.name(rng.choice(("get_current_date", "today", "get_system_date"))), "Return today's date.", [], clock),
            "status": Tool(s.name(rng.choice(("get_system_status", "service_health", "check_service_status"))), "Return whether the back office systems are up.", [], status),
            "fees": Tool(s.name(rng.choice(("get_fee_schedule", "list_fees", "current_fees"))), "Return the current fees.", [], fees),
        })
        return tools


# ---- the policy text


def phrase(rng, *options):
    return rng.choice(options)


def identity_section(w, rng):
    r, t = w.rules, w.tools
    if r.identity == "email":
        how = phrase(rng, f"Ask for the email address on the account and look it up with `{t['find_email'].name}`.",
                     f"Identify the person by the email on file: call `{t['find_email'].name}` with exactly what they give you.",
                     f"The only accepted identifier is the account email, passed to `{t['find_email'].name}`.")
    elif r.identity == "name_zip":
        how = phrase(rng, f"Ask for both the full name and the zip code, then call `{t['find_person'].name}`. One of the two is not enough.",
                     f"You need the person's full name together with their zip code before calling `{t['find_person'].name}`.",
                     f"Identify the person with `{t['find_person'].name}`, which takes their full name and zip code. Collect both first.")
    else:
        how = phrase(rng, f"Ask for the member code and the date of birth, then call `{t['verify_member'].name}`. Both are required.",
                     f"Call `{t['verify_member'].name}` with the member code and date of birth. Do not call it until you have both.",
                     f"A person is verified only when `{t['verify_member'].name}` returns verified for their member code and date of birth.")
    rest = phrase(rng, "Verify once per conversation, before you look up or change anything. If the lookup finds nothing, say so and ask the person to check the details.",
                  "This comes first and happens once in a conversation. When nothing matches, ask them to re-check what they gave you; never guess.",
                  "Nothing about an account may be read or changed before this step succeeds, and it does not need repeating later in the same conversation. If it fails, ask for the details again.")
    return "Verifying the person", f"{how} {rest}"


def opening_section(w, rng):
    name = w.tools[w.rules.open_check].name
    return "Starting a conversation", phrase(rng, f"Before using any other tool in a conversation, call `{name}` once. It takes no arguments.",
                                             f"Your first tool call in every conversation is `{name}`, with no arguments. Do it only once.",
                                             f"Call `{name}` once at the beginning, before every other tool, even if the request seems simple.")


def after_change_lines(w, rng):
    r, t = w.rules, w.tools
    lines = []
    if r.note_after:
        lines.append(phrase(rng, f"Record what was done with `{t['note'].name}`.", f"Add a short account note using `{t['note'].name}`."))
    if r.notify:
        lines.append(phrase(rng, f"Send a notice with `{t['notify'].name}`, channel `{r.notify}`, template `{r.template}`.", f"Call `{t['notify'].name}` using channel `{r.notify}` and template `{r.template}`."))
    return lines


def confirm_line(w, rng):
    return phrase(rng, "State exactly what you are about to do and wait for a clear yes.", "Describe the change and get the person's explicit agreement before going on.", "Repeat the details back, ask if you should go ahead, and wait for them to say yes.")


def numbered(steps, rng):
    if rng.random() < 0.5:
        return "\n".join(f"{index}. {step}" for index, step in enumerate(steps, 1))
    return "\n".join(f"- {step}" for step in steps)


def change_section(w, rng):
    r, t, d = w.rules, w.tools, w.d
    steps = [f"Find the {spoken(d.mid)} with `{t['list_mids'].name}`."]
    if r.precheck:
        steps.append(f"Call `{t['check'].name}` for that {spoken(d.mid)}. If it is not eligible, tell the person why and stop.")
    else:
        steps.append(f"Look at the status yourself: only {a(d.statuses[0])} {spoken(d.mid)} can be changed. For any other status, tell the person and stop.")
    if r.confirm:
        steps.append(confirm_line(w, rng))
    steps.append(f"Call `{t['act'].name}`, passing the reason the person gave.")
    steps += after_change_lines(w, rng)
    steps.append("Tell the person it is done.")
    return f"Requests to {d.verb} {a(d.mid)}", numbered(steps, rng)


def refund_section(w, rng):
    r, t, d = w.rules, w.tools, w.d
    steps = [f"Find the {spoken(d.mid)} with `{t['list_mids'].name}` and get the amount from `{t['quote'].name}`.", "If the amount is zero, explain that nothing can be refunded and stop."]
    steps.append(f"If the amount is more than ${r.credit_limit}, you may not refund it yourself: call `{t['escalate'].name}` instead, tell the person a supervisor will review it, and stop.")
    if r.credit_method:
        steps.append(f"Refunds always go to `{r.credit_method}`.")
    else:
        steps.append(f"Ask how they want the money back. The options are {', '.join(f'`{method}`' for method in METHODS)}.")
    if r.confirm:
        steps.append(confirm_line(w, rng))
    steps.append(f"Call `{t['credit'].name}` with the quoted amount and the reason code below.")
    steps += after_change_lines(w, rng)
    steps.append("Tell the person the refund id.")
    codes = "\n".join(f"- {situation}: `{value}`" for situation, value in r.reason_codes.items())
    return "Refunds", f"{numbered(steps, rng)}\n\nReason codes:\n{codes}"


def move_section(w, rng):
    r, t, d = w.rules, w.tools, w.d
    steps = [f"The person must give the new date and a time window ({', '.join(WINDOWS)}). Ask for whatever is missing.", f"Find the {spoken(d.mid)} with `{t['list_mids'].name}`."]
    if r.move_channel:
        steps.append(f"Use `{r.move_channel}` as the contact channel.")
    else:
        steps.append(f"Ask which contact channel they prefer: {', '.join(CHANNELS)}.")
    if r.confirm:
        steps.append(confirm_line(w, rng))
    steps.append(f"Call `{t['move'].name}`.")
    steps += after_change_lines(w, rng)
    steps.append("Repeat the new date and window to the person.")
    return f"Moving {a(d.mid)} to another date", numbered(steps, rng)


def contact_section(w, rng):
    t = w.tools
    steps = [f"Call `{t['update_contact'].name}` with field `phone` or `email` and the new value the person gave."]
    steps += after_change_lines(w, rng)
    steps.append("Tell the person it is updated.")
    return "Changing contact details", numbered(steps, rng)


def case_section(w, rng):
    r, t = w.rules, w.tools
    table = "\n".join(f"- {category}: priority `{level}`" for category, level in r.priorities.items())
    body = phrase(rng, f"When a person reports a problem, open a case with `{t['open_case'].name}`. Use the category that fits and the priority listed for it.",
                  f"Complaints go into `{t['open_case'].name}`. Pick the matching category below and copy its priority exactly.")
    return "Complaints and problems", f"{body} Give the person the case id.\n\n{table}"


def handover_section(w, rng):
    r, t = w.rules, w.tools
    if r.case_before_transfer and "case" in r.capabilities:
        body = f"If a person asks for a human, first open a case with `{t['open_case'].name}` (category and priority as listed), then call `{t['transfer'].name}`."
    else:
        body = phrase(rng, f"If a person asks for a human, call `{t['transfer'].name}` with a one-line summary.", f"Call `{t['transfer'].name}` as soon as someone asks to speak with a person; include a short summary.")
    return "Handing over to a person", f"{body} Do not transfer for any other reason."


def scope_section(w, rng):
    r, d = w.rules, w.d
    offered = {"change": f"requests to {d.verb} {plural(spoken(d.mid))}", "refund": "refunds", "move": f"moving {plural(spoken(d.mid))} to a new date", "contact": "contact detail changes", "case": "complaints"}
    allowed = ", ".join(offered[key] for key in r.capabilities)
    return "What you can do", phrase(rng, f"You help with status questions and with: {allowed}. Politely decline anything else, including requests about other people's accounts, without using a tool.",
                                     f"Beyond answering questions about a person's own {plural(spoken(d.mid))}, you handle {allowed}. Any other request is out of scope: decline it in a sentence. Never discuss someone else's account.")


def info_section(w, rng):
    r = w.rules
    text = phrase(rng, f"Support hours are {r.opens} to {r.closes}, {r.days}. The hotline is {r.hotline}.", f"We are open {r.days}, {r.opens}-{r.closes}. Phone support: {r.hotline}.")
    if "fees" in w.chosen:
        text += f" Current fees are not listed here; call `{w.tools['fees'].name}` when someone asks, and do not state a fee from memory."
    return "Practical information", text


def glossary_section(w, rng):
    d = w.d
    lines = [f"- {spoken(status)}: {phrase(rng, 'current state of the record', 'as shown by the system', 'as recorded in the back office')}" for status in d.statuses]
    return "Status words", "\n".join(lines)


FILLERS = (
    ("Tone", ("Be courteous and brief. Use plain language and short sentences. Do not use slang.", "Keep replies short and friendly. One or two sentences is usually enough. Avoid jargon and do not repeat the question back.",
              "Write the way a calm, experienced colleague would. Be direct, skip filler, and only offer next steps that are actually available.")),
    ("Privacy", ("Share an account's details only with the verified person. Do not read out full payment details. If someone seems to be acting for another person, treat it as out of scope.",
                 "Information stays with the account holder. Never confirm whether an account exists to anyone who is not verified.")),
    ("Accuracy", ("State only what a tool returned or what this policy says. If you do not know, say so. Never invent identifiers, dates or amounts.",
                  "Base every answer on tool results or on this document. Copy identifiers exactly as they appear, and do not round amounts.")),
    ("Using tools", ("Make one tool call at a time and read the result before deciding the next step. Do not repeat a lookup you already made in this conversation.",
                     "Work one step at a time. Results from earlier in the conversation stay valid, so never look up the same thing twice.")),
    ("Records", ("Everything you do is logged and reviewed by the quality team each week. Reviewers check that the steps here were followed in order.",
                 "Audits sample conversations at random, and skipped steps are treated as errors even when the outcome was right.")),
    ("Closing a conversation", ("When a person says they are finished, thank them and end with a short closing line. Do not add new offers.",
                                "If nothing else is needed, close politely in one sentence.")),
)


def policy_text(w, rng):
    r = w.rules
    head = phrase(rng, f"You are the service agent for {r.company}, {w.d.topic}.", f"This document is the operating policy for {r.company}, {w.d.topic}. You are its virtual agent.",
                  f"Role: customer support agent at {r.company} ({w.d.topic}).", f"You work the support desk at {r.company}, which is {w.d.topic}. Follow this policy exactly.")
    head += " " + phrase(rng, f"Today is {w.today}.", f"The date is {w.today}.") if not r.open_check == "clock" else ""
    sections = [scope_section(w, rng), identity_section(w, rng)]
    if r.open_check:
        sections.insert(rng.randint(0, 1), opening_section(w, rng))
    procedures = {"change": change_section, "refund": refund_section, "move": move_section, "contact": contact_section, "case": case_section}
    middle = [procedures[key](w, rng) for key in r.capabilities]
    middle.append(handover_section(w, rng))
    extras = [info_section(w, rng)]
    if rng.random() < 0.7:
        extras.append(glossary_section(w, rng))
    for title, options in rng.sample(FILLERS, rng.randint(2, len(FILLERS))):
        extras.append((title, rng.choice(options)))
    rng.shuffle(extras)
    sections += middle + extras

    form = rng.choice(("markdown", "caps", "numbered", "plain"))
    parts = [head]
    for index, (title, body) in enumerate(sections, 1):
        parts.append({"markdown": f"## {title}\n{body}", "caps": f"{title.upper()}\n{body}", "numbered": f"{index}. {title}\n{body}", "plain": f"{title}: {body}"}[form])
    return "\n\n".join(parts)


# ---- the conversation


class Session(Conversation):
    def __init__(self, world, rng):
        super().__init__(world, rng)
        self.rules = world.rules
        self.person = None
        self.started = False
        self.listed = False

    def use(self, key, **arguments):
        if self.rules.open_check and not self.started:
            self.started = True
            self.call(self.rules.open_check)
        return self.call(key, **arguments)

    def lookup(self, top, wrong=False):
        rule = self.rules.identity
        if rule == "email":
            email = "x" + top["email"] if wrong else top["email"]
            return self.use("find_email", email=email)
        if rule == "name_zip":
            return self.use("find_person", name=top["name"], zip_code=str(int(top["zip_code"]) + 1) if wrong else top["zip_code"])
        return self.use("verify_member", member_code=top["member_code"], birth_date=top["birth_date"].replace("19", "20", 1) if wrong else top["birth_date"])

    def details(self, top, wrong=False):
        rule, rng = self.rules.identity, self.rng
        if rule == "email":
            email = "x" + top["email"] if wrong else top["email"]
            return phrase(rng, f"My email is {email}.", f"It's under {email}.", f"The address on file is {email}.")
        if rule == "name_zip":
            zip_code = str(int(top["zip_code"]) + 1) if wrong else top["zip_code"]
            return phrase(rng, f"I'm {top['name']}, zip code {zip_code}.", f"Name's {top['name']} and my zip is {zip_code}.")
        birth = top["birth_date"].replace("19", "20", 1) if wrong else top["birth_date"]
        return phrase(rng, f"Member code {top['member_code']}, born {birth}.", f"My member code is {top['member_code']} and my date of birth is {birth}.")

    def ask_details(self):
        rule, rng = self.rules.identity, self.rng
        if rule == "email":
            return phrase(rng, "Sure. What email address is on the account?", "Happy to help. Could you give me the email on your account first?")
        if rule == "name_zip":
            return phrase(rng, "Of course. I need your full name and zip code first.", "Sure. Please tell me your full name and your zip code.")
        return phrase(rng, "Certainly. Please give me your member code and date of birth.", "Before I can do that, I need your member code and your date of birth.")

    def identify(self, top, offered):
        """`offered`: the details are already in the user's message."""
        if self.person is top:
            return
        if not offered:
            self.reply(self.ask_details())
            self.user(self.details(top))
        self.lookup(top)
        self.person = top

    def request(self, text, top, offered, wrong=False):
        """The user's message. With `offered`, it carries the verification details."""
        needed = offered and self.person is not top
        self.user(f"{text} {self.details(top, wrong)}" if needed else text)
        return offered or self.person is top

    def retry_identity(self, top):
        """The first details are wrong. Returns after a successful lookup."""
        self.lookup(top, wrong=True)
        self.reply(self.rng.choice(("I couldn't find a match for those details. Could you check them and try again?", "Nothing matches that. Please double-check and send them again.")))
        self.user(self.rng.choice(("Sorry, typo.", "Oops, let me fix that.", "My mistake.")) + " " + self.details(top))
        self.lookup(top)
        self.person = top

    def list_once(self, top):
        if not self.listed:
            self.listed = True
            self.use("list_mids", top_id=top["id"])

    def confirm(self, what):
        """Returns False when the person backs out."""
        if not self.rules.confirm:
            return True
        self.reply(self.rng.choice((f"To confirm: {what}. Shall I go ahead?", f"I'm about to do this: {what}. Is that okay?", f"Just checking: {what}. Do you want me to proceed?")))
        if self.rng.random() < 0.1:
            self.user(self.rng.choice(("Actually no, leave it.", "No, never mind.", "Hold on, don't do that.")))
            self.reply(self.rng.choice(("No problem, I've left everything as it is.", "Okay, nothing has been changed.")))
            return False
        self.user(self.rng.choice(("Yes, proceed with that.", "Yes please.", "Yes, do that.", "That's right, do it.", "Confirmed.")))
        return True

    def after_change(self, top, note):
        r = self.rules
        if r.note_after:
            self.use("note", top_id=top["id"], note=note)
        if r.notify:
            self.use("notify", top_id=top["id"], channel=r.notify, template=r.template)

    def label(self, mid):
        return mid[self.d.label[0]]

    def named(self, mid):
        return f"{spoken(self.d.mid)} for {self.label(mid)}"


# ---- what a person can ask for


def status_question(c, top, offered):
    mid = c.rng.choice(top["mids"])
    ask = phrase(c.rng, f"What's the status of my {c.named(mid)}?", f"Can you tell me where my {c.named(mid)} stands?", f"Hi, I want to check on my {c.named(mid)}.")
    c.identify(top, c.request(ask, top, offered))
    c.list_once(top)
    c.reply(phrase(c.rng, f"Your {c.named(mid)} is {spoken(mid['status'])}.", f"It shows as {spoken(mid['status'])}, dated {mid['date']}."))


def leaf_question(c, top, offered):
    mid = c.rng.choice(top["mids"])
    field = c.rng.choice(sorted(mid["leaf"]))
    ask = phrase(c.rng, f"What's the {spoken(field)} on the {spoken(c.d.leaf)} for my {c.named(mid)}?", f"I need the {spoken(field)} for my {c.named(mid)}.")
    c.identify(top, c.request(ask, top, offered))
    c.list_once(top)
    result = c.use("get_leaf", mid_id=mid["id"])
    c.reply(phrase(c.rng, f"The {spoken(field)} is {result[field]}.", f"It's {result[field]}."))


def change_request(c, top, offered):
    d, rules = c.d, c.rules
    open_ones = [m for m in top["mids"] if m["status"] == d.statuses[0]]
    mid = c.rng.choice(open_ones) if open_ones and c.rng.random() < 0.75 else c.rng.choice(top["mids"])
    reason = phrase(c.rng, "I no longer need it", "my plans changed", "I found another option", "it's not working for me")
    ask = phrase(c.rng, f"I'd like to {d.verb} my {c.named(mid)}, {reason}.", f"Please {d.verb} my {c.named(mid)}. The reason is {reason}.")
    c.identify(top, c.request(ask, top, offered))
    c.list_once(top)
    if rules.precheck:
        result = c.use("check", mid_id=mid["id"])
        if not result["eligible"]:
            c.reply(f"I can't {d.verb} it: it is {spoken(result['status'])}, and {result['reason'][0].lower()}{result['reason'][1:]}")
            return
    elif mid["status"] != d.statuses[0]:
        c.reply(f"I can't {d.verb} it, because it is {spoken(mid['status'])} and only {a(d.statuses[0])} {spoken(d.mid)} can be changed.")
        return
    if not c.confirm(f"{d.verb} your {c.named(mid)}"):
        return
    c.use("act", mid_id=mid["id"], reason=reason)
    c.after_change(top, f"{d.verb} {mid['id']}: {reason}")
    c.reply(phrase(c.rng, f"Done, your {c.named(mid)} is now {spoken(d.closed)}.", f"That's taken care of. The {c.named(mid)} is {spoken(d.closed)}."))


def refund_request(c, top, offered):
    rules = c.rules
    mid = c.rng.choice(top["mids"])
    situation = c.rng.choice(sorted(rules.reason_codes))
    ask = phrase(c.rng, f"I want a refund on my {c.named(mid)}, {SITUATIONS[situation]}.", f"Can I get my money back for the {c.named(mid)}? {SITUATIONS[situation].capitalize()}.")
    c.identify(top, c.request(ask, top, offered))
    c.list_once(top)
    quote = c.use("quote", mid_id=mid["id"])
    amount = quote["refundable_amount"]
    if not amount:
        c.reply(f"Nothing can be refunded on this one, the quote is $0 because it is {spoken(quote['status'])}.")
        return
    if amount > rules.credit_limit:
        result = c.use("escalate", top_id=top["id"], mid_id=mid["id"], amount=amount, reason=situation)
        c.reply(f"The amount is above what I can refund, so a supervisor will review it. Your ticket is {result['ticket']}.")
        return
    method = rules.credit_method
    if method is None:
        c.reply(phrase(c.rng, f"You can get ${amount} back. Which option do you want: {', '.join(METHODS)}?", f"The refund comes to ${amount}. How would you like it: {', '.join(METHODS)}?"))
        method = c.rng.choice(METHODS)
        c.user(phrase(c.rng, f"{method}, please.", f"Let's do {method}."))
    if not c.confirm(f"refund ${amount} to {method} for your {c.named(mid)}"):
        return
    result = c.use("credit", mid_id=mid["id"], amount=amount, method=method, reason_code=rules.reason_codes[situation])
    c.after_change(top, f"refund {result['refund_id']} for {mid['id']}")
    c.reply(phrase(c.rng, f"Your refund of ${amount} is issued. The refund id is {result['refund_id']}.", f"All set, refund {result['refund_id']} for ${amount} has been issued."))


def move_request(c, top, offered):
    rules = c.rules
    mid = c.rng.choice(top["mids"])
    new_date, window = date(c.rng), c.rng.choice(WINDOWS)
    has_all = c.rng.random() < 0.6
    ask = f"Can you move my {c.named(mid)} to {new_date}" + (f", in the {window}?" if has_all else "?")
    c.identify(top, c.request(ask, top, offered))
    if not has_all:
        c.reply(phrase(c.rng, f"Which time window works for you: {', '.join(WINDOWS)}?", f"Would you like the {WINDOWS[0]}, {WINDOWS[1]} or {WINDOWS[2]}?"))
        c.user(phrase(c.rng, f"The {window}.", f"{window.capitalize()}, please."))
    c.list_once(top)
    channel = rules.move_channel
    if channel is None:
        c.reply(phrase(c.rng, f"How should we reach you about it: {', '.join(CHANNELS)}?", f"Which contact channel do you prefer: {', '.join(CHANNELS)}?"))
        channel = c.rng.choice(CHANNELS)
        c.user(phrase(c.rng, f"By {channel}.", f"{channel} works."))
    if not c.confirm(f"move your {c.named(mid)} to {new_date} in the {window}"):
        return
    c.use("move", mid_id=mid["id"], new_date=new_date, time_window=window, contact_channel=channel)
    c.after_change(top, f"moved {mid['id']} to {new_date}")
    c.reply(phrase(c.rng, f"Your {c.named(mid)} is now on {new_date} in the {window}.", f"Done. New date: {new_date}, {window}."))


def contact_request(c, top, offered):
    field = c.rng.choice(("phone", "email"))
    value = f"+1-555-{c.rng.randint(1000, 9999)}" if field == "phone" else f"{top['name'].split()[0].lower()}{c.rng.randint(10, 99)}@mail.example.net"
    ask = phrase(c.rng, f"Please change my {field} to {value}.", f"I have a new {field}: {value}. Can you update it?")
    c.identify(top, c.request(ask, top, offered))
    c.use("update_contact", top_id=top["id"], field=field, value=value)
    c.after_change(top, f"{field} updated")
    c.reply(phrase(c.rng, f"Your {field} is updated.", f"Done, I've changed your {field} to {value}."))


COMPLAINTS = {"billing": "my bill looks wrong", "service quality": "the service was terrible", "safety": "I felt unsafe at your location", "technical problem": "your website keeps failing for me",
              "staff conduct": "one of your staff was rude to me", "accessibility": "I couldn't get in with my wheelchair"}


def complaint(c, top, offered):
    category = c.rng.choice(sorted(c.rules.priorities))
    text = COMPLAINTS[category]
    ask = phrase(c.rng, f"I want to complain, {text}.", f"I need to report something: {text}.")
    c.identify(top, c.request(ask, top, offered))
    result = c.use("open_case", top_id=top["id"], category=category, priority=c.rules.priorities[category], summary=text)
    c.reply(phrase(c.rng, f"I'm sorry about that. Your case id is {result['case_id']}.", f"Thanks for telling us. I've opened {result['case_id']} for this."))


def human_request(c, top, offered):
    rules = c.rules
    ask = phrase(c.rng, "I'd like to talk to a real person.", "Can you get me a human please?", "Connect me to an agent.")
    if rules.case_before_transfer and "case" in rules.capabilities:
        category = c.rng.choice(sorted(rules.priorities))
        text = COMPLAINTS[category]
        c.identify(top, c.request(f"{ask} {text.capitalize()}.", top, offered))
        c.use("open_case", top_id=top["id"], category=category, priority=rules.priorities[category], summary=text)
        summary = f"{category}: {text}"
    else:
        c.user(ask)
        summary = "The person asked for a human."
    result = c.use("transfer", summary=summary)
    c.reply(f"You're being connected now. You are number {result['queue_position']} in the queue.")


def hours_question(c, top, offered):
    r = c.rules
    c.user(phrase(c.rng, "What are your support hours?", "When can I reach you by phone?", "What's your hotline number, and when are you open?"))
    c.reply(phrase(c.rng, f"We're available {r.days}, {r.opens} to {r.closes}. The hotline is {r.hotline}.", f"You can call {r.hotline}, {r.days}, between {r.opens} and {r.closes}."))


def fee_question(c, top, offered):
    c.user(phrase(c.rng, "How much is the late fee?", "What's the change fee these days?"))
    result = c.use("fees")
    c.reply(f"The late fee is ${result['late_fee']} and the change fee is ${result['change_fee']}.")


OTHER_TOPICS = ("Can you recommend a good place for dinner nearby?", "What's the weather like tomorrow?", "Could you help me write a poem for my sister?", "Who won the game last night?", "Can you translate this sentence into French for me?")


def out_of_scope(c, top, offered):
    c.user(c.rng.choice(OTHER_TOPICS))
    c.reply(phrase(c.rng, f"Sorry, I can't help with that. I can help with your {plural(spoken(c.d.mid))} at {c.rules.company}.", "That's outside what I can do here, sorry. Is there anything about your account I can help with?"))


def other_account(c, top, offered):
    c.user(phrase(c.rng, f"Can you look up my neighbor's {spoken(c.d.mid)}? They asked me to.", f"My brother has an account too. What's the status of his {spoken(c.d.mid)}?"))
    c.reply(phrase(c.rng, "I'm sorry, I can only discuss your own account.", "I can't share anything about someone else's account."))


def not_offered(c, top, offered):
    d, r = c.d, c.rules
    asks = {"change": f"Please {d.verb} my {spoken(d.mid)}.", "refund": f"I want a refund for my {spoken(d.mid)}.", "move": f"Can you move my {spoken(d.mid)} to another date?",
            "contact": "I need to change my phone number.", "case": "I have a complaint I want on record."}
    missing = [key for key in CAPABILITIES if key not in r.capabilities]
    c.user(asks[c.rng.choice(missing)])
    c.reply(phrase(c.rng, f"I'm sorry, that isn't something I can do here. You can call {r.hotline} during support hours.", "Sorry, that's not something I can handle in this channel."))


def wrong_details(c, top, offered):
    """The first details do not match. Only used while the person is not yet verified."""
    mid = c.rng.choice(top["mids"])
    ask = phrase(c.rng, f"What's going on with my {c.named(mid)}?", f"Could you check my {c.named(mid)}?")
    c.user(f"{ask} {c.details(top, wrong=True)}")
    c.retry_identity(top)
    c.list_once(top)
    c.reply(f"Thanks, I found it. Your {c.named(mid)} is {spoken(mid['status'])}.")


def refuses_details(c, top, offered):
    mid = c.rng.choice(top["mids"])
    c.user(f"Just tell me about my {c.named(mid)}, I don't want to verify anything.")
    c.reply(phrase(c.rng, "I understand, but I can't look anything up until you are verified.", "Sorry, I need to verify you before I can open an account."))
    c.user(c.rng.choice(("Fine, forget it.", "Okay, never mind then.", "That's annoying, but okay. Bye.")))
    c.reply(phrase(c.rng, "No problem. Take care.", "All right. Thanks for contacting us."))


def closing(c):
    c.user(c.rng.choice(("That's all, thanks.", "Great, thank you.", "Nothing else, bye.", "Much appreciated, thanks.")))
    c.reply(c.rng.choice(("You're welcome. Take care.", "Thanks for contacting us. Goodbye.", "Happy to help. Have a good day.")))


SCENARIOS = {"change": change_request, "refund": refund_request, "move": move_request, "contact": contact_request, "case": complaint}


def choose_requests(c, rng):
    r = c.rules
    pool = [SCENARIOS[key] for key in r.capabilities for _ in range(4)]
    pool += [status_question] * 2 + [leaf_question, hours_question, out_of_scope, other_account, human_request]
    if "fees" in c.w.chosen:
        pool.append(fee_question)
    if len(r.capabilities) < len(CAPABILITIES):
        pool.append(not_offered)
    requests = rng.choices(pool, k=rng.choice((1, 2, 2, 3, 3, 4)))
    # A transfer ends the conversation with this agent.
    if human_request in requests:
        requests = [item for item in requests if item is not human_request] + [human_request]
    if rng.random() < 0.08:
        requests.insert(0, wrong_details)
    elif rng.random() < 0.04:
        requests = [refuses_details]
    return requests


def choose_tools(world, rng):
    """Keys of the tools this policy offers: what it names, then a few it never asks for."""
    r = world.rules
    keys = {"email": "find_email", "name_zip": "find_person", "member": "verify_member"}
    chosen = {keys[r.identity], "list_mids", "get_leaf", "transfer"}
    needs = {"change": ["act"] + (["check"] if r.precheck else []), "refund": ["quote", "credit", "escalate"], "move": ["move"], "contact": ["update_contact"], "case": ["open_case"]}
    for key in r.capabilities:
        chosen.update(needs[key])
    if r.open_check:
        chosen.add(r.open_check)
    if r.note_after:
        chosen.add("note")
    if r.notify:
        chosen.add("notify")
    if rng.random() < 0.5:
        chosen.add("fees")
    spare = sorted(key for key in ("get_top", "get_mid", "note", "notify", "clock", "status", "fees", "check", "find_name") if key not in chosen)
    room = max(0, rng.randint(9, 14) - len(chosen))
    if "fees" not in chosen:
        spare.remove("fees")
    chosen.update(rng.sample(spare, min(room, len(spare))))
    return chosen


def build(rng):
    domain = rng.choice(DOMAINS)
    world = ServiceWorld(domain, rng)
    world.chosen = choose_tools(world, rng)
    conversation = Session(world, rng)
    top = rng.choice(world.tops)
    requests = choose_requests(conversation, rng)
    names = []
    for scenario in requests:
        scenario(conversation, top, rng.random() < 0.55)
        names.append(scenario.__name__)
    if requests[-1] not in (refuses_details, human_request) and rng.random() < 0.5:
        closing(conversation)
        names.append("closing")
    tools = [world.tools[key].doc() for key in world.chosen]
    rng.shuffle(tools)
    system = policy_text(world, rng)
    messages = [{"role": "system", "content": system}] + conversation.messages
    return domain, names, tools, messages, world.rules.saved({key: world.tools[key] for key in world.chosen})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conversations", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="data_work/processed/policy_v2.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    kinds, used, sizes, lengths = collections.Counter(), collections.Counter(), collections.Counter(), []
    with open(args.out, "w", encoding="utf-8") as f:
        for index in range(args.conversations):
            domain, names, tools, messages, rules = build(rng)
            problem = first_problem(tools, messages)
            if problem:
                raise SystemExit(f"conversation {index}: step {problem[0]} is invalid: {problem[1]} ({names})")
            kinds.update(step_kinds(messages))
            used.update(names)
            sizes[len(tools)] += 1
            lengths.append(len(messages[0]["content"]))
            record = {"id": f"policy_{args.seed}_{index}", "source": "policy", "family": f"policy_{domain.key}", "tools": tools, "messages": messages, "scenarios": names, "rules": rules}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    lengths.sort()
    print(f"{args.conversations} conversations, {sum(kinds.values())} assistant steps")
    for key, value in sorted(kinds.items()):
        print(f"  {key:20s} {value:7d}")
    print("  tools per conversation:", sorted(sizes.items()))
    print(f"  policy length: min {lengths[0]}, median {lengths[len(lengths) // 2]}, max {lengths[-1]}")
    print("  scenarios:", dict(used.most_common()))


if __name__ == "__main__":
    main()

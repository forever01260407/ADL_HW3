"""Generate tool-calling conversations whose tool results are really computed.

    python code/data/gen_chains.py --conversations 2000 --seed 1 --out data_work/processed/chains.jsonl

Each conversation gets a small in-memory world and tools that read and change it. A
scenario is ordinary code: it calls a tool, reads what came back and decides what to do
next, so every argument follows from an earlier message and every reply states facts
from the results. Tool names, parameter names, id formats, tool order and distractor
tools change per conversation, and the same result ends one task and continues another,
which is the decision the model has to learn. `family` is the domain, so a dev split can
hold out whole domains. Nothing here comes from the public test.
"""
import argparse
import collections
import json
import random
import string
from dataclasses import dataclass

from common import call_message, first_problem, make_tool, step_kinds, tool_message

FIRST = "Ava Noah Mia Liam Zoe Ethan Lena Omar Priya Kenji Sofia Mateo Hana Tariq Elena Jonas Amara Diego Yuki Ingrid Ravi Chloe Hugo Nadia".split()
LAST = "Chen Patel Garcia Kim Novak Silva Haddad Larsen Okafor Rossi Tanaka Moreau Singh Kowalski Reyes Fischer Nguyen Ali Berg Costa".split()
CITIES = "Lisbon Osaka Denver Nairobi Toronto Hamburg Seoul Lima Austin Oslo Perth Cairo Dublin Porto Quito Vienna".split()
CURRENCIES = {"EUR": 0.92, "GBP": 0.79, "JPY": 151.0, "CAD": 1.36, "AUD": 1.52}
WEATHER = ("sunny", "cloudy", "light rain", "windy", "snow showers", "foggy")
CARRIERS = ("DHL", "UPS", "FedEx", "GLS", "DPD")


def code(rng, length=8):
    return "".join(rng.choices(string.ascii_uppercase + string.digits, k=length))


def date(rng):
    return f"20{rng.randint(25, 27)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"


def phone(rng):
    return f"+1-555-{rng.randint(1000, 9999)}"


def person(rng):
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


@dataclass(frozen=True)
class Domain:
    key: str
    topic: str
    top: str
    extra: tuple  # (field, values) of the top entity
    mid: str
    label: tuple  # (field, values) of the middle entity
    amount: str
    statuses: tuple  # statuses[0] is the one the action applies to
    verb: str
    closed: str
    leaf: str
    leaf_fields: dict  # field -> value maker


def choice(values):
    return lambda rng: rng.choice(values)


DOMAINS = [
    Domain("store", "an online store", "customer", ("tier", ("bronze", "silver", "gold")), "order",
           ("item", ("desk lamp", "mechanical keyboard", "yoga mat", "air fryer", "backpack", "monitor stand")), "total",
           ("pending", "shipped", "delivered"), "cancel", "cancelled", "shipment",
           {"carrier": choice(CARRIERS), "tracking_code": code, "eta": date}),
    Domain("clinic", "a medical clinic", "patient", ("insurer", ("Aetna", "Cigna", "Allianz", "Bupa")), "appointment",
           ("department", ("cardiology", "dermatology", "neurology", "orthopedics", "pediatrics")), "fee",
           ("scheduled", "completed", "missed"), "cancel", "cancelled", "prescription",
           {"drug": choice(("amoxicillin", "ibuprofen", "cetirizine", "metformin")), "dosage": choice(("5 mg", "10 mg", "250 mg")), "pharmacy": choice(("Northside", "Lakeview", "Central"))}),
    Domain("library", "a public library", "member", ("card_type", ("student", "adult", "senior")), "loan",
           ("title", ("Dune", "Emma", "Beloved", "The Trial", "Middlemarch", "Kindred")), "late_fee",
           ("active", "returned", "lost"), "renew", "renewed", "copy",
           {"shelf": lambda rng: f"{rng.choice('ABCDEF')}-{rng.randint(1, 40)}", "barcode": code, "due_date": date}),
    Domain("airline", "an airline", "passenger", ("loyalty_level", ("none", "silver", "platinum")), "booking",
           ("route", ("LIS-JFK", "NRT-SFO", "DEN-ORD", "OSL-CDG", "SYD-AKL")), "fare",
           ("confirmed", "flown", "refunded"), "cancel", "cancelled", "boarding_pass",
           {"seat": lambda rng: f"{rng.randint(1, 38)}{rng.choice('ABCDEF')}", "gate": lambda rng: f"{rng.choice('ABCD')}{rng.randint(1, 30)}", "departure": date}),
    Domain("hotel", "a hotel chain", "guest", ("membership", ("basic", "plus", "elite")), "reservation",
           ("room_type", ("single", "double", "suite", "family room")), "rate",
           ("booked", "checked_out", "no_show"), "cancel", "cancelled", "invoice",
           {"invoice_number": code, "issued_on": date, "payment_method": choice(("card", "cash", "transfer"))}),
    Domain("university", "a university registrar", "student", ("program", ("biology", "economics", "physics", "history")), "enrollment",
           ("course", ("Linear Algebra", "Microeconomics", "Organic Chemistry", "World History")), "tuition",
           ("enrolled", "completed", "failed"), "withdraw", "withdrawn", "transcript_entry",
           {"grade": choice(("A", "B+", "B", "C")), "instructor": person, "semester": choice(("fall", "spring", "summer"))}),
    Domain("helpdesk", "an IT help desk", "employee", ("department", ("finance", "sales", "legal", "design")), "ticket",
           ("subject", ("VPN access", "laptop replacement", "password reset", "printer issue")), "estimated_cost",
           ("open", "resolved", "on_hold"), "close", "closed", "assignment",
           {"agent": person, "queue": choice(("tier1", "tier2", "network")), "sla_date": date}),
    Domain("bank", "a retail bank", "client", ("segment", ("personal", "business", "private")), "account",
           ("product", ("checking", "savings", "credit line", "money market")), "balance",
           ("active", "dormant", "frozen"), "freeze", "frozen", "card",
           {"last_four": lambda rng: f"{rng.randint(0, 9999):04d}", "network": choice(("Visa", "Mastercard")), "expiry": date}),
    Domain("insurance", "an insurance company", "policyholder", ("plan", ("essential", "standard", "premium")), "claim",
           ("incident", ("water damage", "windshield crack", "theft", "hail damage")), "amount",
           ("submitted", "paid", "denied"), "withdraw", "withdrawn", "adjuster_report",
           {"adjuster": person, "visit_date": date, "report_code": code}),
    Domain("food", "a food delivery app", "user", ("plan", ("free", "plus")), "delivery",
           ("restaurant", ("Thai Basil", "Luigi's", "Taco Loco", "Sushi Ten", "Green Bowl")), "bill",
           ("preparing", "delivered", "refunded"), "cancel", "cancelled", "courier",
           {"courier_name": person, "phone": phone, "vehicle": choice(("bike", "scooter", "car"))}),
    Domain("events", "an event ticketing site", "attendee", ("group", ("general", "press", "vip")), "ticket",
           ("event", ("Jazz Night", "Tech Summit", "Food Fair", "Indie Film Fest")), "price",
           ("valid", "used", "refunded"), "refund", "refunded", "seat",
           {"section": choice(("orchestra", "balcony", "floor")), "row": lambda rng: str(rng.randint(1, 30)), "entrance": lambda rng: f"Gate {rng.randint(1, 9)}"}),
    Domain("hr", "a human resources system", "staff_member", ("office", ("Berlin", "Tokyo", "Boston", "Madrid")), "leave_request",
           ("leave_type", ("annual", "parental", "sick", "study")), "payout",
           ("requested", "approved", "rejected"), "withdraw", "withdrawn", "approval",
           {"approver": person, "decided_on": date, "reference": code}),
    Domain("telecom", "a mobile carrier", "subscriber", ("contract", ("prepaid", "postpaid")), "line",
           ("plan_name", ("Basic 5G", "Unlimited", "Family Share", "Data Only")), "monthly_charge",
           ("active", "suspended", "ported_out"), "suspend", "suspended", "device",
           {"model": choice(("Pixel 8", "iPhone 15", "Galaxy S24")), "imei": lambda rng: str(rng.randint(10**14, 10**15 - 1)), "sim_type": choice(("eSIM", "nano"))}),
    Domain("cloud", "a cloud hosting provider", "project_owner", ("region", ("eu-west", "us-east", "ap-south")), "server",
           ("instance_type", ("small", "medium", "large", "gpu")), "hourly_cost",
           ("running", "stopped", "terminated"), "stop", "stopped", "volume",
           {"size_gb": lambda rng: str(rng.choice((50, 100, 250, 500))), "mount_path": lambda rng: f"/data{rng.randint(1, 9)}", "snapshot_date": date}),
    Domain("car_rental", "a car rental agency", "driver", ("license_class", ("B", "C", "D")), "rental",
           ("vehicle_model", ("Corolla", "Model 3", "Golf", "CR-V")), "daily_rate",
           ("reserved", "returned", "overdue"), "cancel", "cancelled", "vehicle",
           {"plate": code, "pickup_branch": choice(CITIES), "fuel_type": choice(("petrol", "hybrid", "electric"))}),
    Domain("gym", "a gym network", "athlete", ("home_club", ("Downtown", "Riverside", "Uptown")), "membership",
           ("plan", ("monthly", "annual", "off-peak", "family")), "fee",
           ("active", "expired", "paused"), "pause", "paused", "locker",
           {"locker_number": lambda rng: str(rng.randint(1, 400)), "floor": lambda rng: str(rng.randint(1, 4)), "combination": lambda rng: f"{rng.randint(0, 99)}-{rng.randint(0, 99)}"}),
    Domain("supply", "a procurement system", "supplier", ("rating", ("A", "B", "C")), "purchase_order",
           ("product", ("steel bolts", "copper wire", "packing foam", "pallet wrap")), "cost",
           ("placed", "received", "disputed"), "cancel", "cancelled", "delivery_note",
           {"warehouse": choice(CITIES), "received_by": person, "note_code": code}),
    Domain("realty", "a property management firm", "tenant", ("building", ("Elm Court", "Harbor View", "Oak Plaza")), "lease",
           ("unit", ("2A", "4C", "7F", "11B", "3D")), "rent",
           ("active", "ended", "pending_renewal"), "terminate", "terminated", "inspection",
           {"inspector": person, "scheduled_for": date, "report_code": code}),
    Domain("vet", "a veterinary practice", "pet_owner", ("reminder_channel", ("sms", "email", "phone")), "visit",
           ("reason", ("vaccination", "dental cleaning", "skin allergy", "checkup")), "charge",
           ("booked", "finished", "skipped"), "cancel", "cancelled", "treatment",
           {"medication": choice(("carprofen", "doxycycline", "apoquel")), "vet_name": person, "follow_up": date}),
    Domain("parking", "a city parking authority", "permit_holder", ("zone", ("north", "south", "harbor")), "citation",
           ("violation", ("expired meter", "no permit", "blocked hydrant")), "penalty",
           ("unpaid", "paid", "dismissed"), "appeal", "appealed", "officer_note",
           {"officer": person, "issued_on": date, "plate_seen": code}),
]


def spoken(name):
    return name.replace("_", " ")


def plural(noun):
    if noun.endswith("y") and noun[-2] not in "aeiou":
        return noun[:-1] + "ies"
    return noun + ("es" if noun.endswith(("s", "ch")) else "s")


def a(noun):
    return ("an " if noun[0] in "aeiou" else "a ") + spoken(noun)


class Style:
    """How one conversation's tools happen to be named."""

    def __init__(self, rng):
        self.camel = rng.random() < 0.2
        self.bare_id = rng.random() < 0.25
        self.int_ids = rng.random() < 0.35
        self.get = rng.choice(("get_{}", "fetch_{}", "get_{}_details", "retrieve_{}", "{}_info", "read_{}"))
        self.find = rng.choice(("find_{}", "lookup_{}", "search_{}", "{}_search", "locate_{}"))
        self.list = rng.choice(("list_{1}", "list_{0}_{1}", "get_{0}_{1}", "{0}_{1}", "fetch_{1}_for_{0}"))
        self.leaf = rng.choice(("get_{1}", "get_{0}_{1}", "fetch_{1}", "{1}_for_{0}", "lookup_{1}"))
        self.calculate = rng.choice(("calculate", "add_numbers", "sum_values"))
        self.convert = rng.choice(("convert_currency", "currency_convert", "fx_convert"))
        self.email = rng.choice(("send_email", "send_mail", "email_send"))
        self.recipient = rng.choice(("to", "recipient", "to_address"))
        self.weather = rng.choice(("get_weather", "weather_lookup", "current_weather"))
        self.wrap = rng.random() < 0.3

    def name(self, text):
        text = text.replace(" ", "_")
        if not self.camel:
            return text
        head, *rest = text.split("_")
        return head + "".join(part.capitalize() for part in rest)

    def id_field(self, entity):
        return "id" if self.bare_id else self.name(f"{entity}_id")


class Tool:
    def __init__(self, name, description, params, run):
        """params: (logical name, real name, JSON type, description, required)."""
        self.name, self.description, self.params, self.run = name, description, params, run

    def doc(self):
        properties = {real: {"type": kind, "description": text} for _, real, kind, text, _ in self.params}
        required = [real for _, real, _, _, needed in self.params if needed]
        return make_tool({"name": self.name, "description": self.description,
                          "parameters": {"type": "object", "properties": properties, "required": required}})


class World:
    def __init__(self, domain, rng):
        self.d, self.rng = domain, rng
        self.style = Style(rng)
        self.auth = rng.random() < 0.12
        self.token = code(rng, 16)
        self.user_name, self.password = f"agent{rng.randint(10, 99)}", code(rng, 6).lower()
        self.next_id = rng.randint(1000, 9000)
        self.sent, self.tops = [], []
        names = rng.sample(FIRST, 6)
        surname = rng.choice(LAST)
        for index, first in enumerate(names):
            # Two people share a surname so that a name search can be ambiguous.
            last = surname if index < 2 else rng.choice(LAST)
            self.tops.append(self.make_top(f"{first} {last}"))
        self.tools = self.make_tools()

    def new_id(self, entity):
        self.next_id += self.rng.randint(1, 37)
        if self.style.int_ids:
            return self.next_id
        return f"{entity[:3].upper()}-{self.next_id}"

    def make_top(self, name):
        rng, d = self.rng, self.d
        first, last = name.lower().split()
        top = {"id": self.new_id(d.top), "name": name, "email": f"{first}.{last}@example.{rng.choice(('com', 'org', 'net'))}",
               "city": rng.choice(CITIES), d.extra[0]: rng.choice(d.extra[1]), "mids": []}
        labels = rng.sample(d.label[1], min(rng.randint(1, 4), len(d.label[1])))
        for index, label in enumerate(labels):
            status = d.statuses[0] if index == 0 else rng.choice(d.statuses)
            leaf = {field: make(rng) for field, make in d.leaf_fields.items()}
            top["mids"].append({"id": self.new_id(d.mid), d.label[0]: label, d.amount: round(rng.uniform(5, 900), 2),
                                "status": status, "date": date(rng), "leaf": leaf})
        rng.shuffle(top["mids"])
        return top

    def show_top(self, top):
        return {self.style.id_field(self.d.top): top["id"], "name": top["name"], "email": top["email"], "city": top["city"], self.d.extra[0]: top[self.d.extra[0]]}

    def show_mid(self, mid):
        fields = {k: v for k, v in mid.items() if k not in ("id", "leaf")}
        return {self.style.id_field(self.d.mid): mid["id"], **fields}

    def by_id(self, items, wanted):
        return next((item for item in items if str(item["id"]) == str(wanted)), None)

    def mids(self):
        return [mid for top in self.tops for mid in top["mids"]]

    def make_tools(self):
        d, s = self.d, self.style
        id_kind = "integer" if s.int_ids else "string"
        top_id = ("top_id", s.id_field(d.top), id_kind, f"Identifier of the {spoken(d.top)}.", True)
        mid_id = ("mid_id", s.id_field(d.mid), id_kind, f"Identifier of the {spoken(d.mid)}.", True)
        token = [("token", "auth_token", "string", "Session token from the login tool.", True)] if self.auth else []

        def missing(entity):
            return {"error": f"No {spoken(entity)} with that identifier."}

        def find_email(email, **_):
            return {"results": [self.show_top(t) for t in self.tops if t["email"] == email]}

        def find_name(name, **_):
            return {"results": [self.show_top(t) for t in self.tops if name.lower() in t["name"].lower()]}

        def get_top(top_id, **_):
            top = self.by_id(self.tops, top_id)
            return self.show_top(top) if top else missing(d.top)

        def list_mids(top_id, status=None, **_):
            top = self.by_id(self.tops, top_id)
            if not top:
                return missing(d.top)
            return {plural(d.mid): [self.show_mid(m) for m in top["mids"] if status is None or m["status"] == status]}

        def get_mid(mid_id, **_):
            mid = self.by_id(self.mids(), mid_id)
            return self.show_mid(mid) if mid else missing(d.mid)

        def get_leaf(mid_id, **_):
            mid = self.by_id(self.mids(), mid_id)
            return mid["leaf"] if mid else missing(d.mid)

        def act(mid_id, reason=None, **_):
            mid = self.by_id(self.mids(), mid_id)
            if not mid:
                return missing(d.mid)
            if mid["status"] != d.statuses[0]:
                return {"error": f"Only a {spoken(d.statuses[0])} {spoken(d.mid)} can be changed. This one is {spoken(mid['status'])}."}
            mid["status"] = d.closed
            return {"success": True, s.id_field(d.mid): mid["id"], "status": d.closed}

        def update_city(top_id, city, **_):
            top = self.by_id(self.tops, top_id)
            if not top:
                return missing(d.top)
            top["city"] = city
            return {"success": True, s.id_field(d.top): top["id"]}

        def calculate(expression):
            terms = [float(part) for part in expression.split("+")]
            return {"result": round(sum(terms), 2)}

        def convert(amount, target):
            return {"amount": round(amount * CURRENCIES[target], 2), "currency": target}

        def send_email(to, subject, body):
            self.sent.append(to)
            return {"sent": True, "message_id": code(self.rng, 10)}

        def weather(city):
            return {"city": city, "conditions": self.rng.choice(WEATHER), "temperature_c": self.rng.randint(-5, 34)}

        def login(username, password):
            return {"token": self.token, "expires_in": 3600}

        mid_many = spoken(plural(d.mid))
        tools = {
            "find_email": Tool(s.name(s.find.format(f"{d.top}_by_email")), f"Find {a(d.top)} of {d.topic} by exact email address.",
                               [("email", "email", "string", "Email address.", True)] + token, find_email),
            "find_name": Tool(s.name(s.find.format(f"{d.top}_by_name")), f"Search {plural(spoken(d.top))} by full or partial name.",
                              [("name", "name", "string", "Full or partial name.", True)] + token, find_name),
            "get_top": Tool(s.name(s.get.format(d.top)), f"Get the profile of one {spoken(d.top)}.", [top_id] + token, get_top),
            "list_mids": Tool(s.name(s.list.format(d.top, plural(d.mid))), f"List the {mid_many} of {a(d.top)}.",
                              [top_id, ("status", "status", "string", f"Only return {mid_many} with this status.", False)] + token, list_mids),
            "get_mid": Tool(s.name(s.get.format(d.mid)), f"Get one {spoken(d.mid)} by identifier.", [mid_id] + token, get_mid),
            "get_leaf": Tool(s.name(s.leaf.format(d.mid, d.leaf)), f"Get the {spoken(d.leaf)} linked to {a(d.mid)}.", [mid_id] + token, get_leaf),
            "act": Tool(s.name(f"{d.verb}_{d.mid}"), f"{d.verb.capitalize()} {a(d.mid)}. Only works while it is {spoken(d.statuses[0])}.",
                        [mid_id, ("reason", "reason", "string", "Optional note.", False)] + token, act),
            "update_city": Tool(s.name(f"update_{d.top}_city"), f"Change the city on {a(d.top)}'s profile.",
                                [top_id, ("city", "city", "string", "New city.", True)] + token, update_city),
            "calculate": Tool(s.calculate, "Add numbers. The expression is numbers joined with +.",
                              [("expression", "expression", "string", "For example 12.5+3+4.25.", True)], calculate),
            "convert": Tool(s.convert, "Convert a USD amount to another currency.",
                            [("amount", "amount", "number", "Amount in USD.", True), ("target", "target_currency", "string", "Three-letter code.", True)], convert),
            "send_email": Tool(s.email, "Send an email.", [("to", s.recipient, "string", "Recipient address.", True), ("subject", "subject", "string", "Subject line.", True),
                                                                ("body", "body", "string", "Message body.", True)], send_email),
            "weather": Tool(s.weather, "Current weather in a city.", [("city", "city", "string", "City name.", True)], weather),
        }
        if self.auth:
            tools["login"] = Tool(s.name("login"), "Start a session and get a token.",
                                  [("username", "username", "string", "User name.", True), ("password", "password", "string", "Password.", True)], login)
        return tools


class Conversation:
    def __init__(self, world, rng):
        self.w, self.d, self.rng = world, world.d, rng
        self.messages = []
        self.has_token = False
        self.opened = False

    def say(self, options, **slots):
        slots = {"top": spoken(self.d.top), "mid": spoken(self.d.mid), "a_mid": a(self.d.mid), "mids": spoken(plural(self.d.mid)), "leaf": spoken(self.d.leaf), "verb": self.d.verb, **slots}
        template = self.rng.choice(options)
        text = template.format(**slots)
        return text[0].upper() + text[1:] if template.startswith("{verb}") else text

    def user(self, text):
        if self.w.auth and not self.opened:
            text += f" My login is {self.w.user_name}, password {self.w.password}."
        self.opened = True
        self.messages.append({"role": "user", "content": text})

    def reply(self, text):
        self.messages.append({"role": "assistant", "content": text})

    def shaped(self, result):
        return {"status": "ok", "data": result} if self.w.style.wrap and "error" not in result else result

    def _invoke(self, key, arguments):
        tool = self.w.tools[key]
        real = {}
        for logical, name, *_ in tool.params:
            if logical in arguments and arguments[logical] is not None:
                real[name] = arguments[logical]
        return tool, real, tool.run(**arguments)

    def _login(self):
        if self.w.auth and not self.has_token:
            self.has_token = True
            self.call("login", username=self.w.user_name, password=self.w.password)

    def call(self, key, **arguments):
        if key != "login":
            self._login()
            if self.w.auth and any(p[0] == "token" for p in self.w.tools[key].params):
                arguments["token"] = self.w.token
        tool, real, result = self._invoke(key, arguments)
        self.messages.append(call_message([(tool.name, real)]))
        self.messages.append(tool_message(tool.name, self.shaped(result)))
        return result

    def together(self, key, many):
        """Several independent calls in one assistant message."""
        self._login()
        done = []
        for arguments in many:
            if self.w.auth and any(p[0] == "token" for p in self.w.tools[key].params):
                arguments = {**arguments, "token": self.w.token}
            done.append(self._invoke(key, arguments))
        self.messages.append(call_message([(tool.name, real) for tool, real, _ in done]))
        self.messages.extend(tool_message(tool.name, self.shaped(result)) for tool, _, result in done)
        return [result for _, _, result in done]

    # ---- steps shared by the scenarios
    def pick_top(self, need=None):
        tops = [t for t in self.w.tops if need is None or need(t)]
        return self.rng.choice(tops or self.w.tops)

    def find(self, top):
        return self.call("find_email", email=top["email"])

    def list_mids(self, top, status=None):
        return self.call("list_mids", top_id=top["id"], status=status)

    def open_mid(self, top):
        return next((m for m in top["mids"] if m["status"] == self.d.statuses[0]), None)


def has_open(world):
    """Exactly one open item, or "the pending order" would not say which one."""
    return lambda top: sum(m["status"] == world.d.statuses[0] for m in top["mids"]) == 1


def label_of(world, mid):
    return mid[world.d.label[0]]


def money(value):
    return f"${value:,.2f}"


# ---- scenarios: each one is a task for a single user turn or a short exchange
def profile_answer(c):
    """One lookup is enough."""
    top = c.pick_top()
    field = c.rng.choice(("city", c.d.extra[0]))
    c.user(c.say(("What {field} do we have on file for {email}?", "Look up {email} and tell me their {field}.",
                  "Which {field} is listed for the {top} with the email {email}?", "I need the {field} of {name}. Their email is {email}."),
                 field=spoken(field), email=top["email"], name=top["name"]))
    c.find(top)
    c.reply(c.say(("{name}'s {field} is {value}.", "The {field} on file for {name} is {value}."), name=top["name"], field=spoken(field), value=top[field]))


def count_mids(c):
    """Two hops, then stop."""
    top = c.pick_top()
    c.user(c.say(("How many {mids} does {email} have?", "Can you list the {mids} for {name} ({email})?", "What {mids} are on the account of {email}?"),
                 email=top["email"], name=top["name"]))
    c.find(top)
    c.list_mids(top)
    items = ", ".join(f"{label_of(c.w, m)} ({spoken(m['status'])})" for m in top["mids"])
    noun = c.say(["{mid}"] if len(top["mids"]) == 1 else ["{mids}"])
    c.reply(f"{top['name']} has {len(top['mids'])} {noun}: {items}.")


def leaf_detail(c):
    """Three hops: person, their item, the record behind it."""
    top = c.pick_top()
    mid = c.rng.choice(top["mids"])
    field = c.rng.choice(list(c.d.leaf_fields))
    c.user(c.say(("What is the {field} for the {label} {mid} of {email}?", "{name} ({email}) asked about the {field} on their {label} {mid}. Can you check?",
                  "Find the {field} of {name}'s {label} {mid}. Email on file: {email}."),
                 field=spoken(field), label=label_of(c.w, mid), email=top["email"], name=top["name"]))
    c.find(top)
    c.list_mids(top)
    c.call("get_leaf", mid_id=mid["id"])
    c.reply(c.say(("The {field} is {value}.", "{name}'s {label} {mid} shows {field} {value}."), field=spoken(field), value=mid["leaf"][field], name=top["name"], label=label_of(c.w, mid)))
    return top, mid


def act_on_open(c):
    """Find the item, then change it."""
    top = c.pick_top(has_open(c.w))
    mid = c.open_mid(top)
    c.user(c.say(("Please {verb} the {state} {mid} of {email}.", "{name} wants to {verb} their {state} {mid}. Email: {email}.",
                  "{verb} the {mid} that is still {state} for {email}, thanks."),
                 state=spoken(c.d.statuses[0]), email=top["email"], name=top["name"]))
    c.find(top)
    c.list_mids(top, status=c.d.statuses[0] if c.rng.random() < 0.5 else None)
    c.call("act", mid_id=mid["id"])
    c.reply(c.say(("Done. The {label} {mid} is now {closed}.", "{name}'s {label} {mid} has been marked {closed}."), label=label_of(c.w, mid), closed=spoken(c.d.closed), name=top["name"]))


def conditional(c):
    """Whether to act depends on what the lookup returned."""
    top = c.pick_top()
    mid = c.rng.choice(top["mids"])
    state = spoken(c.d.statuses[0])
    c.user(c.say(("If the {label} {mid} of {email} is still {state}, {verb} it. Otherwise just tell me its status.",
                  "Check the {label} {mid} for {email}. {verb} it only if it is {state}.",
                  "{name} ({email}): if their {label} {mid} shows {state}, please {verb} it, if not let me know where it stands."),
                 label=label_of(c.w, mid), email=top["email"], state=state, name=top["name"]))
    c.find(top)
    c.list_mids(top)
    if mid["status"] == c.d.statuses[0]:
        c.call("act", mid_id=mid["id"])
        c.reply(c.say(("It was {state}, so it is now {closed}.", "The {mid} was still {state}. I changed it to {closed}."), state=state, closed=spoken(c.d.closed)))
    else:
        c.reply(c.say(("No change made. The {mid} is {status}.", "It is {status}, so I left it alone."), status=spoken(mid["status"])))


def pick_extreme(c):
    """Choose from a list by a value, then use that item's id."""
    top = c.pick_top(lambda t: len(t["mids"]) > 1)
    high = c.rng.random() < 0.5
    chosen = (max if high else min)(top["mids"], key=lambda m: m[c.d.amount])
    field = c.rng.choice(list(c.d.leaf_fields))
    c.user(c.say(("Among the {mids} of {email}, which has the {which} {amount}? I need its {field}.", "Get the {field} for the {mid} with the {which} {amount} on {email}'s account."),
                 which="highest" if high else "lowest", amount=spoken(c.d.amount), field=spoken(field), email=top["email"]))
    c.find(top)
    c.list_mids(top)
    c.call("get_leaf", mid_id=chosen["id"])
    c.reply(c.say(("That is the {label} {mid} at {price}. Its {field} is {value}.",), label=label_of(c.w, chosen), price=money(chosen[c.d.amount]), field=spoken(field), value=chosen["leaf"][field]))


def total_converted(c):
    """Use numbers from a result in later calculations."""
    top = c.pick_top(lambda t: len(t["mids"]) > 1)
    target = c.rng.choice(list(CURRENCIES))
    c.user(c.say(("Add up the {amount} across all {mids} of {email} and give me the total in {target}.", "What is the combined {amount} for {name} ({email}) in {target}?"),
                 amount=spoken(c.d.amount), email=top["email"], name=top["name"], target=target))
    c.find(top)
    c.list_mids(top)
    total = c.call("calculate", expression="+".join(str(m[c.d.amount]) for m in top["mids"]))["result"]
    converted = c.call("convert", amount=total, target=target)
    c.reply(f"The total is {money(total)}, which is {converted['amount']} {target}.")


def notify(c):
    """Look something up, then tell someone about it."""
    top = c.pick_top()
    mid = c.rng.choice(top["mids"])
    c.user(c.say(("Email {name} ({email}) the current status of their {label} {mid}.", "Let {email} know by email where their {label} {mid} stands."),
                 name=top["name"], email=top["email"], label=label_of(c.w, mid)))
    c.find(top)
    c.list_mids(top)
    text = f"Your {label_of(c.w, mid)} {spoken(c.d.mid)} is {spoken(mid['status'])}."
    c.call("send_email", to=top["email"], subject=f"Update on your {spoken(c.d.mid)}", body=text)
    c.reply(c.say(("I emailed {name} the update.", "The status email to {name} has been sent."), name=top["name"]))


def weather_at_home(c):
    """A value from one tool feeds a different kind of tool."""
    top = c.pick_top()
    c.user(c.say(("What's the weather like where {email} lives?", "Check today's weather in the city on file for {name} ({email})."), email=top["email"], name=top["name"]))
    c.find(top)
    result = c.call("weather", city=top["city"])
    c.reply(f"It's {result['conditions']} in {top['city']}, {result['temperature_c']}°C.")


def not_found(c):
    """An empty result ends the task."""
    wrong = f"{c.rng.choice(FIRST).lower()}.{code(c.rng, 4).lower()}@example.com"
    c.user(c.say(("Pull up the {mids} for {email}.", "{verb} the latest {mid} for {email}."), email=wrong))
    c.call("find_email", email=wrong)
    c.reply(c.say(("I couldn't find a {top} with the email {email}. Could you double-check it?", "There is no {top} registered under {email}, so I stopped there."), email=wrong))


def not_found_then_name(c):
    """An empty result, then another way to find the person."""
    top = c.pick_top()
    wrong = f"{code(c.rng, 5).lower()}@example.com"
    c.user(c.say(("Find {name}'s {extra}. I think the email is {email}, but use the name if that fails.",), name=top["name"], email=wrong, extra=spoken(c.d.extra[0])))
    c.call("find_email", email=wrong)
    result = c.call("find_name", name=top["name"])
    if len(result["results"]) == 1:
        c.reply(f"{top['name']}'s {spoken(c.d.extra[0])} is {top[c.d.extra[0]]}.")
    else:
        c.reply(f"Several people match {top['name']}. Which city are they in?")


def ambiguous_name(c):
    """Two matches: ask, then continue with the answer."""
    first, second = c.w.tops[0], c.w.tops[1]
    surname = first["name"].split()[1]
    c.user(c.say(("What {mids} does {surname} have?", "Show me the {mids} for the {top} named {surname}."), surname=surname))
    result = c.call("find_name", name=surname)
    names = [hit["name"] for hit in result["results"]]
    if len(names) < 2:
        c.reply(f"{surname} has an account, but I need a little more detail to pick the right one.")
        return
    c.reply(f"I found {len(names)} matches: {', '.join(names)}. Which one do you mean?")
    chosen = c.rng.choice((first, second))
    c.user(c.say(("{name}, please.", "The one called {name}.", "I mean {name}."), name=chosen["name"]))
    c.list_mids(chosen)
    items = ", ".join(label_of(c.w, m) for m in chosen["mids"])
    c.reply(f"{chosen['name']} has: {items}.")


def ask_first(c):
    """A required detail is missing, so ask before calling anything."""
    top = c.pick_top(has_open(c.w))
    mid = c.open_mid(top)
    c.user(c.say(("I'd like to {verb} {a_mid}.", "Can you {verb} my {mid}?", "Please {verb} the {mid} we talked about."),))
    c.reply(c.say(("Sure. What email is the account under?", "Happy to help. Which email address is it registered with?", "I need the email on the account first. What is it?")))
    c.user(c.say(("It's {email}.", "{email}", "The email is {email}. It's the {label} one."), email=top["email"], label=label_of(c.w, mid)))
    c.find(top)
    c.list_mids(top)
    c.call("act", mid_id=mid["id"])
    c.reply(c.say(("Done, your {label} {mid} is now {closed}.",), label=label_of(c.w, mid), closed=spoken(c.d.closed)))


def follow_up(c):
    """The next request reuses an id from earlier in the conversation."""
    top, mid = leaf_detail(c)
    c.user(c.say(("And what's its current status?", "Thanks. What status is that {mid} in?")))
    c.call("get_mid", mid_id=mid["id"])
    c.reply(c.say(("It is {status}.", "That {mid} is {status}."), status=spoken(mid["status"])))


def each_item(c):
    """The same call once per item."""
    top = c.pick_top(lambda t: len(t["mids"]) > 1)
    field = c.rng.choice(list(c.d.leaf_fields))
    c.user(c.say(("Give me the {field} for every {mid} of {email}.", "I need the {field} of all {mids} under {email}."), field=spoken(field), email=top["email"]))
    c.find(top)
    c.list_mids(top)
    if c.rng.random() < 0.5:
        c.together("get_leaf", [{"mid_id": m["id"]} for m in top["mids"]])
    else:
        for m in top["mids"]:
            c.call("get_leaf", mid_id=m["id"])
    items = ", ".join(f"{label_of(c.w, m)}: {m['leaf'][field]}" for m in top["mids"])
    c.reply(f"Here they are. {items}.")


def update_and_verify(c):
    """Change something, then read it back."""
    top = c.pick_top()
    city = c.rng.choice([city for city in CITIES if city != top["city"]])
    c.user(c.say(("{name} ({email}) moved to {city}. Update their profile and confirm it saved.", "Change the city for {email} to {city}, then check that it took."),
                 name=top["name"], email=top["email"], city=city))
    c.find(top)
    c.call("update_city", top_id=top["id"], city=city)
    c.call("get_top", top_id=top["id"])
    c.reply(f"Updated. {top['name']}'s profile now shows {city}.")


def refused_action(c):
    """The tool refuses; report it instead of trying again."""
    top = c.pick_top(lambda t: any(m["status"] != c.d.statuses[0] for m in t["mids"]))
    mid = next((m for m in top["mids"] if m["status"] != c.d.statuses[0]), top["mids"][0])
    c.user(c.say(("{verb} {mid} {id}, please.", "Go ahead and {verb} {mid} number {id}."), id=mid["id"]))
    was_open = mid["status"] == c.d.statuses[0]
    c.call("act", mid_id=mid["id"])
    if was_open:
        c.reply(c.say(("Done, it is now {closed}.",), closed=spoken(c.d.closed)))
    else:
        c.reply(c.say(("I couldn't do that. It is already {status}.", "That didn't go through because it is {status}."), status=spoken(mid["status"])))


def direct_id(c):
    """The user already has the id, so no search is needed."""
    top = c.pick_top()
    mid = c.rng.choice(top["mids"])
    field = c.rng.choice(list(c.d.leaf_fields))
    c.user(c.say(("Which {field} goes with {mid} {id}?", "I have {mid} {id} here. What's its {field}?"), field=spoken(field), id=mid["id"]))
    c.call("get_leaf", mid_id=mid["id"])
    c.reply(f"Its {spoken(field)} is {mid['leaf'][field]}.")


def two_cities(c):
    """Independent calls requested in one message."""
    first, second = c.rng.sample(CITIES, 2)
    c.user(c.say(("How's the weather in {first} and {second}?", "Compare today's weather in {first} and {second} for me."), first=first, second=second))
    if c.rng.random() < 0.5:
        results = c.together("weather", [{"city": first}, {"city": second}])
    else:
        results = [c.call("weather", city=first), c.call("weather", city=second)]
    c.reply(f"{first}: {results[0]['conditions']}, {results[0]['temperature_c']}°C. {second}: {results[1]['conditions']}, {results[1]['temperature_c']}°C.")


def no_tool_needed(c):
    c.user(c.say(("What can you help me with?", "Thanks for the help!", "Is this the right place to ask about my {mid}?", "Good morning!")))
    c.reply(c.say(("I can look up {mids} and profiles and make simple changes. What do you need?", "Happy to help with anything about your {mid}. What would you like to do?")))


SCENARIOS = {
    profile_answer: 3, count_mids: 3, leaf_detail: 4, act_on_open: 4, conditional: 5, pick_extreme: 4, total_converted: 3, notify: 3, weather_at_home: 2,
    not_found: 2, not_found_then_name: 2, ambiguous_name: 2, ask_first: 3, follow_up: 3, each_item: 3, update_and_verify: 3, refused_action: 2, direct_id: 2, two_cities: 1, no_tool_needed: 1,
}
SYSTEM = (
    "You are the assistant for {topic}. Use the tools when you need data.",
    "You help staff at {topic}. Answer briefly and only state what the tools return.",
    "You are a helpful assistant with access to the tools of {topic}.",
    "Assist with requests about {topic}. Ask for anything you need before acting.",
)


def build(rng):
    domain = rng.choice(DOMAINS)
    world = World(domain, rng)
    conversation = Conversation(world, rng)
    scenarios = rng.choices(list(SCENARIOS), weights=list(SCENARIOS.values()), k=rng.choice((1, 1, 2, 2, 3)))
    for scenario in scenarios:
        scenario(conversation)
    used = {call["function"]["name"] for m in conversation.messages for call in m.get("tool_calls") or []}
    keep = rng.choice((0.3, 0.6, 1.0))
    tools = [tool.doc() for tool in world.tools.values() if tool.name in used or rng.random() < keep]
    for other in rng.sample(DOMAINS, rng.choice((0, 1, 1, 2, 2, 3, 4, 5))):
        if other.key != domain.key:
            strangers = World(other, rng).tools
            keys = rng.sample(sorted(strangers), rng.randint(2, 8))
            taken = {tool["function"]["name"] for tool in tools}
            tools += [strangers[key].doc() for key in keys if strangers[key].name not in taken]
    rng.shuffle(tools)
    messages = conversation.messages
    if rng.random() < 0.5:
        messages = [{"role": "system", "content": rng.choice(SYSTEM).format(topic=domain.topic)}] + messages
    return domain, [s.__name__ for s in scenarios], tools, messages


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conversations", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="data_work/processed/chains.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    kinds, used, sizes = collections.Counter(), collections.Counter(), collections.Counter()
    with open(args.out, "w", encoding="utf-8") as f:
        for index in range(args.conversations):
            domain, names, tools, messages = build(rng)
            problem = first_problem(tools, messages)
            if problem:
                raise SystemExit(f"conversation {index}: step {problem[0]} is invalid: {problem[1]} ({names})")
            kinds.update(step_kinds(messages))
            used.update(names)
            sizes[len(tools) // 10 * 10] += 1
            record = {"id": f"chains_{args.seed}_{index}", "source": "chains", "family": f"chains_{domain.key}", "tools": tools, "messages": messages, "scenarios": names}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"{args.conversations} conversations, {sum(kinds.values())} assistant steps")
    for key, value in sorted(kinds.items()):
        print(f"  {key:20s} {value:7d}")
    print("  tools per conversation:", sorted(sizes.items()))


if __name__ == "__main__":
    main()

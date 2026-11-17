"""Whole-negotiation buyer evaluation sets with scripted sellers, for E24 (runbook STEP 05).

    python code/data/gen_buyer_eval.py --split select --seed 24001 --per-flow 34 --out FILE
    python code/data/gen_buyer_eval.py --split confirm --seed 24002 --per-flow 34 --out FILE

Independent of gen_buyer.py (the training generator): its own catalog (categories that
generator does not use), personas, phrasing and, above all, seller scripts. Each negotiation
follows one named flow; the flows of the two splits do not overlap. Only the fixed parts of the
prompt (rule text, the four tools: data_work/processed/buyer_interface.json) and the message
layout the model is tested on are shared, since they are the interface.

The buyer turns are a fixed reference history (teacher-forced evaluation: the model's replies
never change what the seller says next). The reference order, when the flow has one, gives the
negotiation's reference price; flows without a legal deal end with end_session.

select flows:  stepwise, final_first, repeated, offer_late, over_budget, wrong_terms
confirm flows: price_up_latest, few_turns, below_target_first
"""
import argparse
import json
import random

INTERFACE = "data_work/processed/buyer_interface.json"
FLOWS = {"select": ("stepwise", "final_first", "repeated", "offer_late", "over_budget", "wrong_terms"),
         "confirm": ("price_up_latest", "few_turns", "below_target_first")}

CATALOG = {
    "Photography": [("camera tripod", 30, 140, ("aluminum legs", "ball head", "1.6 m max")), ("lens cleaning kit", 9, 35, ("air blower", "microfiber cloths", "brush pen")),
                    ("camera bag", 25, 110, ("padded dividers", "rain cover", "sling strap")), ("ring light", 20, 85, ("10 inch", "three color modes", "phone holder"))],
    "Cycling": [("bike helmet", 30, 120, ("MIPS liner", "rear light", "size M")), ("bike lock", 18, 70, ("hardened shackle", "two keys", "frame mount")),
                ("floor pump", 20, 65, ("gauge on top", "Presta and Schrader", "steel barrel")), ("saddle bag", 12, 40, ("1 L", "reflective strip", "velcro straps"))],
    "Crafts": [("sewing machine", 90, 320, ("60 stitches", "auto threader", "LED lamp")), ("cutting mat", 12, 45, ("A2 size", "self healing", "grid lines")),
               ("glue gun", 10, 40, ("dual temperature", "20 sticks", "stand")), ("embroidery kit", 15, 55, ("hoop", "floss set", "pattern sheets"))],
    "Music": [("ukulele", 35, 150, ("concert size", "mahogany top", "gig bag")), ("guitar tuner", 8, 30, ("clip-on", "chromatic", "color screen")),
              ("keyboard stand", 25, 80, ("X frame", "height adjustable", "foam pads")), ("cajon drum", 50, 190, ("birch body", "snare wires", "padded seat"))],
    "Aquarium": [("aquarium filter", 22, 90, ("canister", "300 L per hour", "media trays")), ("aquarium heater", 15, 55, ("100 W", "auto shut-off", "glass tube")),
                 ("tank light", 25, 100, ("full spectrum", "timer", "extendable bracket")), ("water test kit", 14, 50, ("eight tests", "color chart", "dropper bottles"))],
    "Gaming": [("game controller", 35, 90, ("wireless", "rechargeable", "rear paddles")), ("gaming headset stand", 15, 45, ("RGB base", "two USB ports", "aluminum")),
               ("capture card", 80, 220, ("4K pass-through", "USB 3", "low latency")), ("desk mouse pad", 10, 40, ("XL", "stitched edge", "rubber base"))],
}
GRADES = ("Starter", "Studio", "Field", "Mk II", "Essential", "Deluxe", "Open-box", "Travel")
SPEC = ("{grade} {noun} - {attrs}. Listed at ${list}.", "{grade} {noun}; {attrs}; listed ${list}")
PEOPLE = ("Ines", "Bram", "Sofia", "Kofi", "Mirela", "Tariq", "Yuna", "Lars", "Chidi", "Paula", "Arjun", "Greta")
ROLES = ("night-shift pharmacist", "music teacher", "warehouse planner", "new parent", "landscape gardener", "junior architect", "retired engineer", "barista")
WHY = ("Yours broke last week and you need a replacement.", "You are picking up a new hobby this month.", "You promised one to a cousin.")
CARE = ("You care more about not overpaying than about speed.", "You would walk away rather than stretch your money.", "You like to keep things friendly but firm.")

OPEN = ("Hello! I'm interested in the {noun}. What price can you offer?", "Hi there, I'd like to buy the {noun}. What's your price?",
        "Good day. Could you send me an offer for the {noun}?")
COUNTER = ("I was thinking more like ${p}.", "Would ${p} be possible?", "My offer is ${p}.", "Could we meet at ${p}?")
ASK_OFFER = ("Could you put that in a formal offer?", "Can you send that as an offer I can accept?", "Please send a proper offer when you're ready.")
ASK_RIGHT = ("That's not the item I asked about. I need the {noun} ({pid}), one unit.", "I'm only after one {noun} ({pid}). Could you offer that instead?")
TAKE = ("Okay, I'll take it.", "That works, I'll buy it.", "Deal.", "")
WALK = ("Sorry, that's beyond what I can spend. I'll leave it here.", "I can't make that work. Thank you for your time.")

S_OFFER = ("Here's my price.", "This is what I can do.", "Happy to make you an offer.", "")
S_MOVE = ("I can shave a bit off.", "Let me lower it a little.", "Fine, a bit lower then.", "")
S_FINAL = ("That's my last offer on it.", "I can't go any lower than that.", "That price is final for me.")
S_SAME = ("Same price as before, I'm afraid.", "I'm staying at that price.", "Still the same offer.")
S_CHAT = ("It's in great shape, barely used.", "It comes with everything in the original box.", "Lots of people have asked about this one.",
          "Let me check the details and get back to you.")
S_TEXT_PRICE = ("I'd probably want somewhere near ${p} for it.", "Thinking roughly ${p}, but let me confirm.")
S_WRONG_PID = ("I could do this one instead.", "How about this similar model?")
S_WRONG_QTY = ("I'd sell a pair of them.", "These go as a set of {q}.")
S_UP = ("Sorry, I mispriced it before; here's the corrected price.", "My costs went up, so I need to raise it. That price is final.")


def money(cents):
    return f"{cents // 100}.{cents % 100:02d}"


def price_near(rng, cents):
    cents = max(100, cents)
    return cents - cents % 100 + rng.choice((0, 50, 99, 95, 49))


class Negotiation:
    def __init__(self, rng, interface, flow, number):
        self.rng, self.flow = rng, flow
        category = rng.choice(sorted(CATALOG))
        items = rng.sample([(p, g) for p in CATALOG[category] for g in GRADES], 6)
        self.catalog, ids = [], set()
        for (noun, low, high, attrs), grade in items:
            pid = "B0" + "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789") for _ in range(8))
            while pid in ids:
                pid = "B0" + "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789") for _ in range(8))
            ids.add(pid)
            listed = price_near(rng, rng.randrange(low * 100, high * 100))
            self.catalog.append({"category": category, "product_id": pid, "noun": noun, "list": listed,
                                 "spec": rng.choice(SPEC).format(grade=grade, noun=noun, attrs=", ".join(rng.sample(attrs, 2)), list=money(listed))})
        self.item = rng.choice(self.catalog)
        self.pid, self.noun, self.listed = self.item["product_id"], self.item["noun"], self.item["list"]
        self.budget = round(self.listed * rng.uniform(0.85, 1.15))
        self.target = round(self.budget * rng.uniform(0.72, 0.9))
        self.seller = f"vendor_{rng.randrange(2, 40)}"
        self.turns = 4 if flow == "few_turns" else rng.choice((14, 12, 10))
        self.next_id = rng.randrange(200, 80000)
        situation = {"acceptable_product_ids": [self.pid], "agent_id": f"ta_buyer_{rng.randrange(30, 60)}",
                     "catalog": [{k: c[k] for k in ("category", "product_id", "spec")} for c in self.catalog],
                     "private_profile": {"budget": self.budget / 100, "pain_point": rng.choice(WHY), "target_price": self.target / 100},
                     "public_brief": f"Buying a {self.noun} (1 unit(s)).", "seller_id": self.seller}
        persona = [f"You are {rng.choice(PEOPLE)}, a {rng.choice(ROLES)}, looking for a {self.noun}.", situation["private_profile"]["pain_point"], rng.choice(CARE)]
        system = "\n".join([interface["system_head"], *persona, interface["system_tail"], json.dumps(situation, ensure_ascii=False, indent=1, sort_keys=True), ""])
        self.tools = interface["tools"]
        self.messages = [{"role": "system", "content": system}, {"role": "user", "content": self.state()}]
        self.reference_price = None
        self.offer_id = None
        self.counter = None

    def state(self):
        return json.dumps({"phase": "negotiation", "turns_remaining_in_phase": self.turns})

    def mid(self):
        self.next_id += self.rng.randrange(1, 5)
        return f"m_{self.next_id}"

    def seller_says(self, *blocks):
        """blocks: (text, offer) with offer (pid, cents, qty) or None."""
        parts = []
        for text, offer in blocks:
            message_id = self.mid()
            lines = [json.dumps({"from_id": self.seller, "message_id": message_id})]
            if offer:
                lines.append(f"Offer: {offer[0]} at ${money(offer[1])} per unit, qty {offer[2]}." + (f" {text}" if text else ""))
                if offer[0] == self.pid and offer[2] == 1:
                    self.offer_id = message_id
            else:
                lines.append(text)
            parts.append("\n".join(lines))
        parts.append(self.state())
        self.messages.append({"role": "user", "content": "\n\n".join(parts)})

    def buyer(self, text, name="wait", arguments=None):
        message = {"role": "assistant", "content": text}
        message["tool_calls"] = [{"type": "function", "function": {"name": name, "arguments": arguments or {}}}]
        self.messages.append(message)
        self.turns -= 1
        if name == "wait":
            data = {"phase": "negotiation", "action": "wait"}
            if text:
                data["reply_message_id"] = self.mid()
            self.messages.append({"role": "tool", "name": "wait", "content": json.dumps({"ok": True, "data": data})})

    def counter_below(self, offer_cents):
        top = min(self.target, offer_cents) - 100
        value = round(self.target * self.rng.uniform(0.75, 0.9)) if self.counter is None else self.counter + max(0, (top - self.counter) // 2)
        self.counter = max(100, min(value, top))
        return self.rng.choice(COUNTER).format(p=money(self.counter))

    def order(self, cents):
        self.reference_price = cents
        self.buyer(self.rng.choice(TAKE), "order", {"product_id": self.pid, "agree_price": cents / 100, "qty": 1, "evidence_message_ids": [self.offer_id]})

    def walk(self):
        self.buyer(self.rng.choice(WALK), "end_session", {"reason": "price above budget"})

    def run(self):
        rng, f = self.rng, self.flow
        self.buyer(rng.choice(OPEN).format(noun=self.noun))
        legal_high = lambda: rng.randrange(self.target + 100, max(self.target + 200, self.budget - 50))  # noqa: E731
        if f == "stepwise":
            price = max(self.budget + rng.randrange(200, 1500), legal_high())
            floor = rng.randrange(self.target - 300, self.target + 200) if rng.random() < 0.5 else legal_high()
            floor = min(floor, self.budget - 50)
            for _ in range(rng.choice((2, 3, 4))):
                self.seller_says((rng.choice(S_MOVE if self.counter else S_OFFER), (self.pid, price, 1)))
                if price <= self.target:
                    return self.order(price)
                self.buyer(self.counter_below(price))
                price = max(floor, price - round(self.listed * rng.uniform(0.04, 0.1)))
            self.seller_says((rng.choice(S_FINAL), (self.pid, price, 1)))
            return self.order(price) if price <= self.budget else self.walk()
        if f == "final_first":
            price = legal_high()
            self.seller_says((rng.choice(S_FINAL), (self.pid, price, 1)))
            return self.order(price)
        if f == "repeated":
            price = legal_high()
            self.seller_says((rng.choice(S_OFFER), (self.pid, price, 1)))
            self.buyer(self.counter_below(price))
            self.seller_says((rng.choice(S_SAME), (self.pid, price, 1)))
            if rng.random() < 0.5:
                self.buyer(self.counter_below(price))
                self.seller_says((rng.choice(S_SAME), (self.pid, price, 1)))
            return self.order(price)
        if f == "offer_late":
            for _ in range(rng.choice((2, 3))):
                if rng.random() < 0.5:
                    self.seller_says((rng.choice(S_TEXT_PRICE).format(p=money(rng.randrange(self.target, self.budget + 500))), None))
                    self.buyer(rng.choice(ASK_OFFER))
                else:
                    self.seller_says((rng.choice(S_CHAT), None))
                    self.buyer(rng.choice(ASK_OFFER))
            price = rng.randrange(round(self.target * 0.85), self.target)
            self.seller_says((rng.choice(S_OFFER), (self.pid, price, 1)))
            return self.order(price)
        if f == "over_budget":
            price = self.budget + rng.randrange(800, 3000)
            for _ in range(rng.choice((2, 3))):
                self.seller_says((rng.choice(S_MOVE if self.counter else S_OFFER), (self.pid, price, 1)))
                self.buyer(self.counter_below(price))
                price = max(self.budget + rng.randrange(100, 400), price - rng.randrange(200, 600))
            self.seller_says((rng.choice(S_FINAL), (self.pid, price, 1)))
            return self.walk()
        if f == "wrong_terms":
            price = rng.randrange(round(self.target * 0.8), self.budget - 50)
            if rng.random() < 0.5:
                other = rng.choice([c for c in self.catalog if c["product_id"] != self.pid])
                self.seller_says((rng.choice(S_WRONG_PID), (other["product_id"], min(price, self.budget // 2), 1)))
            else:
                q = rng.choice((2, 3))
                self.seller_says((rng.choice(S_WRONG_QTY).format(q=q), (self.pid, price // q, q)))
            self.buyer(rng.choice(ASK_RIGHT).format(noun=self.noun, pid=self.pid))
            self.seller_says((rng.choice(S_OFFER), (self.pid, price, 1)))
            if price <= self.target:
                return self.order(price)
            self.buyer(self.counter_below(price))
            self.seller_says((rng.choice(S_FINAL), (self.pid, price, 1)))
            return self.order(price)
        if f == "price_up_latest":
            first = rng.randrange(self.target + 100, self.budget - 400)
            self.seller_says((rng.choice(S_OFFER), (self.pid, first, 1)))
            self.buyer(self.counter_below(first))
            raised = rng.randrange(first + 100, self.budget - 50)
            self.seller_says((S_UP[1], (self.pid, raised, 1)))
            return self.order(raised)
        if f == "few_turns":
            price = legal_high()
            self.seller_says((rng.choice(S_OFFER), (self.pid, price, 1)))
            self.buyer(self.counter_below(price))
            self.seller_says((rng.choice(S_SAME), (self.pid, price, 1)))
            return self.order(price)
        if f == "below_target_first":
            price = rng.randrange(round(self.target * 0.8), self.target)
            self.seller_says((rng.choice(S_OFFER), (self.pid, price, 1)))
            return self.order(price)
        raise ValueError(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=sorted(FLOWS), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--per-flow", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    interface = json.load(open(INTERFACE, encoding="utf-8"))
    rng = random.Random(a.seed)
    with open(a.out, "x", encoding="utf-8") as out:
        for flow in FLOWS[a.split]:
            for k in range(a.per_flow):
                n = Negotiation(rng, interface, flow, k)
                while n.budget - 600 <= n.target + 300:  # room for legal prices above the target (and a raise)
                    n = Negotiation(rng, interface, flow, k)
                n.run()
                out.write(json.dumps({"id": f"be_{a.split}_{flow}_{k}", "source": "buyer_eval", "family": f"buyer_flow_{flow}", "task": "part2",
                                      "flow": flow, "reference_price": n.reference_price, "tools": n.tools, "messages": n.messages}, ensure_ascii=False) + "\n")
    print(f"{a.split}: {len(FLOWS[a.split]) * a.per_flow} negotiations -> {a.out}")


if __name__ == "__main__":
    main()

"""E26 buyer negotiations: public-like shape, the E26 acceptance rule (user decision 2026-11-09 ~15:35).

    python code/data/gen_buyer_e26.py --split train  --seed 26101 --negotiations 1000 --out FILE
    python code/data/gen_buyer_e26.py --split select --seed 26001 --per-flow 28 --out FILE
    python code/data/gen_buyer_e26.py --split confirm --seed 26002 --per-flow 22 --out FILE

Written after reading the public negotiation set for its shape (allowed by the user, following the
TA's spoken guidance to synthesize from the public test data): 16 turns at the start, short talks,
long natural seller messages, a catalog line "name — brand — type — list price — rating". Nothing
from public is copied: brands, products, prices, ids and every sentence here are invented.

The buyer rule (reference replies), decided from what is visible at the step:
  legal  = right product id, right qty, price x qty <= budget
  accept = legal and (price <= target or price < 1.05 x target or this is the 2nd+ offer seen)
  accept -> order at the latest offer;  legal but not accept -> counter + wait;
  over budget -> counter + wait, end_session after the seller's 3rd over-budget offer or a
  final-worded over-budget offer;  wrong product / qty -> ask for the right offer.
Splits use disjoint flows and their own seller wording; the buyer wording is shared.
Reuses gen_buyer.py's Talk (message layout, ids, turn counter) and Buyer counter rule.
"""
import argparse
import json
import random
import sys

sys.path.insert(0, "code/data")

import gen_buyer as G  # noqa: E402
from common import first_problem  # noqa: E402

BRANDS = ("Lunaria", "Okeo", "Brightfield", "Velmora", "Tidepool", "Halden", "Corvina", "Asterly", "Mirelle", "Oakwick", "Saffra", "Nordlys")
PRODUCTS = {
    "Skin Care": [("Hydrating Gel Cleanser", "Cleanser", "6.8 fl oz", 9, 28), ("Overnight Repair Cream", "Moisturizer", "1.7 oz", 14, 46), ("Vitamin C Brightening Serum", "Serum", "1 fl oz", 12, 40),
                  ("Mineral Sunscreen SPF 40", "Sunscreen", "3 oz", 10, 32), ("Clay Detox Mask", "Face Mask", "4 oz", 8, 26)],
    "Hair Care": [("Argan Repair Shampoo", "Shampoo", "12 fl oz", 7, 24), ("Leave-In Smoothing Conditioner", "Conditioner", "8 fl oz", 8, 25), ("Heat Shield Spray", "Styling Spray", "5 fl oz", 6, 21),
                  ("Scalp Care Tonic", "Scalp Treatment", "3.4 fl oz", 11, 35)],
    "Bath & Body": [("Shea Butter Body Lotion", "Body Lotion", "16 fl oz", 6, 22), ("Eucalyptus Bath Salts", "Bath Salts", "2 lb", 7, 24), ("Exfoliating Sugar Scrub", "Body Scrub", "10 oz", 8, 27),
                    ("Lavender Body Wash", "Body Wash", "18 fl oz", 5, 18)],
    "Makeup": [("Long-Wear Liquid Liner", "Eyeliner", "0.1 fl oz", 5, 19), ("Satin Matte Lipstick", "Lipstick", "0.12 oz", 6, 22), ("Buildable Volume Mascara", "Mascara", "0.3 fl oz", 6, 21),
               ("Soft Focus Setting Powder", "Face Powder", "0.35 oz", 9, 30)],
    "Oral Care": [("Charcoal Whitening Toothpaste", "Toothpaste", "4 oz", 3, 12), ("Mint Alcohol-Free Mouthwash", "Mouthwash", "16 fl oz", 4, 14), ("Expanding Floss Picks", "Floss", "90 count", 3, 10)],
}

W = {  # seller wording per split: (open, move, final, accept, chat)
    "train": {
        "open": ("Thanks for reaching out about the {name}. I have it in stock and can ship today; {price} is where I'd start for one unit.",
                 "Good to hear from you. The {name} is a {rating}-rated {kind} and this one is sealed and fresh, so I'm offering it at {price}.",
                 "Happy to help with the {name}. For a single unit I can do {price}, which is already a bit under what most shops ask."),
        "move": ("I hear you on the price. I can come down to {price} for the {name}; that still covers my cost.",
                 "Let me meet you partway. {price} for the {name}, and it goes out with today's shipment.",
                 "I can sharpen it a little: {price} for one {kind}, still a good value for the quality."),
        "final": ("I've gone as far as I can on the {name}. {price} is my best and final number.",
                  "That's my last step on this one: {price} for the {name}. I can't go any lower."),
        "accept": ("{price} works for me. That's a fair split, so here's the offer for one {name}.",
                   "Alright, {price} it is. I'll send the binding offer for the {name} at that price."),
        "chat": ("Before I quote, can you tell me a bit about what you need the {name} for? I want to make sure it's the right {kind}.",),
        "wrong_item": ("I'm out of that exact one, but this similar {kind} is close and I can offer it right away.",),
        "wrong_qty": ("These usually go as a pack of {qty}, so I've put the offer together that way.",),
    },
    "select": {
        "open": ("Hi! Yes, the {name} is available. It's rated {rating} out of 5 and I can let it go for {price} for one.",
                 "Thanks for asking about the {name}. My price for one {kind} is {price}, and it ships in its original packaging."),
        "move": ("Fair enough, you've made your point. I'll drop the {name} to {price}.", "I want this to work for you, so {price} for the {name} is my adjusted offer."),
        "final": ("Honestly, {price} is the lowest I can take for the {name}. That's my final offer.",),
        "accept": ("You drive a fair bargain. {price} for the {name} is fine, and the offer below reflects it.",),
        "chat": ("Quick question first: do you need the {name} for daily use or as a gift? I can recommend the right size.",),
        "wrong_item": ("I don't have that listing right now, but this alternative {kind} is very similar.",),
        "wrong_qty": ("I can do a set of {qty} at this unit price, which is my usual bundle.",),
    },
    "confirm": {
        "open": ("Good afternoon. The {name} you asked about is ready to go; I'm quoting {price} for a single unit.",
                 "I've got the {name} on hand, a {rating}-star {kind}. One unit would be {price}."),
        "move": ("Since you're serious about it, I'll bring the {name} down to {price}.",),
        "final": ("I really can't do better than {price} on the {name}. That price is final.",),
        "accept": ("Okay, I'll match {price} for the {name}. The updated offer is attached.",),
        "chat": ("Can I ask what matters most to you with the {name}: price, size, or delivery speed?",),
        "wrong_item": ("That one's sold out, so I'm offering a comparable {kind} instead.",),
        "wrong_qty": ("I'd rather sell the {name} as a bundle of {qty}; here's that offer.",),
    },
}
OPENINGS = ("Hi there, I'm looking for the {name} ({pid}), one unit. What's the best price you can do?",
            "Hello! I'd like to buy one {name}. Could you send me your price for a single unit?",
            "Hi, I need one {name} ({pid}). I'd like to keep things fair, so what can you offer?",
            "Good day. I'm interested in the {name}, just one. What would you charge?")
LEADS_LEGAL = ("That's a bit more than I hoped to spend.", "Thanks for the offer, but I'd like a little more room.", "I appreciate it, though it's still above what I had in mind.", "")
LEADS_OVER = ("That's well beyond what I can do.", "I can't stretch that far.", "That price doesn't work for me.")
ASK_RIGHT_ITEM = ("Thanks, but I'm only after the {name} ({pid}), one unit. Could you send an offer for that one?",)
ASK_RIGHT_QTY = ("I only need one {name}, not {qty}. Could you make the offer for a single unit?",)
ANSWER_CHAT = ("It's for everyday use, just one unit. What's your price on the {name}?",)
TAKE = ("Great, that works. I'll take it.", "That's fair, let's do it.", "Deal, thank you.", "")
LEAVE = ("Thanks for your time, but that's more than I can spend. I'll pass.", "I'm afraid we're too far apart on price. Thanks anyway.")

FLOWS = {
    "train": ("first_at_target", "first_near", "second_legal", "over_then_legal", "over_leave", "wrong_then_legal", "counter_accepted", "chat_then_offer"),
    "select": ("two_offers_latest", "far_then_accept_counter", "over_over_legal", "wrong_qty_then_near", "near_after_chat", "over_final_leave", "first_far_second_far_legal"),
    "confirm": ("exact_budget_second", "boundary_105", "one_cent_over_then_leave", "wrong_item_then_at_target", "search_then_second_legal"),
}


def money(c):
    return f"${c // 100}.{c % 100:02d}"


class World:
    def __init__(self, rng):
        self.category = rng.choice(sorted(PRODUCTS))
        self.seller_id = f"s{rng.randrange(1, 9)}_seller"
        ids, self.catalog = set(), []
        items = rng.sample([(p, b) for p in PRODUCTS[self.category] for b in BRANDS], 6)
        for (prod, kind, size, low, high), brand in items:
            while True:
                pid = "B0" + "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789") for _ in range(8))
                if pid not in ids:
                    ids.add(pid)
                    break
            listed = rng.randrange(low * 100, high * 100)
            listed = listed - listed % 100 + rng.choice((99, 95, 49, 50, 0, 25))
            rating = rng.choice(("4.1", "4.3", "4.4", "4.5", "4.6", "4.7"))
            name = f"{brand} {prod}, {size}"
            spec = f"{name} — brand {brand} — type: {kind} — list price {money(listed)} — rating {rating}/5"
            self.catalog.append({"category": self.category, "product_id": pid, "spec": spec, "noun": name, "list": listed, "kind": kind.lower(), "rating": rating})
        self.chosen = rng.choice(self.catalog)
        self.others = [c for c in self.catalog if c is not self.chosen]
        self.pid, self.noun, self.listed = self.chosen["product_id"], self.chosen["noun"], self.chosen["list"]
        self.kind, self.rating = self.chosen["kind"], self.chosen["rating"]
        self.budget = round(self.listed * rng.uniform(0.80, 1.15))
        self.target = round(self.budget * rng.uniform(0.72, 0.92))

    def system(self, rng, interface):
        return G.World.system(self, rng, interface)


def accept(world, cents, n_offers):
    return cents <= world.target or cents * 100 < world.target * 105 or n_offers >= 2


class Neg:
    def __init__(self, rng, interface, split, flow):
        self.rng, self.split, self.flow, self.w = rng, split, flow, W[split]
        while True:
            self.world = World(rng)
            wd = self.world
            if wd.budget - round(wd.target * 1.05) >= 150 and wd.target >= 300:
                break
        self.talk = G.Talk(rng, self.world, interface)
        self.talk.turns_left = 16
        self.talk.messages[1]["content"] = self.talk.state_line()
        self.buyer = G.Buyer(rng, self.world, self.talk)
        self.n_offers = 0
        self.over_seen = 0

    def say(self, key, cents=None, pid=None, qty=1, offer=True, **kw):
        wd = self.world
        text = self.rng.choice(self.w[key]).format(name=wd.noun, kind=wd.kind, rating=wd.rating, price=money(cents) if cents else "", qty=qty, **kw)
        blocks = [G.Block(text, (pid or wd.pid, cents, qty), "offer", final=(key == "final")) if offer else G.Block(text, tag="pitch")]
        self.talk.deliver(blocks)
        if offer:
            self.n_offers += 1

    def say_two(self, key1, c1, key2, c2):
        wd = self.world
        b1 = G.Block(self.rng.choice(self.w[key1]).format(name=wd.noun, kind=wd.kind, rating=wd.rating, price=money(c1), qty=1), (wd.pid, c1, 1), "offer")
        b2 = G.Block("Sorry, correction on the price for the " + wd.noun + ": " + money(c2) + " is the right figure.", (wd.pid, c2, 1), "offer")
        self.talk.deliver([b1, b2])
        self.n_offers += 2

    def opening(self, search=False):
        wd, rng = self.world, self.rng
        if search:  # the search is the opening turn; the seller speaks next (as in public and gen_buyer.py)
            results = [{k: c[k] for k in ("category", "product_id", "spec")} for c in wd.catalog if c["kind"] == wd.kind]
            self.talk.reply("", "search_catalog", {"query": wd.kind}, {"count": len(results), "results": results})
            return
        self.talk.wait(rng.choice(OPENINGS).format(name=wd.noun, pid=wd.pid))

    # buyer actions following the E26 rule
    def respond(self, cents, qty=1, pid=None):
        wd, rng = self.world, self.rng
        pid = pid or wd.pid
        if pid != wd.pid:
            return self.talk.wait(rng.choice(ASK_RIGHT_ITEM).format(name=wd.noun, pid=wd.pid))
        if qty != 1:
            return self.talk.wait(rng.choice(ASK_RIGHT_QTY).format(name=wd.noun, qty=qty))
        if cents > wd.budget:
            self.over_seen += 1
            if self.over_seen >= 3 or self.talk.final:
                return self.leave()
            return self.talk.wait(self.buyer.counter_text(rng.choice(LEADS_OVER)))
        if accept(wd, cents, self.n_offers):
            return self.order()
        return self.talk.wait(self.buyer.counter_text(rng.choice(LEADS_LEGAL)))

    def order(self):
        pid, cents, qty = self.talk.offer
        text = self.rng.choice(TAKE)
        self.talk.reply(text, "order", {"product_id": pid, "agree_price": cents / 100, "qty": qty, "evidence_message_ids": [self.talk.offer_id]})
        self.done = "order"

    def leave(self):
        self.talk.reply(self.rng.choice(LEAVE), "end_session", {"reason": "price above budget"})
        self.done = "leave"

    # price helpers (cents)
    def at_target(self):
        return self.rng.randrange(round(self.world.target * 0.8), self.world.target + 1)

    def near(self):
        t = self.world.target
        return self.rng.randrange(t + 1, (t * 105 + 99) // 100)  # strictly between target and 1.05 x target

    def far_legal(self):
        t, b = self.world.target, self.world.budget
        lo = (t * 105 + 99) // 100 + 1
        return self.rng.randrange(lo, b + 1)

    def over(self):
        b = self.world.budget
        return b + self.rng.randrange(1, max(2, b // 5))

    def run(self):
        f, rng, wd = self.flow, self.rng, self.world
        self.done = None
        self.opening(search=(f == "search_then_second_legal") or rng.random() < 0.08)
        if f == "first_at_target":
            self.say("open", self.at_target()); self.respond(self.talk.offer[1])
        elif f == "first_near":
            self.say("open", self.near()); self.respond(self.talk.offer[1])
        elif f in ("second_legal", "first_far_second_far_legal"):
            first = self.far_legal(); self.say("open", first); self.respond(first)
            second = rng.randrange(min(first, (wd.target * 105 + 99) // 100 + 1), first + 1)
            self.say("move", second); self.respond(second)
        elif f == "over_then_legal":
            self.say("open", self.over()); self.respond(self.talk.offer[1])
            self.say("move", rng.choice((self.far_legal(), self.near(), self.at_target()))); self.respond(self.talk.offer[1])
        elif f in ("over_leave", "over_final_leave"):
            price = self.over() + rng.randrange(100, 400)
            for k in range(3):
                key = "final" if (k == 2 or (f == "over_final_leave" and k == 1)) else ("move" if k else "open")
                self.say(key, price); self.respond(price)
                if self.done:
                    break
                price = max(wd.budget + 1, price - rng.randrange(20, 150))
        elif f in ("wrong_then_legal", "wrong_item_then_at_target"):
            other = rng.choice(wd.others)
            self.say("wrong_item", min(other["list"], wd.budget), pid=other["product_id"]); self.respond(self.talk.offer[1], pid=other["product_id"])
            price = self.at_target() if f == "wrong_item_then_at_target" else self.far_legal()
            self.say("open", price); self.respond(price)
        elif f == "wrong_qty_then_near":
            q = rng.choice((2, 3))
            self.say("wrong_qty", self.near(), qty=q); self.respond(self.talk.offer[1], qty=q)
            self.say("open", self.near()); self.respond(self.talk.offer[1])
        elif f in ("counter_accepted", "far_then_accept_counter"):
            first = self.far_legal(); self.say("open", first); self.respond(first)
            self.say("accept", self.buyer.counter); self.respond(self.buyer.counter)
        elif f in ("chat_then_offer", "near_after_chat"):
            self.say("chat", offer=False)
            self.talk.wait(rng.choice(ANSWER_CHAT).format(name=wd.noun))
            price = self.near() if f == "near_after_chat" else rng.choice((self.at_target(), self.near(), self.far_legal()))
            self.say("open", price); self.respond(price)
            if not self.done:
                second = rng.randrange(wd.target, price + 1); self.say("move", second); self.respond(second)
        elif f == "two_offers_latest":
            hi, lo = self.over(), self.at_target()
            self.say_two("open", hi, "move", lo); self.respond(lo)
        elif f == "over_over_legal":
            p = self.over() + rng.randrange(50, 300)
            self.say("open", p); self.respond(p)
            p2 = max(wd.budget + 1, p - rng.randrange(20, 120)); self.say("move", p2); self.respond(p2)
            self.say("move", self.far_legal()); self.respond(self.talk.offer[1])
        elif f == "exact_budget_second":
            self.say("open", self.over()); self.respond(self.talk.offer[1])
            self.say("move", wd.budget); self.respond(wd.budget)
        elif f == "boundary_105":
            edge = (wd.target * 105 + 99) // 100  # first price with price*100 >= target*105: not "near"
            self.say("open", edge); self.respond(edge)
            self.say("move", rng.randrange(wd.target, edge)); self.respond(self.talk.offer[1])
        elif f == "one_cent_over_then_leave":
            self.say("open", wd.budget + 1); self.respond(wd.budget + 1)
            self.say("final", wd.budget + 1); self.respond(wd.budget + 1)
        elif f == "search_then_second_legal":
            first = self.far_legal(); self.say("open", first); self.respond(first)
            self.say("move", rng.randrange(min(first, (wd.target * 105 + 99) // 100 + 1), first + 1)); self.respond(self.talk.offer[1])
        else:
            raise ValueError(f)
        if not self.done:
            raise RuntimeError(f"{f} did not end")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=sorted(FLOWS), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--negotiations", type=int, default=0)
    ap.add_argument("--per-flow", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    interface = json.load(open(G.INTERFACE, encoding="utf-8"))
    rng = random.Random(a.seed)
    flows = FLOWS[a.split]
    plan = [rng.choice(flows) for _ in range(a.negotiations)] if a.negotiations else [f for f in flows for _ in range(a.per_flow)]
    counts = {}
    with open(a.out, "x", encoding="utf-8") as out:
        for k, flow in enumerate(plan):
            n = Neg(rng, interface, a.split, flow)
            n.run()
            rec = {"id": f"e26{a.split[0]}_{flow}_{k}", "source": "e26_buyer", "family": f"e26{a.split[0]}_{flow}", "task": "part2", "flow": flow,
                   "tools": interface["tools"], "messages": n.talk.messages}
            if a.split != "train":
                o = n.talk.messages[-1]["tool_calls"][0]["function"]
                rec["reference_price"] = round(o["arguments"]["agree_price"] * 100) if o["name"] == "order" else None
            problem = first_problem(rec["tools"], rec["messages"])
            if problem:
                sys.exit(f"{rec['id']}: invalid step {problem}")
            counts[flow] = counts.get(flow, 0) + 1
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(a.split, len(plan), "negotiations", counts, "->", a.out)


if __name__ == "__main__":
    main()

"""E25 whole-negotiation buyer evaluation sets (select_new, confirm_new) with scripted sellers.

    python code/data/gen_buyer_eval_e25.py --split select --seed 25001 --per-flow 28 --out FILE
    python code/data/gen_buyer_eval_e25.py --split confirm --seed 25002 --per-flow 22 --out FILE

Builds on gen_buyer_eval.py's Negotiation (the E24 evaluation catalog, personas and message
layout, which gen_buyer.py and the E25 patch do not use) with new seller scripts and new seller
wording. None of these flows is a flow of the E24 sets or of gen_buyer_patch_e25.py: they differ
in the order of events (where the deciding offer comes, what precedes it, how many offers a
message holds). They test the same visible rules the training data follows (order a legal offer
that is at/under target, final, repeated or at the deadline; never order above budget, a wrong
item or an outdated price), so they measure the behaviour, not an unseen rule. Lineage and the
shared parts are written to the E25 lineage file by the session, not here.

select:  drop_below_target, final_mid, final_over, counter_accepted, two_offers, chat_then_final,
         repeat_over, set_then_final, deadline_quiet
confirm: budget_exact, one_cent_over, raise_final_legal, wrong_item_at_target, stall_legal
"""
import argparse
import json
import random
import sys

sys.path.insert(0, "code/data")

import gen_buyer_eval as E  # noqa: E402

FLOWS = {"select": ("drop_below_target", "final_mid", "final_over", "counter_accepted", "two_offers", "chat_then_final", "repeat_over",
                    "set_then_final", "deadline_quiet"),
         "confirm": ("budget_exact", "one_cent_over", "raise_final_legal", "wrong_item_at_target", "stall_legal")}

V_OPEN = ("For the {noun} I'm starting from the number below, and it ships from my shelf this week.",
          "Here is where I'd like to start on the {noun}; it has been kept in its original packing.",
          "My asking price for the {noun} is in the offer below, cleaned and tested yesterday.")
V_MOVE = ("I can move a little on the {noun} since you asked nicely, here is the new number.",
          "Alright, I'll take a bit off the {noun} for you, see the updated offer.",
          "Let me sharpen the price on the {noun} a touch, this is the revised offer.")
V_FINAL = ("For the {noun}, this is my final offer and I won't revisit it.", "I can't go any lower on the {noun}, I have checked my margins twice.",
           "The {noun} price is final for me at this point, take it or leave it.")
V_SAME = ("I'm keeping the {noun} at exactly the same figure as my previous offer.", "Nothing changes on my side, the {noun} stays at the figure I already sent.")
V_ACCEPT = ("Fine, you've worn me down, I'll match your number on the {noun}.", "That figure for the {noun} works for me, sending it as an offer.")
V_CORRECT = ("Hold on, I typed the wrong figure for the {noun} a moment ago, the corrected offer follows.",)
V_TALK = ("For the {noun} I was picturing something close to ${p}, but I need to double check my notes first.",
          "Off the top of my head the {noun} would go for about ${p}, not a formal quote yet.")
V_SET = ("I'd rather move the {noun} as a bundle of {q}, it saves me repacking.",)
V_OTHER = ("I don't have that exact {noun} handy, but this other listing is close and cheaper.",)
V_RAISE = ("I underpriced the {noun} in my first message, the corrected and final price is in this offer.",)
# confirm_new has its own seller wording (E25 same-source check, 13:00: confirm must not reuse select's lines).
C_WORDS = {
    "V_OPEN": ("Since you asked about the {noun}, my opening number is in this offer and I can hold it until Friday.",
               "The {noun} you mentioned is available from my storage unit; the asking price is in the offer."),
    "V_MOVE": ("I checked my records and I can lower the {noun} slightly, the adjusted offer is attached.",
               "For a quick sale I'm willing to trim the {noun}, see the adjusted numbers."),
    "V_FINAL": ("That is my last offer for the {noun}; my supplier will not let me reduce it again.",
                "After thinking it over, the {noun} cannot go any lower than this figure."),
    "V_SAME": ("My offer on the {noun} has not changed since my previous message.", "I have reconsidered and the {noun} remains at the identical figure."),
    "V_RAISE": ("Apologies, the warehouse billed me more for the {noun}, so this corrected price is final.",),
    "V_OTHER": ("The exact {noun} you named is sold out, though this alternative listing is similar and a little cheaper.",),
}
B_ASK_FORMAL = ("Thanks. Could you send that as a formal offer so I can accept it?", "Please put the {noun} in a proper offer, one unit.")
B_ASK_SINGLE = ("I only need a single {noun}. Could you offer one unit?", "One {noun} is all I need, please send an offer for qty 1.")
B_ASK_RIGHT = ("That's a different listing. I'm after {pid}, the {noun}. Could you offer that one?",)


class Neg(E.Negotiation):
    def note(self, options, **kw):
        return self.rng.choice(options).format(noun=self.noun, **kw)

    def legal_above(self, top=None):
        top = self.budget if top is None else min(top, self.budget)
        return self.rng.randrange(self.target + 100, top + 1)

    def over(self):
        return self.budget + self.rng.randrange(150, max(300, self.budget // 6))

    def at_target(self):
        return self.rng.randrange(round(self.target * 0.8), self.target + 1)

    def falling(self, price, floor, n):
        """n offers falling from price, with buyer counters; returns the last price offered."""
        for k in range(n):
            self.seller_says((self.note(V_MOVE if k else V_OPEN), (self.pid, price, 1)))
            self.buyer(self.counter_below(price))
            nxt = max(floor, price - round(self.listed * self.rng.uniform(0.02, 0.07)))
            if nxt == price:
                break
            price = nxt
        return price

    def run(self):
        rng, f = self.rng, self.flow
        if f == "deadline_quiet":  # few turns from the start, so the deadline comes after a short haggle
            self.turns = rng.choice((6, 7))
            self.messages[1]["content"] = self.state()
        self.buyer(rng.choice(E.OPEN).format(noun=self.noun))
        if f == "drop_below_target":
            start = self.legal_above() if rng.random() < 0.5 else self.over()
            self.seller_says((self.note(V_OPEN), (self.pid, start, 1)))
            self.buyer(self.counter_below(start))
            self.seller_says((self.note(V_MOVE), (self.pid, self.at_target(), 1)))
            return self.order(self.latest_price())
        if f == "final_mid":
            last = self.falling(self.legal_above() + rng.randrange(0, 600), self.target + 200, rng.choice((2, 3)))
            self.seller_says((self.note(V_FINAL), (self.pid, self.legal_above(top=last), 1)))
            return self.order(self.latest_price())
        if f == "final_over":
            last = self.falling(self.over() + rng.randrange(200, 800), self.budget + 150, rng.choice((2, 3)))
            self.seller_says((self.note(V_FINAL), (self.pid, rng.randrange(self.budget + 100, last + 1), 1)))
            return self.walk()
        if f == "counter_accepted":
            start = self.legal_above()
            self.seller_says((self.note(V_OPEN), (self.pid, start, 1)))
            self.buyer(self.counter_below(start))
            self.seller_says((self.note(V_ACCEPT), (self.pid, self.counter, 1)))
            return self.order(self.counter)
        if f == "two_offers":
            if rng.random() < 0.5:
                low = self.at_target()
                self.seller_says((self.note(V_OPEN), (self.pid, self.legal_above(), 1)), (self.note(V_CORRECT), (self.pid, low, 1)))
                return self.order(low)
            self.seller_says((self.note(V_OPEN), (self.pid, self.at_target(), 1)), (self.note(V_CORRECT), (self.pid, self.over(), 1)))
            self.buyer(self.counter_below(self.budget))
            self.seller_says((self.note(V_FINAL), (self.pid, self.over(), 1)))
            return self.walk()
        if f == "chat_then_final":
            self.seller_says((self.note(V_TALK, p=E.money(self.legal_above())), None))
            self.buyer(self.note(B_ASK_FORMAL))
            price = self.legal_above()
            self.seller_says((self.note(V_FINAL), (self.pid, price, 1)))
            return self.order(price)
        if f == "repeat_over":
            price = self.over()
            for k in range(3):
                self.seller_says((self.note(V_SAME if k else V_OPEN), (self.pid, price, 1)))
                if k < 2:
                    self.buyer(self.counter_below(price))
            return self.walk()
        if f == "set_then_final":
            q = rng.choice((2, 3))
            self.seller_says((self.note(V_SET, q=q), (self.pid, self.legal_above(), q)))
            self.buyer(self.note(B_ASK_SINGLE))
            price = self.legal_above()
            self.seller_says((self.note(V_FINAL), (self.pid, price, 1)))
            return self.order(price)
        if f == "deadline_quiet":
            price = self.legal_above() if self.budget - self.target > 700 else self.budget
            while True:
                self.seller_says((self.note(V_MOVE if self.counter else V_OPEN), (self.pid, price, 1)))
                if self.turns <= 2:
                    return self.order(price)
                self.buyer(self.counter_below(price))
                nxt = max(self.target + 100, price - round(self.listed * rng.uniform(0.01, 0.03)))
                price = nxt if nxt != price else price - 1
        if f == "budget_exact":
            self.falling(self.over() + rng.randrange(100, 500), self.budget + 100, rng.choice((1, 2)))
            self.seller_says((self.note(V_FINAL), (self.pid, self.budget, 1)))
            return self.order(self.budget)
        if f == "one_cent_over":
            self.falling(self.over() + rng.randrange(100, 500), self.budget + 100, rng.choice((1, 2)))
            self.seller_says((self.note(V_FINAL), (self.pid, self.budget + 1, 1)))
            return self.walk()
        if f == "raise_final_legal":
            first = rng.randrange(self.target + 100, self.budget - 200)
            self.seller_says((self.note(V_OPEN), (self.pid, first, 1)))
            self.buyer(self.counter_below(first))
            raised = rng.randrange(first + 50, self.budget + 1)
            self.seller_says((self.note(V_RAISE), (self.pid, raised, 1)))
            return self.order(raised)
        if f == "wrong_item_at_target":
            other = rng.choice([c for c in self.catalog if c["product_id"] != self.pid])
            self.seller_says((self.note(V_OTHER), (other["product_id"], min(self.at_target(), self.budget // 2), 1)))
            self.buyer(B_ASK_RIGHT[0].format(pid=self.pid, noun=self.noun))
            price = self.at_target()
            self.seller_says((self.note(V_OPEN), (self.pid, price, 1)))
            return self.order(price)
        if f == "stall_legal":
            price = self.legal_above()
            self.seller_says((self.note(V_OPEN), (self.pid, price, 1)))
            self.buyer(self.counter_below(price))
            self.seller_says((self.note(V_SAME), (self.pid, price, 1)))
            return self.order(price)
        raise ValueError(f)

    def latest_price(self):
        import re
        last = [m for m in self.messages if m["role"] == "user"][-1]["content"]
        return int(round(float(re.findall(r"Offer: \S+ at \$(\d+\.\d{2})", last)[-1]) * 100))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=sorted(FLOWS), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--per-flow", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    interface = json.load(open(E.INTERFACE, encoding="utf-8"))
    if a.split == "confirm":
        globals().update(C_WORDS)
    rng = random.Random(a.seed)
    with open(a.out, "x", encoding="utf-8") as out:
        for flow in FLOWS[a.split]:
            for k in range(a.per_flow):
                n = Neg(rng, interface, flow, k)
                while n.budget - 700 <= n.target + 300:
                    n = Neg(rng, interface, flow, k)
                n.run()
                out.write(json.dumps({"id": f"e25{a.split[0]}_{flow}_{k}", "source": "buyer_eval_e25", "family": f"e25{a.split[0]}_{flow}", "task": "part2",
                                      "flow": flow, "reference_price": n.reference_price, "tools": n.tools, "messages": n.messages}, ensure_ascii=False) + "\n")
    print(f"{a.split}: {len(FLOWS[a.split]) * a.per_flow} negotiations -> {a.out}")


if __name__ == "__main__":
    main()

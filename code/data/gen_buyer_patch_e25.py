"""E25 buyer patch: short negotiations built around one decision each, in legal / illegal pairs.

    python code/data/gen_buyer_patch_e25.py --negotiations N --seed S --out FILE

Uses the training world of gen_buyer.py (catalog, personas, budget/target draw, message layout,
buyer counter rule), so the patch looks like the rest of data_v5 Part 2. What is new is the seller
script: each flow puts the buyer in front of one decision whose right answer depends on comparing
the visible numbers, and every flow has a twin where the same wording calls for the other action.

  final_legal    seller calls a price final; total within budget -> order at that price
  final_over     same wording, total above budget (sometimes by one cent) -> end_session
  first_at_target first offer at or under the target -> order at once
  switch_right   wrong item / wrong quantity first, then the right offer (at target -> order;
                 above target -> counter, then final legal -> order or final over -> leave)
  stalled        seller repeats the same price: legal -> order; above budget twice -> leave
  deadline       few turns left, no "final" words: legal -> order at <= 2 turns; over -> leave
  price_up       seller raises the price after a counter and calls it final: the latest price
                 decides (legal -> order at the new price, over -> leave)

Final wordings are new (none of the E24 or E25 evaluation phrasings); check_buyer_patch_e25.py
recognises them. The opening step of every negotiation is listed in `no_target` (data_v5 already
has many openings); every later buyer step is a training target. Money is integer cents.
"""
import argparse
import collections
import json
import random
import sys

sys.path.insert(0, "code/data")

import gen_buyer as G  # noqa: E402
from common import first_problem  # noqa: E402

FLOWS = ("final_legal", "final_over", "first_at_target", "switch_right", "stalled", "deadline", "price_up")
WEIGHTS = (5, 2, 4, 2, 3, 2, 3)

FINAL_E25 = ("This is the lowest I'll go on the {noun}.", "Final price on the {noun}, no more discounts.", "That's as low as I can go for this {noun}.",
             "I won't go below this for the {noun}.", "Bottom line on the {noun}, this is it.", "Final price. I can't shave anything else off the {noun}.")
SAME_E25 = ("I'll keep the {noun} at the same number.", "The {noun} stays at that price.", "Same number as before for the {noun}.",
            "I'm holding the {noun} where it is.")
CONCEDE_E25 = ("I can trim the {noun} a little.", "Let me knock a bit off the {noun}.", "OK, a small cut on the {noun}.", "I'll come down some on the {noun}.")
OPEN_E25 = ("Here's what I'm asking for the {noun}.", "My price for the {noun} is below.", "This is my starting price on the {noun}.", "")
UP_E25 = ("My supplier raised the cost of the {noun}, so the new price is final.", "I priced the {noun} too low before; the corrected price is final.")


class Script:
    def __init__(self, rng, interface, flow, short_turns=False):
        self.rng, self.flow = rng, flow
        while True:
            self.world = G.World(rng)
            w = self.world
            if w.budget - w.target >= 400 and w.target >= 600:
                break
        self.talk = G.Talk(rng, self.world, interface)
        if short_turns:
            self.talk.turns_left = rng.choice((4, 5, 6))
            self.talk.messages[1]["content"] = self.talk.state_line()
        self.buyer = G.Buyer(rng, self.world, self.talk)
        self.actions = []

    def note(self, options):
        return self.rng.choice(options).format(noun=self.world.noun)

    def say(self, *blocks):
        self.talk.deliver(list(blocks))

    def offer(self, cents, note="", pid=None, qty=1):
        return G.Block(note, (pid or self.world.pid, cents, qty), "offer")

    def legal_above(self):
        w = self.world
        if self.rng.random() < 0.12:
            return w.budget  # exactly the budget is still legal
        return self.rng.randrange(w.target + 100, w.budget + 1)

    def over(self):
        w = self.world
        if self.rng.random() < 0.15:
            return w.budget + 1  # one cent over
        return w.budget + self.rng.randrange(50, max(100, w.budget // 6))

    def at_target(self):
        w = self.world
        return self.rng.randrange(round(w.target * 0.8), w.target + 1)

    def counter(self, hard=False):
        lead = self.rng.choice(G.HARD_LEADS if hard else G.SOFT_LEADS)
        self.talk.wait(self.buyer.counter_text(lead))
        self.actions.append("counter_illegal" if hard else "counter_legal")

    def order(self):
        self.buyer.close_with_order()
        self.actions.append("order")

    def leave(self):
        self.talk.reply(self.rng.choice(G.LEAVE), "end_session", {"reason": "no acceptable deal"})
        self.actions.append("leave")

    def ask_right(self, wrong_pid=None, qty=None):
        if wrong_pid:
            self.talk.wait(self.buyer.fill(self.rng.choice(G.WRONG_PRODUCT), wrong=wrong_pid))
        else:
            self.talk.wait(self.buyer.fill(self.rng.choice(G.WRONG_QTY), qty=qty))
        self.actions.append("ask_right")

    def haggle_then_final(self, legal):
        """One to three falling offers with counters, then a final offer no higher than the last one.
        legal: the final total is within budget (the earlier offers may be above it); otherwise every
        offer, the final one included, stays above the budget."""
        rng, w = self.rng, self.world
        price = rng.choice((self.legal_above(), self.over())) if legal else self.over() + rng.randrange(100, 600)
        floor = w.target + 100 if legal else w.budget + 1
        for k in range(rng.choice((1, 2, 3))):
            self.say(self.offer(price, self.note(CONCEDE_E25 if k else OPEN_E25)))
            self.counter(hard=price > w.budget)
            last, price = price, max(floor, price - round(w.listed * rng.uniform(0.02, 0.08)))
            if price == last:  # no room left to concede: go straight to the final offer
                break
        if legal:
            top = min(price, w.budget)
            final = top if rng.random() < 0.15 else rng.randrange(min(w.target + 100, top), top + 1)
        else:
            final = w.budget + 1 if rng.random() < 0.15 else rng.randrange(w.budget + 1, price + 1)
        self.say(self.offer(final, self.note(FINAL_E25)))

    def run(self):
        rng, w, f = self.rng, self.world, self.flow
        self.buyer.open()
        if f == "final_legal":
            self.haggle_then_final(True)
            return self.order()
        if f == "final_over":
            self.haggle_then_final(False)
            return self.leave()
        if f == "first_at_target":
            if rng.random() < 0.4:
                self.say(G.Block(rng.choice(G.PITCHES).format(noun=w.noun), tag="pitch"))
                self.talk.wait(self.buyer.fill(rng.choice(G.ASK_PRICE)))
                self.actions.append("ask_price")
            self.say(self.offer(self.at_target(), self.note(OPEN_E25)))
            return self.order()
        if f == "switch_right":
            if rng.random() < 0.5:
                other = rng.choice(w.others)["product_id"]
                self.say(self.offer(min(self.at_target(), w.budget // 2), rng.choice(G.SWAP_NOTES), pid=other))
                self.ask_right(wrong_pid=other)
            else:
                q = rng.choice((2, 3))
                self.say(self.offer(self.at_target(), rng.choice(G.SET_NOTES).format(qty=q), qty=q))
                self.ask_right(qty=q)
            branch = rng.random()
            if branch < 0.4:
                self.say(self.offer(self.at_target(), self.note(OPEN_E25)))
                return self.order()
            if branch < 0.7:
                first = rng.randrange(w.target + 150, w.budget + 1)
                self.say(self.offer(first, self.note(OPEN_E25)))
                self.counter()
                self.say(self.offer(rng.randrange(w.target + 100, first), self.note(FINAL_E25)))
                return self.order()
            first = self.over() + rng.randrange(50, 400)
            self.say(self.offer(first, self.note(OPEN_E25)))
            self.counter(hard=True)
            self.say(self.offer(rng.randrange(w.budget + 1, first), self.note(FINAL_E25)))
            return self.leave()
        if f == "stalled":
            if rng.random() < 0.55:
                price = rng.randrange(w.target + 100, w.budget + 1)
                self.say(self.offer(price, self.note(OPEN_E25)))
                self.counter()
                self.say(self.offer(price, self.note(SAME_E25)))
                return self.order()
            price = self.over()
            self.say(self.offer(price, self.note(OPEN_E25)))
            self.counter(hard=True)
            self.say(self.offer(price, self.note(SAME_E25)))
            self.counter(hard=True)
            self.say(self.offer(price, self.note(SAME_E25)))
            return self.leave()
        if f == "deadline":
            legal = rng.random() < 0.6
            price = rng.randrange(w.target + 300, w.budget + 1) if legal else self.over() + rng.randrange(100, 500)
            floor = w.target + 100 if legal else w.budget + 1
            first = True
            while True:
                if not first:  # falling prices; a legal price never repeats (that would be the "stalled" case)
                    last, price = price, max(floor, price - round(w.listed * rng.uniform(0.01, 0.04)))
                    if legal and price == last:
                        price = last - 1
                first = False
                self.say(self.offer(price, self.note(CONCEDE_E25 if self.actions else OPEN_E25)))
                if self.talk.turns_left <= 2:
                    return self.order() if legal else self.leave()
                self.counter(hard=not legal)
        if f == "price_up":
            first = rng.randrange(w.target + 100, w.budget - 150)
            self.say(self.offer(first, self.note(OPEN_E25)))
            self.counter()
            if rng.random() < 0.55:
                self.say(self.offer(rng.randrange(first + 50, w.budget + 1), self.note(UP_E25)))
                return self.order()
            self.say(self.offer(self.over(), self.note(UP_E25)))
            return self.leave()
        raise ValueError(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--negotiations", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    interface = json.load(open(G.INTERFACE, encoding="utf-8"))
    rng = random.Random(a.seed)
    flows, actions = collections.Counter(), collections.Counter()
    with open(a.out, "x", encoding="utf-8") as out:
        for n in range(a.negotiations):
            flow = rng.choices(FLOWS, WEIGHTS)[0]
            s = Script(rng, interface, flow, short_turns=flow == "deadline")
            s.run()
            msgs = s.talk.messages
            first_assistant = next(i for i, m in enumerate(msgs) if m["role"] == "assistant")
            record = {"id": f"e25p_{a.seed}_{n}", "source": "e25_patch", "family": f"e25p_{flow}", "task": "part2", "flow": flow,
                      "no_target": [first_assistant], "tools": interface["tools"], "messages": msgs}
            problem = first_problem(record["tools"], msgs)
            if problem:
                sys.exit(f"{record['id']}: invalid step {problem}")
            flows[flow] += 1
            actions.update(s.actions)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(a.out, "flows", dict(flows), "target actions", dict(actions))


if __name__ == "__main__":
    main()

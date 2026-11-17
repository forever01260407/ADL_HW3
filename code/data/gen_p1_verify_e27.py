"""E27 Part 1: policy-bound service conversations where the policy forces lookups and an eligibility
check before the action (the most frequent E23 must-call error type on public: going straight from
a list result to the action and skipping the required check/lookup).

    python code/data/gen_p1_verify_e27.py --split train --seed 27101 --conversations 1500 --out FILE
    python code/data/gen_p1_verify_e27.py --split dev --seed 27001 --conversations 160 --out FILE

Shape learned from reading public Part 1 (user decision 2026-11-09 ~16:30): a numbered policy in the
system message, identity by full name + date of birth through a search tool (similar names told
apart by date of birth), lookups to resolve ids, a site/location resolved from its name, an
eligibility check required before the action, multi-turn users who give values in different
messages. Every organisation, tool name, field, person, value and sentence here is invented; no
public text, tool, or value is used. Tool results are computed from a small world, so each
argument comes from something visible before the step. Families are (domain, flow); dev holds out
whole domains.
"""
import argparse
import json
import random
import sys

sys.path.insert(0, "code/data")
from common import call_message, first_problem, make_tool, tool_message  # noqa: E402

FIRST = "Ava Noah Mia Liam Zoe Ethan Lena Omar Priya Kenji Sofia Mateo Hana Tariq Elena Jonas Amara Diego Yuki Ingrid Ravi Chloe Hugo Nadia Felix Rosa".split()
LAST = "Chen Patel Garcia Kim Novak Silva Haddad Larsen Okafor Rossi Tanaka Moreau Singh Kowalski Reyes Fischer Nguyen Ali Berg Costa Duarte Lindqvist".split()
SIMILAR = {"Chen": "Chan", "Patel": "Patil", "Garcia": "Garza", "Kim": "Kimm", "Novak": "Nowak", "Silva": "Sylva", "Rossi": "Rossa", "Singh": "Sing", "Reyes": "Reyna",
           "Fischer": "Fisher", "Nguyen": "Nguyenn", "Costa": "Coster", "Berg": "Bergh", "Tanaka": "Tanabe", "Larsen": "Larson", "Moreau": "Moreno"}
MONTHS = "January February March April May June July August September October November December".split()

# domain: org, person/item/site nouns, tool names, item attributes, action extra parameter
DOMAINS = {
    "tool_library": dict(org="Harbor Point Tool Library", person="member", item="loan", site="pickup counter", search="lookup_members", list_items="list_member_loans",
                         list_sites="list_counters", check="check_extension_eligibility", action="extend_loan", act_verb="extend", extra=("extra_days", (7, 14)),
                         decoys=("get_loan_record", "return_loan", "list_fines"), things=("cordless drill", "tile saw", "pressure washer", "ladder", "hedge trimmer")),
    "clinic_lab": dict(org="Northgate Clinic Lab", person="patient", item="lab order", site="draw station", search="find_patient", list_items="list_lab_orders",
                       list_sites="list_draw_stations", check="check_booking_window", action="book_blood_draw", act_verb="book", extra=("time_slot", ("08:30", "10:00", "13:15")),
                       decoys=("get_lab_order", "cancel_blood_draw", "list_results"), things=("lipid panel", "thyroid panel", "vitamin D test", "A1C test")),
    "parking": dict(org="Riverside Parking Office", person="resident", item="permit", site="parking zone", search="search_residents", list_items="list_resident_permits",
                    list_sites="list_zones", check="check_renewal_eligibility", action="renew_permit", act_verb="renew", extra=("months", (3, 6, 12)),
                    decoys=("get_permit", "revoke_permit", "list_violations"), things=("annual street permit", "visitor permit", "motorcycle permit")),
    "gym": dict(org="Summit Fitness Club", person="member", item="class pass", site="studio", search="lookup_member", list_items="list_class_passes",
                list_sites="list_studios", check="check_class_capacity", action="reserve_class", act_verb="reserve", extra=("class_date", ("2026-11-12", "2026-11-14", "2026-11-18")),
                decoys=("get_pass", "cancel_reservation", "list_trainers"), things=("spin pass", "yoga pass", "pilates pass", "boxing pass")),
    "vet": dict(org="Willow Creek Animal Hospital", person="client", item="pet record", site="clinic", search="find_client", list_items="list_client_pets",
                list_sites="list_clinics", check="check_vaccine_due", action="schedule_vaccination", act_verb="schedule", extra=("vaccine", ("rabies", "distemper", "bordetella")),
                decoys=("get_pet", "cancel_appointment", "list_invoices"), things=("dog", "cat", "puppy", "kitten")),
    "garden": dict(org="Eastfield Community Gardens", person="gardener", item="plot lease", site="garden site", search="search_gardeners", list_items="list_plot_leases",
                   list_sites="list_garden_sites", check="check_transfer_eligibility", action="transfer_plot", act_verb="transfer", extra=("plot_size", ("small", "medium", "large")),
                   decoys=("get_lease", "end_lease", "list_waitlist"), things=("raised bed lease", "ground plot lease", "greenhouse bench lease")),
    "coworking": dict(org="Foundry Coworking", person="member", item="membership", site="location", search="lookup_coworker", list_items="list_memberships",
                      list_sites="list_locations", check="check_upgrade_eligibility", action="upgrade_membership", act_verb="upgrade", extra=("plan", ("dedicated desk", "private office")),
                      decoys=("get_membership", "pause_membership", "list_invoices"), things=("hot desk membership", "weekend membership", "flex membership")),
    "marina": dict(org="Bayview Marina", person="boat owner", item="mooring contract", site="dock", search="find_owner", list_items="list_mooring_contracts",
                   list_sites="list_docks", check="check_relocation_eligibility", action="relocate_mooring", act_verb="relocate", extra=("berth_length_m", (8, 10, 12)),
                   decoys=("get_contract", "terminate_contract", "list_fees"), things=("seasonal mooring", "annual mooring", "winter storage")),
    "school": dict(org="Lakeside Adult Education", person="student", item="enrollment", site="campus", search="search_students", list_items="list_enrollments",
                   list_sites="list_campuses", check="check_section_seats", action="switch_section", act_verb="switch", extra=("section", ("A", "B", "C")),
                   decoys=("get_enrollment", "withdraw", "list_grades"), things=("Spanish I", "Bookkeeping Basics", "Intro to Python", "Watercolor")),
    "transit": dict(org="Metro Transit Card Center", person="cardholder", item="fare card", site="service center", search="lookup_cardholder", list_items="list_fare_cards",
                    list_sites="list_service_centers", check="check_replacement_eligibility", action="order_replacement_card", act_verb="replace", extra=("delivery", ("mail", "pickup")),
                    decoys=("get_card", "block_card", "list_trips"), things=("reduced fare card", "monthly pass card", "student card")),
    "pharmacy_bulk": dict(org="Greenleaf Supply Co-op", person="account holder", item="standing order", site="depot", search="search_accounts", list_items="list_standing_orders",
                          list_sites="list_depots", check="check_order_change_window", action="change_standing_order", act_verb="change", extra=("quantity", (2, 3, 4, 5)),
                          decoys=("get_standing_order", "cancel_standing_order", "list_deliveries"), things=("coffee beans order", "paper goods order", "cleaning supplies order")),
    "hall_rental": dict(org="Oak Street Community Center", person="renter", item="booking", site="hall", search="find_renter", list_items="list_renter_bookings",
                        list_sites="list_halls", check="check_hall_availability", action="move_booking", act_verb="move", extra=("new_date", ("2026-12-05", "2026-12-12", "2026-12-19")),
                        decoys=("get_booking", "cancel_booking", "list_deposits"), things=("birthday party booking", "meeting booking", "workshop booking")),
}
DEV_DOMAINS = ("marina", "school", "hall_rental")
FLOWS = ("act_after_verify", "status_then_act", "not_eligible", "similar_names", "site_named_early", "already_verified_second_request")


def tool(name, desc, props, required):
    return make_tool({"name": name, "description": desc, "parameters": {"type": "object", "properties": props, "required": required}})


class Conv:
    def __init__(self, rng, dkey, flow, number):
        self.rng, self.d, self.dkey, self.flow = rng, DOMAINS[dkey], dkey, flow
        d = self.d
        last = rng.choice([x for x in LAST if x in SIMILAR] if flow == "similar_names" else LAST)
        self.name = f"{rng.choice(FIRST)} {last}"
        y, m, day = rng.randrange(1950, 2004), rng.randrange(1, 13), rng.randrange(1, 28)
        self.dob = f"{y}-{m:02d}-{day:02d}"
        self.dob_spoken = rng.choice((f"{MONTHS[m - 1]} {day}, {y}", self.dob, f"{day} {MONTHS[m - 1]} {y}"))
        self.pid = f"{d['person'][:1].upper()}-{rng.randrange(100, 999)}"
        self.items = []
        for k, thing in enumerate(rng.sample(d["things"], min(len(d["things"]), rng.choice((1, 2, 3))))):
            self.items.append({"id": f"{d['item'][:2].upper()}-{rng.randrange(1000, 9999)}", "name": thing, "status": "active"})
        self.target = rng.choice(self.items)
        self.sites = [{"id": f"S{rng.randrange(10, 99)}", "name": n} for n in rng.sample(("North", "Central", "Harbor", "East", "Westside", "Old Town"), 3)]
        self.site = rng.choice(self.sites)
        self.extra = rng.choice(d["extra"][1])
        self.eligible = flow != "not_eligible"
        self.msgs = [{"role": "system", "content": self.policy()}]

    def policy(self):
        d, rng = self.d, self.rng
        rules = [f"Identity: before sharing any {d['item']} information or making a change, verify the {d['person']} by full name AND date of birth using {d['search']}. "
                 f"If two {d['person']}s have similar names, use the date of birth to pick the right one. If either is missing, ask for it.",
                 f"Look up the {d['person']}'s {d['item']}s with {d['list_items']} to get the right id; never guess ids.",
                 f"Before any request to {d['act_verb']} a {d['item']}, call {d['check']} for that {d['item']}. Only if it says eligible may you call {d['action']}. "
                 f"If it is not eligible, explain the reason and do not call {d['action']}.",
                 f"Resolve a {d['site']} from its name with {d['list_sites']} and use its id.",
                 "After a change, tell the person what was done using the details returned by the tool."]
        extra = [f"Do not discuss other people's {d['item']}s.", "Keep answers short and factual.", f"Office hours are 9:00-17:00; changes are processed immediately."]
        rules += rng.sample(extra, 2)
        head = f"You are the {d['person']} support assistant for {d['org']}. You help verified {d['person']}s with their {d['item']}s.\n\nCurrent time: {rng.choice(('Monday', 'Wednesday', 'Friday'))} 2026-11-{rng.randrange(2, 9):02d} 10:{rng.randrange(10, 59)}\n\nPolicy:\n"
        return head + "\n".join(f"{i + 1}. {r}" for i, r in enumerate(rules))

    def tools(self):
        d = self.d
        ts = [tool(d["search"], f"Search {d['person']}s by name and date of birth.", {"name": {"type": "string"}, "date_of_birth": {"type": "string", "description": "YYYY-MM-DD"}}, ["name", "date_of_birth"]),
              tool(d["list_items"], f"List a {d['person']}'s {d['item']}s.", {"person_id": {"type": "string"}, "status": {"type": "string", "enum": ["active", "all"]}}, ["person_id"]),
              tool(d["list_sites"], f"List every {d['site']} with its id.", {}, []),
              tool(d["check"], f"Check whether a {d['item']} may be changed now.", {"item_id": {"type": "string"}}, ["item_id"]),
              tool(d["action"], f"{d['act_verb'].capitalize()} a {d['item']}.", {"item_id": {"type": "string"}, "site_id": {"type": "string"},
                                                                                  d["extra"][0]: {"type": "integer" if isinstance(self.extra, int) else "string"}}, ["item_id", "site_id", d["extra"][0]])]
        for name in d["decoys"]:
            ts.append(tool(name, f"{name.replace('_', ' ').capitalize()}.", {"item_id": {"type": "string"}}, ["item_id"]))
        self.rng.shuffle(ts)
        return ts

    def user(self, text):
        self.msgs.append({"role": "user", "content": text})

    def say(self, text):
        self.msgs.append({"role": "assistant", "content": text})

    def call(self, name, args, result):
        self.msgs.append(call_message([(name, args)]))
        self.msgs.append(tool_message(name, result))

    def run(self):
        d, rng, f = self.d, self.rng, self.flow
        req = f"I'd like to {d['act_verb']} my {self.target['name']}" + (f" at the {self.site['name']} {d['site']}" if f == "site_named_early" else "")
        ex_k, ex_v = d["extra"]
        ex_text = f"{ex_k.replace('_', ' ')} {self.extra}"
        if f == "status_then_act":
            self.user(f"Hi, can you tell me which {d['item']}s I have with you?")
        else:
            self.user(f"Hello. {req}.")
        self.say(f"I can help with that. First, please give me your full name and date of birth.")
        self.user(f"Sure, I'm {self.name}, born {self.dob_spoken}.")
        people = [{"person_id": self.pid, "name": self.name, "date_of_birth": self.dob}]
        if f == "similar_names":
            first, last = self.name.split()
            people.append({"person_id": f"{self.pid[0]}-{rng.randrange(100, 999)}", "name": f"{first} {SIMILAR[last]}", "date_of_birth": f"{int(self.dob[:4]) + rng.choice((-3, 2, 5))}{self.dob[4:]}"})
            rng.shuffle(people)
        self.call(d["search"], {"name": self.name, "date_of_birth": self.dob}, {"results": people, "count": len(people)})
        items = [{"item_id": it["id"], "name": it["name"], "status": it["status"]} for it in self.items]
        self.call(d["list_items"], {"person_id": self.pid}, {"person_id": self.pid, "results": items, "count": len(items)})
        if f == "status_then_act":
            self.say("You're verified. Your " + d["item"] + "s: " + "; ".join(f"{it['name']} ({it['id']}, {it['status']})" for it in self.items) + ".")
            self.user(f"Thanks. {req} at the {self.site['name']} {d['site']}, {ex_text} please.")
        elif f == "already_verified_second_request":
            self.say(f"Thanks, you're verified. Which {d['site']} should I use, and what {ex_k.replace('_', ' ')} would you like?")
            self.user(f"The {self.site['name']} one, {ex_text}.")
        else:
            self.say(f"Thanks, you're verified. Which {d['site']} should I use, and what {ex_k.replace('_', ' ')} would you like?") if f != "site_named_early" else \
                self.say(f"Thanks, you're verified. What {ex_k.replace('_', ' ')} would you like?")
            self.user(f"{'The ' + self.site['name'] + ' one, ' if f != 'site_named_early' else ''}{ex_text}.")
        # after the user's message: lookups the policy requires, then the check, then the action
        self.call(d["list_sites"], {}, {"results": [{"site_id": s["id"], "name": s["name"]} for s in self.sites], "count": len(self.sites)})
        reason = rng.choice(("it was changed less than 30 days ago", "there is an unpaid balance on the account", "the limit for this period has been reached"))
        self.call(d["check"], {"item_id": self.target["id"]}, {"item_id": self.target["id"], "eligible": self.eligible, "reason": "ok" if self.eligible else reason})
        if not self.eligible:
            self.say(f"I checked your {self.target['name']} ({self.target['id']}): it can't be changed right now because {reason}. Nothing was changed.")
            return
        conf = f"C-{rng.randrange(10000, 99999)}"
        args = {"item_id": self.target["id"], "site_id": self.site["id"], ex_k: self.extra}
        self.call(d["action"], args, {"status": "done", "confirmation": conf, **args})
        self.say(f"Done: your {self.target['name']} ({self.target['id']}) was {d['act_verb']}{'d' if d['act_verb'].endswith('e') else 'ed'} at the {self.site['name']} {d['site']} "
                 f"with {ex_text}. Confirmation {conf}.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=("train", "dev"), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--conversations", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    domains = [k for k in DOMAINS if (k in DEV_DOMAINS) == (a.split == "dev")]
    with open(a.out, "x", encoding="utf-8") as out:
        for n in range(a.conversations):
            dk, fl = rng.choice(domains), rng.choice(FLOWS)
            c = Conv(rng, dk, fl, n)
            c.run()
            tools = c.tools()
            p = first_problem(tools, c.msgs)
            if p:
                sys.exit(f"{dk}/{fl}/{n}: {p}")
            out.write(json.dumps({"id": f"e27{a.split[0]}_{dk}_{fl}_{n}", "source": "e27_policy", "family": f"e27_{dk}", "task": "part1",
                                  "tools": tools, "messages": c.msgs}, ensure_ascii=False) + "\n")
    print(a.split, a.conversations, "conversations ->", a.out)


if __name__ == "__main__":
    main()

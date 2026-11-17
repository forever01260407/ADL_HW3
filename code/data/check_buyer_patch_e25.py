"""Check E25 buyer files (patch training data and the new evaluation sets) from the rendered messages.

    python code/data/check_buyer_patch_e25.py FILE [--self-test] [--report OUT.json]

Runs check_buyer.py's rules (official order_ok_hw plus Decimal arithmetic, the must-order cases,
text faults, structure, gold echo through the official scorer, and --self-test mutations) with
one change: the final-price wording list also covers the phrasings E25 adds, so a final offer in
new words is recognised as final. The list is written here from what a seller could plausibly
say, not imported from the generators. It also prints the mix of target steps (steps listed in a
record's `no_target` are left out), split as the plan asks: legal order / illegal-terms continue or
leave / legal-but-keep-negotiating, and the flow (family) counts.
"""
import argparse
import collections
import json
import re
import sys

sys.path.insert(0, "code/data")

import check_buyer as C  # noqa: E402

C.FINAL_WORDS = re.compile(
    r"price is final|any lower|drop it further|last offer|no more room|lowest I'll go|no more discounts|as low as I can go|"
    r"won't go below|bottom line|can't shave anything|final price|final for me|final offer",
    re.I,
)


def mix(records):
    out = collections.Counter()
    for r in records:
        skip = set(r.get("no_target") or ())
        for step in C.steps(r):
            if step["index"] in skip:
                continue
            case = C.case_of(step)
            name = step["calls"][0]["function"]["name"] if step["calls"] else "text_only"
            if name == "order":
                out["legal_order"] += 1
            elif case.startswith("legal"):
                out["legal_keep_negotiating"] += 1
            else:
                out["illegal_or_no_offer_continue_or_leave"] += 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--report")
    a = ap.parse_args()
    records = [json.loads(line) for line in open(a.file, encoding="utf-8")]
    m = mix(records)
    total = sum(m.values())
    flows = collections.Counter(r.get("family") for r in records)
    no_target_ok = all(r.get("no_target") in (None, [next(i for i, x in enumerate(r["messages"]) if x["role"] == "assistant")]) for r in records)
    print("target steps", total, {k: f"{v} ({100 * v / total:.1f}%)" for k, v in sorted(m.items())})
    print("families", dict(sorted(flows.items())), "no_target only the opening:", no_target_ok)
    if a.report:
        json.dump({"file": a.file, "target_steps": total, "mix": dict(m), "families": dict(flows), "no_target_only_opening": no_target_ok}, open(a.report, "x"), indent=1)
    sys.argv = [sys.argv[0], a.file] + (["--self-test"] if a.self_test else [])
    if not no_target_ok:
        sys.exit("no_target lists more than the opening step")
    C.main()


if __name__ == "__main__":
    main()

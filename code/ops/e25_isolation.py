"""E25 data isolation (plan section 5 / C3): public check once, whole-family exclusion, same-source check.

    python -I code/ops/e25_isolation.py --raw DIR --frozen DIR --report OUT.json

DIR holds buyer_patch_train.jsonl, buyer_select_new.jsonl, buyer_confirm_new.jsonl. Steps:
1. code/data/dedupe_against_public.py on all three (public text stays inside that process; only
   our ids come back). Any collided record removes its whole family (flow) from that file.
2. Family sets of the three files must be pairwise disjoint.
3. Same-source check: every user message of select/confirm against every user message of the patch
   (and confirm against select), exact match or >= 0.6 of 5-word shingles, after removing the fixed
   engine lines; seller wording that is pure format ("Offer: ... per unit, qty 1.") is part of the
   interface and is removed too. Hits remove the family from the evaluation file.
4. Minimum sizes (plan C1): patch >= 240 target steps and 6 families; select >= 180 negotiations and
   6 families; confirm >= 80 and 3 families. Failing any -> exit 1 and nothing is frozen.
Nothing is reworded or regenerated after the public check.
"""
import argparse
import collections
import hashlib
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, "code/data")
import dedupe_against_public as D  # noqa: E402

FILES = ("buyer_patch_train", "buyer_select_new", "buyer_confirm_new")
OFFER_LINE = re.compile(r"^Offer: \S+ at \$\d+\.\d{2} per unit, qty \d+\.\s*", re.M)
MIN = {"buyer_patch_train": (240, 6), "buyer_select_new": (180, 6), "buyer_confirm_new": (80, 3)}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def read(p):
    return [json.loads(line) for line in open(p, encoding="utf-8")]


def texts(record):
    out = []
    for t in D.user_texts(record["messages"]):
        t = OFFER_LINE.sub("", t).strip()
        if t:
            out.append(t)
    return out


def index(records):
    exact, grams = set(), set()
    for r in records:
        for t in texts(r):
            tok = D.words(t)
            exact.add(hash(tuple(tok)))
            grams |= D.shingles(tok)
    return exact, grams


def hits(records, exact, grams):
    bad = set()
    for r in records:
        for t in texts(r):
            tok = D.words(t)
            mine = D.shingles(tok)
            if hash(tuple(tok)) in exact or (mine and len(mine & grams) / len(mine) >= D.NEAR):
                bad.add(r["id"])
                break
    return bad


def target_steps(r):
    skip = set(r.get("no_target") or ())
    return sum(1 for i, m in enumerate(r["messages"]) if m["role"] == "assistant" and i and i not in skip)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--offer-frame-as-format", action="store_true", help="E26: pass to dedupe_against_public.py")
    ap.add_argument("--patch-min-families", type=int, default=6,
                    help="user-approved reduction 2026-11-09 ~13:00: 4 (whole-family exclusion left 4 patch flows); recorded in the session")
    a = ap.parse_args()
    MIN["buyer_patch_train"] = (240, a.patch_min_families)
    data = {n: read(f"{a.raw}/{n}.jsonl") for n in FILES}
    pub = f"{a.raw}/collisions_public.json"
    subprocess.run([sys.executable, "-I", "code/data/dedupe_against_public.py", "--out", pub] + (["--offer-frame-as-format"] if a.offer_frame_as_format else [])
                   + [f"{a.raw}/{n}.jsonl" for n in FILES], check=True)
    public_drop = set(json.load(open(pub))["drop_ids"])
    fam = {n: {r["id"]: r["family"] for r in rs} for n, rs in data.items()}
    report = {"inputs": {n: sha(f"{a.raw}/{n}.jsonl") for n in FILES}, "public_report_sha256": sha(pub), "public_hits": {}, "families_dropped": {}}
    dropped = collections.defaultdict(set)
    for n in FILES:
        h = sorted(i for i in fam[n] if i in public_drop)
        report["public_hits"][n] = h
        dropped[n] |= {fam[n][i] for i in h}
    fsets = {n: set(fam[n].values()) for n in FILES}
    report["families_disjoint"] = all(not (fsets[x] & fsets[y]) for x in FILES for y in FILES if x < y)
    ex, gr = index(data["buyer_patch_train"])
    same = {"buyer_select_new": hits(data["buyer_select_new"], ex, gr), "buyer_confirm_new": hits(data["buyer_confirm_new"], ex, gr)}
    ex2, gr2 = index(data["buyer_select_new"])
    same["buyer_confirm_new"] |= hits(data["buyer_confirm_new"], ex2, gr2)
    report["same_source_hits"] = {n: sorted(v) for n, v in same.items()}
    for n, ids in same.items():
        dropped[n] |= {fam[n][i] for i in ids}
    report["families_dropped"] = {n: sorted(dropped[n]) for n in FILES}
    os.makedirs(a.frozen, exist_ok=True)
    ok = report["families_disjoint"]
    kept = {}
    for n in FILES:
        kept[n] = [r for r in data[n] if r["family"] not in dropped[n]]
        fams = {r["family"] for r in kept[n]}
        size = sum(target_steps(r) for r in kept[n]) if n == "buyer_patch_train" else len(kept[n])
        need, need_f = MIN[n]
        report[n] = {"kept_records": len(kept[n]), "size_measure": size, "families": len(fams), "min": [need, need_f], "passes": size >= need and len(fams) >= need_f}
        ok = ok and report[n]["passes"]
    report["passed"] = ok
    if ok:
        for n in FILES:
            with open(f"{a.frozen}/{n}.jsonl", "x", encoding="utf-8") as f:
                for r in kept[n]:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            report[n]["frozen_sha256"] = sha(f"{a.frozen}/{n}.jsonl")
    json.dump(report, open(a.report, "x"), indent=1)
    print(json.dumps({k: v for k, v in report.items() if k not in ("inputs",)}, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

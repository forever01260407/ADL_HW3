"""Drop every family that has a public-test collision (whole-family exclusion).

    python code/data/drop_hit_families.py COLLISIONS.json IN.jsonl OUT.jsonl

COLLISIONS.json is the output of code/data/dedupe_against_public.py (its "drop_ids" are our own record ids;
no public text is stored). Any family with at least one hit is removed as a whole, so no near-variant of a
colliding conversation stays in training. Used for the E27 verify chains (the e27_parking family).
"""
import json
import sys

col, src, dst = sys.argv[1:4]
drop_ids = set(json.load(open(col, encoding="utf-8"))["drop_ids"])
recs = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
bad = {r["family"] for r in recs if r["id"] in drop_ids}
kept = [r for r in recs if r["family"] not in bad]
with open(dst, "x", encoding="utf-8") as f:
    for r in kept:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(json.dumps({"families_dropped": sorted(bad), "kept": len(kept), "removed": len(recs) - len(kept)}))

"""E29 training pool and schedule (user decision ~20:30: one more run aimed at Part 1 70%).

    python code/ops/e29_build.py pool --base BASE_POOL --codex CODEX_TRAIN --v5 DATA_V5 --out POOL
    python code/ops/e29_build.py schedule --pool POOL --updates N --out-dir DIR

pool: BASE_POOL (code/ops/e27_build.py pool output: data_v5 with first-step text targets, the E27 verify chains,
  data_v5 Part 2) plus the Codex natural-language Part 1 records (source codex_p1_stateful_natural_v2,
  records/CODEX_P1_NATURAL_V2_READY_20261109.md). Each Codex conversation gets 2-10 distractor tools, about half
  from the other Codex domains (same verbs, other item family: near-miss names) and half from real data_v5 tool
  schemas (varied names), shuffled with the conversation's own 8 tools (slide 30: add unrelated tools so the model
  has to pick). Labels are unchanged: the policy names the tools of the conversation's own family.
schedule: positions = 16 x updates, drawn without repeats by target bucket (fractions of all positions):
  part2 0.339 (data_v5 Part 2), Codex 0.20 (75% call-after-tool / 25% text-after-tool targets),
  first-step text targets 0.15, E27 verify chains 0.10 (non-first-step targets), other Part 1 = the rest.
  The report counts, over all Part 1 positions, the targets after a tool result that call vs reply in text
  (the lever: a text reply where a call is due costs 10x, an extra call where text is due costs 1x). Seed 4901.
"""
import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, "code")
sys.path.insert(0, "code/data")
CX_SOURCE = "codex_p1_stateful_natural_v2"


def pool(a):
    rng = random.Random(4900)
    cx = [json.loads(l) for l in open(a.codex, encoding="utf-8") if l.strip()]
    assert all(r["source"] == CX_SOURCE for r in cx)
    by_dom = collections.defaultdict(dict)  # family prefix (item family) -> {tool name: tool}
    for r in cx:
        dom = r["family"].rsplit("_", 1)[0]
        for t in r["tools"]:
            by_dom[dom][t["function"]["name"]] = t
    v5_tools = {}
    for line in open(a.v5, encoding="utf-8"):
        r = json.loads(line)
        if r.get("task") != "part1":
            continue
        for t in r.get("tools") or []:
            n = t.get("function", {}).get("name")
            if n and n not in v5_tools:
                v5_tools[n] = t
    v5_names = sorted(v5_tools)
    stats = collections.Counter()
    with open(a.out, "x", encoding="utf-8") as out:
        for line in open(a.base, encoding="utf-8"):
            out.write(line if line.endswith("\n") else line + "\n")
            stats["base_records"] += 1
        for r in cx:
            own = {t["function"]["name"] for t in r["tools"]}
            dom = r["family"].rsplit("_", 1)[0]
            k = rng.randint(2, 10)
            n_near = k // 2
            near_pool = sorted(n for d, ts in by_dom.items() if d != dom for n in ts if n not in own)
            near = rng.sample(near_pool, n_near)
            far = [n for n in rng.sample(v5_names, k - n_near + 5) if n not in own and n not in near][: k - n_near]
            extra = [by_dom[next(d for d in by_dom if n in by_dom[d])][n] for n in near] + [v5_tools[n] for n in far]
            tools = list(r["tools"]) + extra
            names = [t["function"]["name"] for t in tools]
            assert len(names) == len(set(names)), r["id"]
            rng.shuffle(tools)
            r = dict(r, tools=tools)
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            stats["codex_records"] += 1
            stats[f"codex_tools_{len(tools)}"] += 1
    print(json.dumps(dict(sorted(stats.items())), indent=1))


def bucket_of_targets(pool_path):
    """step id -> (bucket, kind) where kind is call_after_tool / text_after_tool / call_after_user / text_after_user."""
    from common import expand_steps
    out = {}
    for line in open(pool_path, encoding="utf-8"):
        r = json.loads(line)
        for st in expand_steps(r):
            m = st["messages"]
            prev, tgt = m[-2]["role"], m[-1]
            kind = ("call" if tgt.get("tool_calls") else "text") + "_after_" + prev
            if st["task"] == "part2":
                b = "part2"
            elif r["source"] == CX_SOURCE:
                b = "cx_call" if kind == "call_after_tool" else ("cx_text" if kind == "text_after_tool" else "cx_other")
            elif "~fs" in st["id"]:
                b = "first_step"
            elif st["id"].startswith("e27t_"):
                b = "verify"
            else:
                b = "part1"
            out[st["id"]] = (b, kind)
    return out


def schedule(a):
    import model_setup as M
    import train as T
    tok = M.F.tokenizer()
    examples, too_long = T.prepare({"train_file": a.pool, "max_examples": None, "seed": 42, "max_length": 8192}, tok)
    info = bucket_of_targets(a.pool)
    by = collections.defaultdict(list)
    for e in examples:
        by[info[e["id"]][0]].append(e["id"])
    total = a.updates * 16
    q = {"part2": round(total * 0.339), "cx_call": round(total * 0.20 * 0.75), "cx_text": round(total * 0.20 * 0.25),
         "first_step": round(total * 0.15), "verify": round(total * 0.10)}
    q["part1"] = total - sum(q.values())
    rng = random.Random(4901)
    ids = []
    for b in ("part2", "cx_call", "cx_text", "first_step", "verify", "part1"):
        ids += rng.sample(sorted(by[b]), q[b])
    rng.shuffle(ids)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(f"{a.out_dir}/E27_schedule.json", "x") as f:  # name kept: the E27 controller and trainer read this file
        json.dump(ids, f)
    kinds = collections.Counter(info[i][1] for i in ids if info[i][0] != "part2")
    after_tool = kinds["call_after_tool"] + kinds["text_after_tool"]
    rep = {"pool": a.pool, "pool_sha256": M.file_hash(a.pool), "examples": len(examples), "too_long_dropped": too_long,
           "available": {k: len(v) for k, v in sorted(by.items())}, "positions": {"total": total, **q},
           "part1_target_kinds": dict(kinds), "part1_after_tool_call_share": round(kinds["call_after_tool"] / after_tool, 3) if after_tool else None,
           "checks": {"no_repeat": len(set(ids)) == total, "length": len(ids) == total}, "schedule_sha256": M.file_hash(f"{a.out_dir}/E27_schedule.json")}
    json.dump(rep, open(f"{a.out_dir}/schedule_report.json", "x"), indent=1)
    print(json.dumps(rep, indent=1))
    sys.exit(0 if all(rep["checks"].values()) else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pool")
    p.add_argument("--base", required=True)
    p.add_argument("--codex", required=True)
    p.add_argument("--v5", required=True)
    p.add_argument("--out", required=True)
    s = sub.add_parser("schedule")
    s.add_argument("--pool", required=True)
    s.add_argument("--updates", type=int, required=True)
    s.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    {"pool": pool, "schedule": schedule}[a.cmd](a)


if __name__ == "__main__":
    main()

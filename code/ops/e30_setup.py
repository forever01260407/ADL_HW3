"""E30 (user decision ~21:40): continue E28 with the GPT-written long-policy data (gpt_policy_v1), fast path.

    python code/ops/e30_setup.py --gpt-train FILE [--updates 150]

Why: on public, half of the weight is conversations with a long policy (system >= 2,000 chars), where every model so
far gets only ~36% of the must-call steps (no-policy conversations: ~76%). Template data (E27 verify chains, E29
natural_v2r1) learned its own generator (98-99%) without moving that number. gpt_policy_v1 is written by an LLM for
diversity (records/GPT_POLICY_V1_BRIEF_20261109.md); only steps right after a tool result are targets.
From E28 step_250 (already has the first-step text behaviour), lr 3e-5, U updates (16 per update), drawn without
repeats by bucket: Part 2 0.339 (data_v5, keeps the buyer), gpt_policy_v1 0.35, first-step text 0.08 and E27 verify
0.05 (replay of what E28 learned), other data_v5 Part 1 = the rest. Seed 5001.
Fast path because of the 23:59 deadline: no own-set evaluation and no judge; code/ops/e30_run.sh trains, then runs
the official generation and scoring on both public files (rule-based buyer metrics only). The user decides.
"""
import argparse
import collections
import datetime as dt
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time

sys.path.insert(0, "code")
sys.path.insert(0, "code/data")
E28 = "runs/E28_session_20261109_182859"
D27 = "data_work/processed/e27"
GPT_SOURCE = "gpt_policy_v1"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument("--gpt-train", required=True, nargs="+", help="one or more checked GPT files")
ap.add_argument("--updates", type=int, default=150)
ap.add_argument("--gpt-share", type=float, default=0.35)
ap.add_argument("--d-gpu", default="23:40")
ap.add_argument("--source", default="gpt_policy_v1", help="comma-separated source fields of the GPT records (v2: gpt_policy_v2)")
a = ap.parse_args()
GPT_SOURCES = set(a.source.split(","))
tz = dt.timezone(dt.timedelta(hours=8))
T0 = int(time.time())
name = "E30_session_" + dt.datetime.fromtimestamp(T0, tz).strftime("%Y%m%d_%H%M%S")
S, F = f"runs/{name}", f"data_work/frozen/e30_{name}"
os.makedirs(S)
os.makedirs(F)
hh, mm = (int(x) for x in a.d_gpu.split(":"))
at = lambda h, m: int(dt.datetime(2026, 11, 9, h, m, tzinfo=tz).timestamp())  # noqa: E731
fmt = lambda e: {"epoch": e, "taipei": dt.datetime.fromtimestamp(e, tz).strftime("%Y-%m-%d %H:%M:%S")}  # noqa: E731
base_adapter = f"{E28}/E27/checkpoints/step_250"
json.dump({"session": name, "T0": fmt(T0), "deadlines": {"D_train": fmt(at(hh, mm)), "D_gpu": fmt(at(hh, mm)), "official_deadline": fmt(at(23, 59) + 59)},
           "base": "E28_250", "user_decisions": {"goal": "Part 1 70%", "data": "gpt_policy_v1 (LLM-written long policies)", "fast_path": "no own eval, no judge; user decides"}},
          open(f"{S}/session.json", "x"), indent=1)
gpt = [json.loads(l) for path in a.gpt_train for l in open(path, encoding="utf-8") if l.strip()]
assert gpt and all(r.get("source") in GPT_SOURCES and r.get("task") == "part1" for r in gpt), f"not {GPT_SOURCES} part1 records"
assert len({r["id"] for r in gpt}) == len(gpt), "duplicate ids across the GPT files"
with open(f"{F}/gpt_train.jsonl", "x", encoding="utf-8") as f:
    for r in gpt:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
shutil.copyfile(f"{D27}/verify_train_isolated.jsonl", f"{F}/verify_train.jsonl")
PY = sys.executable
subprocess.run([PY, "code/ops/e27_build.py", "pool", "--v5", "data_work/frozen/data_v5/train.jsonl", "--verify", f"{F}/verify_train.jsonl", "--out", f"{F}/base_pool.jsonl"], check=True)
with open(f"{F}/train_pool.jsonl", "x", encoding="utf-8") as out:
    for line in open(f"{F}/base_pool.jsonl", encoding="utf-8"):
        out.write(line)
    for r in gpt:
        out.write(json.dumps(r, ensure_ascii=False) + "\n")
# schedule
import model_setup as M  # noqa: E402
import train as T  # noqa: E402
from common import expand_steps  # noqa: E402

tok = M.F.tokenizer()
examples, too_long = T.prepare({"train_file": f"{F}/train_pool.jsonl", "max_examples": None, "seed": 42, "max_length": 8192}, tok)
info = {}
for line in open(f"{F}/train_pool.jsonl", encoding="utf-8"):
    r = json.loads(line)
    for st in expand_steps(r):
        m = st["messages"]
        kind = ("call" if m[-1].get("tool_calls") else "text") + "_after_" + m[-2]["role"]
        b = ("part2" if st["task"] == "part2" else "gpt" if r["source"] in GPT_SOURCES else "first_step" if "~fs" in st["id"]
             else "verify" if st["id"].startswith("e27t_") else "part1")
        info[st["id"]] = (b, kind)
by = collections.defaultdict(list)
for e in examples:
    by[info[e["id"]][0]].append(e["id"])
total = a.updates * 16
q = {"part2": round(total * 0.339), "gpt": min(len(by["gpt"]), round(total * a.gpt_share)), "first_step": round(total * 0.08), "verify": round(total * 0.05)}
q["part1"] = total - sum(q.values())
rng = random.Random(5001)
ids = []
for b in ("part2", "gpt", "first_step", "verify", "part1"):
    ids += rng.sample(sorted(by[b]), q[b])
rng.shuffle(ids)
os.makedirs(f"{S}/plan")
json.dump(ids, open(f"{S}/plan/E30_schedule.json", "x"))
kinds = collections.Counter(info[i][1] for i in ids if info[i][0] != "part2")
rep = {"examples": len(examples), "too_long_dropped": too_long, "available": {k: len(v) for k, v in sorted(by.items())}, "positions": {"total": total, **q},
       "part1_target_kinds": dict(kinds), "gpt_target_kinds": dict(collections.Counter(info[i][1] for i in by["gpt"])),
       "checks": {"no_repeat": len(set(ids)) == total, "length": len(ids) == total}}
json.dump(rep, open(f"{S}/plan/schedule_report.json", "x"), indent=1)
print(json.dumps(rep, indent=1))
assert all(rep["checks"].values())
cfg = json.load(open(f"{E28}/E27_config.json"))
cfg.update({"train_file": f"{F}/train_pool.jsonl", "learning_rate": 3e-05, "max_updates": a.updates, "save_at": [a.updates], "schedule_file": f"{S}/plan/E30_schedule.json"})
json.dump(cfg, open(f"{S}/E30_config.json", "x"), indent=1)
json.dump({"base_adapter": base_adapter, "base_sha256": sha(f"{base_adapter}/adapter_model.safetensors"), "gpt_train_sha256": sha(f"{F}/gpt_train.jsonl"),
           "pool_sha256": sha(f"{F}/train_pool.jsonl"), "schedule_sha256": sha(f"{S}/plan/E30_schedule.json"), "config_sha256": sha(f"{S}/E30_config.json")},
          open(f"{S}/plan/hashes.json", "x"), indent=1)
print(S)

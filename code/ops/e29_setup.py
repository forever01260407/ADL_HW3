"""Create and freeze the E29 session (user decision ~20:30: one more run for Part 1 70%, GPU until 22:00).

    python code/ops/e29_setup.py [--updates 250]

Recipe: from E23; E28's data (data_v5 Part 1 with first-step text targets, E27 verify chains, data_v5 Part 2) plus the
Codex natural-language Part 1 data (natural_v2r1) with distractor tools, mixed by target bucket in code/ops/e29_build.py
(Codex 20% of positions, 75% call-after-tool / 25% text-after-tool; first-step text targets 15% of all positions).
This changes several things at once (new data, first-step share, mix): it tests the recipe, not one factor.
Selection: the E27 controller and rules, except that the buyer guards on our own buyer set are off (E27/E28 showed it
does not track public: E23 fills 48% there and 92% on public); the full public run is made for any checkpoint that
passes integrity and the Part 1 gain, and the user chooses among E23 / E28 / E29 from the public numbers.
The E27 controller is reused unchanged, so inside the session the model folder is "E27" and the tag "E27_<step>".
Deadlines: D_train 21:15, D_select 21:25, D_gpu 22:00.
"""
import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

E27 = "runs/E27_session_20261109_165030"
D = "data_work/processed/e27"
CX = "data_work/processed/codex_p1_next_20261109_natural_v2r1"
CX_SHA = {"train.jsonl": "be03b9811173f796cd745626f04f0ddedf2d89caf4c6b8d450e9e4d363bdf45d",
          "stress.jsonl": "8f0f0754b7ddbc6b4270c6ca1fcd5018bf13122e6ea6b40da77f9c0a0804d8b0",
          "challenge.jsonl": "34c4645bdd526dee2639ee214496af41a6f23232da45e48f181b593d1bc2d82c"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument("--updates", type=int, default=250)
a = ap.parse_args()
for n, h in CX_SHA.items():
    assert sha(f"{CX}/{n}") == h, f"{n}: not the natural_v2r1 file of the handoff"
tz = dt.timezone(dt.timedelta(hours=8))
T0 = int(time.time())
name = "E29_session_" + dt.datetime.fromtimestamp(T0, tz).strftime("%Y%m%d_%H%M%S")
S = f"runs/{name}"
F = f"data_work/frozen/e29_{name}"
os.makedirs(S)
os.makedirs(F)
at = lambda h, m: int(dt.datetime(2026, 11, 9, h, m, tzinfo=tz).timestamp())  # noqa: E731
fmt = lambda e: {"epoch": e, "taipei": dt.datetime.fromtimestamp(e, tz).strftime("%Y-%m-%d %H:%M:%S")}  # noqa: E731
base_adapter = "runs/E23/checkpoints/step_1475"
base_public = {"p1": 60.5337, "fill_rate": 0.91983, "order_accuracy": 0.97075, "relative_price": 1.02675, "text_accuracy": 0.99133, "source": "runs/E23/full_1475"}
prot = sorted(glob.glob("runs/E2[1356]*/checkpoints/step_*/adapter_model.safetensors") + glob.glob("runs/E2*_session_*/C[01]/checkpoints/*/adapter_model.safetensors")
              + glob.glob("runs/E2[78]_session_*/E27/checkpoints/*/adapter_model.safetensors")
              + glob.glob("release/*/adapter_checkpoint/adapter_model.safetensors") + glob.glob("release/candidates/*/adapter_checkpoint/adapter_model.safetensors"))
session = {"session": name, "T0": fmt(T0), "deadlines": {"D_train": fmt(at(21, 15)), "D_select": fmt(at(21, 25)), "D_gpu": fmt(at(22, 0)), "official_deadline": fmt(at(23, 59) + 59)},
           "base": "E23", "user_decisions": {"goal": "Part 1 70%", "gpu_until": "22:00 (user, ~20:30)",
                                             "data": "E28 data + Codex natural_v2r1 Part 1 (20%, 75/25 call/text after tool) with distractor tools; first-step text 15% of positions",
                                             "selection": "integrity + own Part 1 gain; buyer guards on the own buyer set off; full public for the user's choice among E23/E28/E29"},
           "protected": {p: sha(p) for p in prot}}
json.dump(session, open(f"{S}/session.json", "x"), indent=1)
os.chmod(f"{S}/session.json", 0o444)
shutil.copyfile(f"{D}/verify_train_isolated.jsonl", f"{F}/verify_train.jsonl")
for n in CX_SHA:
    shutil.copyfile(f"{CX}/{n}", f"{F}/cx_{n}")
PY = sys.executable
subprocess.run([PY, "code/ops/e27_build.py", "pool", "--v5", "data_work/frozen/data_v5/train.jsonl", "--verify", f"{F}/verify_train.jsonl", "--out", f"{F}/base_pool.jsonl"], check=True)
subprocess.run([PY, "code/ops/e29_build.py", "pool", "--base", f"{F}/base_pool.jsonl", "--codex", f"{F}/cx_train.jsonl", "--v5", "data_work/frozen/data_v5/train.jsonl",
                "--out", f"{F}/train_pool.jsonl"], check=True)
subprocess.run([PY, "code/ops/e29_build.py", "schedule", "--pool", f"{F}/train_pool.jsonl", "--updates", str(a.updates), "--out-dir", f"{S}/plan"], check=True)
os.makedirs(f"{S}/own_eval/steps")
for n in ("verify_dev", "dev", "dev_new", "dev_bfcl", "buyer_select_new"):
    shutil.copyfile(f"{E27}/own_eval/steps/{n}.jsonl", f"{S}/own_eval/steps/{n}.jsonl")
for n in ("stress", "challenge"):
    subprocess.run([PY, "code/eval_own.py", "export", "--data", f"{F}/cx_{n}.jsonl", "--out", f"{S}/own_eval/steps/cx_{n}.jsonl"], check=True)
os.makedirs(f"{S}/own_eval/E23")
shutil.copyfile(f"{E27}/own_eval/E23/adapter.json", f"{S}/own_eval/E23/adapter.json")
for n in ("verify_dev", "dev", "dev_new", "dev_bfcl", "buyer_select_new"):
    for suf in (".jsonl", ".provenance.json"):
        shutil.copyfile(f"{E27}/own_eval/E23/{n}{suf}", f"{S}/own_eval/E23/{n}{suf}")
cfg = json.load(open(f"{E27}/E27_config.json"))
cfg.update({"train_file": f"{F}/train_pool.jsonl", "max_updates": a.updates, "save_at": [a.updates], "schedule_file": f"{S}/plan/E27_schedule.json"})
json.dump(cfg, open(f"{S}/E27_config.json", "x"), indent=1)
P27 = json.load(open(f"{E27}/plan.json"))
limits = dict(P27["limits"])
limits.update({"train_expect": int(6 * a.updates), "train": int(8 * a.updates)})
sel = dict(P27["e27_select"])
sel.update({"fill_drop_pp": 100.0, "order_acc_drop_pp": 100.0, "text_acc_drop_pp": 100.0,
            "note": "buyer guards on the own buyer set are off (E27/E28: it does not track public); buyer is judged on the full public run"})
plan = {"session": name, "experiment": "E29", "base": {"tag": "E23", "adapter": base_adapter, "public": base_public},
        "eval_sets": P27["eval_sets"] + ["cx_stress", "cx_challenge"], "p1_sets": P27["p1_sets"], "buyer_records": P27["buyer_records"],
        "checkpoints": [a.updates], "e27_select": sel, "e27_adopt": P27["e27_adopt"], "limits": limits, "poll": 15, "commands": {},
        "hashes": {"base_adapter": sha(f"{base_adapter}/adapter_model.safetensors"), "train_pool": sha(f"{F}/train_pool.jsonl"),
                   "schedule": sha(f"{S}/plan/E27_schedule.json"), "config": sha(f"{S}/E27_config.json"), "codex_data": CX_SHA,
                   "code": {p: sha(p) for p in ("code/train.py", "code/ops/phase_e27.py", "code/ops/e27_steps.py", "code/ops/e27_build.py", "code/ops/e29_build.py",
                                                "code/ops/e29_setup.py", "code/ops/e27_score_p1.py", "code/data/gen_p1_verify_e27.py", "code/ops/phase_e25.py",
                                                "code/ops/run_supervised.py", "code/ops/eval_e24.py", "code/ops/e25_steps.py", "code/eval_buyer_full.py")}}}
p = f"{S}/plan.json"
json.dump(plan, open(p, "x"), indent=1, ensure_ascii=False)
open(f"{S}/plan.sha256", "x").write(sha(p) + "  plan.json\n")
os.chmod(p, 0o444)
os.chmod(f"{S}/plan.sha256", 0o444)
print(S, "frozen", sha(p))

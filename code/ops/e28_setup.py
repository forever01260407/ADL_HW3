"""Create and freeze the E28 session: the E27 recipe with E23's own Part 2 data (data_v5) instead of E26's buyer data.

    python code/ops/e28_setup.py [--updates 400 --d-train 19:10 --d-select 19:25]

The first E28 session (runs/E28_session_20261109_180541) used the defaults and was aborted before training at 18:21
(the public evaluation would have ended after 20:00). The second (runs/E28_session_20261109_181944, --updates 250 --d-train 19:05 --d-select 19:15) was refused
by the supervisor before training: it still carried E27's expected training time (2400 s). The third uses
--updates 250 --d-train 19:05 --d-select 19:20 with expected/maximum training time scaled to the updates.

Why (18:05): E27_400 gained +5.08 on the public Part 1 headline (60.53 -> 65.61) but its public fill rate fell
92.0% -> 86.3% (more wait / end_session at steps where a valid order was possible). The E26 acceptance rule did not
transfer from our own buyer set (E23 fills 48% there, 92% on public). E28 keeps everything of E27 on the Part 1 side
(first-step text targets, verify chains, same lr / seed) and puts E23's own Part 2 data back, so the buyer
should stay near E23 while Part 1 keeps the E27 gain. Same selection and adoption rules as E27.
The E27 controller and helpers are reused unchanged (code/ops/phase_e27.py, e27_steps.py), so inside this session the
trained model folder is still called "E27" and its checkpoints "E27_<step>"; they belong to E28.
Deadlines: D_train 19:10, D_select 19:25, D_gpu 20:00 (user's hard stop).
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

E26 = "runs/E26_session_20261109_160454"
E27 = "runs/E27_session_20261109_165030"
D = "data_work/processed/e27"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument("--updates", type=int, default=400)
ap.add_argument("--d-train", default="19:10")
ap.add_argument("--d-select", default="19:25")
a = ap.parse_args()
hm = lambda s: tuple(int(x) for x in s.split(":"))  # noqa: E731
save_at = [300, 400] if a.updates == 400 else [a.updates]
tz = dt.timezone(dt.timedelta(hours=8))
T0 = int(time.time())
name = "E28_session_" + dt.datetime.fromtimestamp(T0, tz).strftime("%Y%m%d_%H%M%S")
S = f"runs/{name}"
F = f"data_work/frozen/e28_{name}"
os.makedirs(S)
os.makedirs(F)
at = lambda h, m: int(dt.datetime(2026, 11, 9, h, m, tzinfo=tz).timestamp())  # noqa: E731
fmt = lambda e: {"epoch": e, "taipei": dt.datetime.fromtimestamp(e, tz).strftime("%Y-%m-%d %H:%M:%S")}  # noqa: E731
base_adapter = "runs/E23/checkpoints/step_1475"
base_public = {"p1": 60.5337, "fill_rate": 0.91983, "order_accuracy": 0.97075, "relative_price": 1.02675, "text_accuracy": 0.99133, "source": "runs/E23/full_1475"}
prot = sorted(glob.glob("runs/E2[1356]*/checkpoints/step_*/adapter_model.safetensors") + glob.glob("runs/E2*_session_*/C[01]/checkpoints/*/adapter_model.safetensors")
              + glob.glob("runs/E27_session_*/E27/checkpoints/*/adapter_model.safetensors")
              + glob.glob("release/*/adapter_checkpoint/adapter_model.safetensors") + glob.glob("release/candidates/*/adapter_checkpoint/adapter_model.safetensors"))
session = {"session": name, "T0": fmt(T0), "deadlines": {"D_train": fmt(at(*hm(a.d_train))), "D_select": fmt(at(*hm(a.d_select))), "D_gpu": fmt(at(20, 0)), "official_deadline": fmt(at(23, 59) + 59)},
           "base": "E23", "user_decisions": {"goal": "Part 1 70%", "first_step_text": "use the official first-step rule: confirm in text after a user message in Part 1",
                                             "data": "E27 Part 1 data unchanged; Part 2 back to E23's own data_v5 (E26 buyer rule lost public fill in E27)",
                                             "selection": "own sets under the official rules pick one checkpoint; one full public decides adoption; base kept otherwise",
                                             "hard_stop": "20:00"},
           "protected": {p: sha(p) for p in prot}}
json.dump(session, open(f"{S}/session.json", "x"), indent=1)
os.chmod(f"{S}/session.json", 0o444)
# data: the E27 verify data, data_v5 for everything else (no --buyer -> data_v5 Part 2)
shutil.copyfile(f"{D}/verify_train_isolated.jsonl", f"{F}/verify_train.jsonl")
shutil.copyfile(f"{D}/verify_dev.jsonl", f"{F}/verify_dev.jsonl")
shutil.copyfile(f"{D}/isolation_e27.json", f"{S}/isolation_e27.json")
subprocess.run([sys.executable, "code/ops/e27_build.py", "pool", "--v5", "data_work/frozen/data_v5/train.jsonl", "--verify", f"{F}/verify_train.jsonl", "--out", f"{F}/train_pool.jsonl"], check=True)
subprocess.run([sys.executable, "code/ops/e27_build.py", "schedule", "--pool", f"{F}/train_pool.jsonl", "--updates", str(a.updates), "--out-dir", f"{S}/plan"], check=True)
# eval steps: identical files to E27's (provenance is by content), and E23's existing predictions on them
os.makedirs(f"{S}/own_eval/steps")
for n in ("verify_dev", "dev", "dev_new", "dev_bfcl", "buyer_select_new"):
    shutil.copyfile(f"{E27}/own_eval/steps/{n}.jsonl", f"{S}/own_eval/steps/{n}.jsonl")
os.makedirs(f"{S}/own_eval/E23")
shutil.copyfile(f"{E27}/own_eval/E23/adapter.json", f"{S}/own_eval/E23/adapter.json")
for n in ("verify_dev", "dev", "dev_new", "dev_bfcl", "buyer_select_new"):
    for suf in (".jsonl", ".provenance.json"):
        shutil.copyfile(f"{E27}/own_eval/E23/{n}{suf}", f"{S}/own_eval/E23/{n}{suf}")
cfg = json.load(open(f"{E27}/E27_config.json"))
cfg.update({"train_file": f"{F}/train_pool.jsonl", "max_updates": a.updates, "save_at": save_at, "schedule_file": f"{S}/plan/E27_schedule.json"})
json.dump(cfg, open(f"{S}/E27_config.json", "x"), indent=1)
P27 = json.load(open(f"{E27}/plan.json"))
limits = dict(P27["limits"])
if a.updates != 400:  # E27 took 1906 s for 400 updates (about 4.8 s each); the second E28 session was refused for copying 2400 s
    limits.update({"train_expect": int(6 * a.updates), "train": int(8 * a.updates)})
plan = {"session": name, "experiment": "E28", "base": {"tag": "E23", "adapter": base_adapter, "public": base_public},
        "eval_sets": P27["eval_sets"], "p1_sets": P27["p1_sets"], "buyer_records": P27["buyer_records"], "checkpoints": save_at,
        "e27_select": P27["e27_select"], "e27_adopt": P27["e27_adopt"], "limits": limits, "poll": 15, "commands": {},
        "hashes": {"base_adapter": sha(f"{base_adapter}/adapter_model.safetensors"), "train_pool": sha(f"{F}/train_pool.jsonl"),
                   "schedule": sha(f"{S}/plan/E27_schedule.json"), "config": sha(f"{S}/E27_config.json"),
                   "code": {p: sha(p) for p in ("code/train.py", "code/ops/phase_e27.py", "code/ops/e27_steps.py", "code/ops/e27_build.py", "code/ops/e27_score_p1.py",
                                                "code/ops/e28_setup.py", "code/data/gen_p1_verify_e27.py", "code/ops/phase_e25.py", "code/ops/run_supervised.py",
                                                "code/ops/eval_e24.py", "code/ops/e25_steps.py", "code/eval_buyer_full.py")}}}
p = f"{S}/plan.json"
json.dump(plan, open(p, "x"), indent=1, ensure_ascii=False)
open(f"{S}/plan.sha256", "x").write(sha(p) + "  plan.json\n")
os.chmod(p, 0o444)
os.chmod(f"{S}/plan.sha256", 0o444)
print(S, "frozen", sha(p))

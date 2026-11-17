"""Create and freeze the E27 session (user decision ~16:40: aim Part 1 at 70%, first-step text, verify chains).

    python code/ops/e27_setup.py --base E23|C1

base = the model E26 ended with (E23, or E26's C1 if it was adopted after its public check). The E27
run starts from it; its Part 2 positions always use the E26 buyer data (the user's third behaviour: order when the
acceptance rule is met), so the public check also tests E26's buyer rule.
Deadlines: D_train 19:00, D_select 19:20, D_gpu 20:00 (user: hard stop 20:00).
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
E26_F = "data_work/frozen/e26_E26_session_20261109_160454"
D = "data_work/processed/e27"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument("--base", choices=("E23", "C1"), required=True)
a = ap.parse_args()
tz = dt.timezone(dt.timedelta(hours=8))
T0 = int(time.time())
name = "E27_session_" + dt.datetime.fromtimestamp(T0, tz).strftime("%Y%m%d_%H%M%S")
S = f"runs/{name}"
F = f"data_work/frozen/e27_{name}"
os.makedirs(S)
os.makedirs(F)
at = lambda h, m: int(dt.datetime(2026, 11, 9, h, m, tzinfo=tz).timestamp())  # noqa: E731
fmt = lambda e: {"epoch": e, "taipei": dt.datetime.fromtimestamp(e, tz).strftime("%Y-%m-%d %H:%M:%S")}  # noqa: E731
base_adapter = "runs/E23/checkpoints/step_1475" if a.base == "E23" else f"{E26}/C1/checkpoints/step_200"
if a.base == "E23":
    base_public = {"p1": 60.5337, "fill_rate": 0.91983, "order_accuracy": 0.97075, "relative_price": 1.02675, "text_accuracy": 0.99133, "source": "runs/E23/full_1475"}
else:
    t = json.load(open(f"{E26}/final_public/metrics/tools.json"))["headline"]["with_first_step_rule"]["all"]["weighted_step_acc"]
    n = json.load(open(f"{E26}/final_public/metrics/negotiation.json"))["metrics"]
    base_public = {"p1": 100 * t, **{k: n[k] for k in ("fill_rate", "order_accuracy", "relative_price", "text_accuracy")}, "source": f"{E26}/final_public"}
prot = sorted(glob.glob("runs/E2[1356]*/checkpoints/step_*/adapter_model.safetensors") + glob.glob("runs/E2*_session_*/C[01]/checkpoints/*/adapter_model.safetensors")
              + glob.glob("release/*/adapter_checkpoint/adapter_model.safetensors") + glob.glob("release/candidates/*/adapter_checkpoint/adapter_model.safetensors"))
session = {"session": name, "T0": fmt(T0), "deadlines": {"D_train": fmt(at(19, 0)), "D_select": fmt(at(19, 20)), "D_gpu": fmt(at(20, 0)), "official_deadline": fmt(at(23, 59) + 59)},
           "base": a.base, "user_decisions": {"goal": "Part 1 70%", "first_step_text": "use the official first-step rule: confirm in text after a user message in Part 1",
                                               "data": "independently synthesized policy/verify chains after reading public task types and E23 error distribution; no copying",
                                               "selection": "own sets under the official rules pick one checkpoint; one full public decides adoption; base kept otherwise",
                                               "hard_stop": "20:00"},
           "protected": {p: sha(p) for p in prot}}
json.dump(session, open(f"{S}/session.json", "x"), indent=1)
os.chmod(f"{S}/session.json", 0o444)
# data
shutil.copyfile(f"{D}/verify_train_isolated.jsonl", f"{F}/verify_train.jsonl")
shutil.copyfile(f"{D}/verify_dev.jsonl", f"{F}/verify_dev.jsonl")
shutil.copyfile(f"{D}/isolation_e27.json", f"{S}/isolation_e27.json")
cmd = [sys.executable, "code/ops/e27_build.py", "pool", "--v5", "data_work/frozen/data_v5/train.jsonl", "--verify", f"{F}/verify_train.jsonl", "--out", f"{F}/train_pool.jsonl"]
cmd += ["--buyer", f"{E26_F}/buyer_patch_train.jsonl"]  # user's behaviour 3 (order when the acceptance rule is met): E26 buyer data, whatever the base
subprocess.run(cmd, check=True)
subprocess.run([sys.executable, "code/ops/e27_build.py", "schedule", "--pool", f"{F}/train_pool.jsonl", "--updates", "400", "--out-dir", f"{S}/plan"], check=True)
# eval steps and the base's existing predictions
os.makedirs(f"{S}/own_eval/steps")
subprocess.run([sys.executable, "code/eval_own.py", "export", "--data", f"{F}/verify_dev.jsonl", "--out", f"{S}/own_eval/steps/verify_dev.jsonl"], check=True)
for n in ("dev", "dev_new", "dev_bfcl", "buyer_select_new"):
    shutil.copyfile(f"{E26}/own_eval/steps/{n}.jsonl", f"{S}/own_eval/steps/{n}.jsonl")
os.makedirs(f"{S}/own_eval/{a.base}")
shutil.copyfile(f"{E26}/own_eval/{a.base}/adapter.json", f"{S}/own_eval/{a.base}/adapter.json")
for n in ("dev", "dev_new", "dev_bfcl", "buyer_select_new"):
    for suf in (".jsonl", ".provenance.json"):
        shutil.copyfile(f"{E26}/own_eval/{a.base}/{n}{suf}", f"{S}/own_eval/{a.base}/{n}{suf}")
cfg = {"train_file": f"{F}/train_pool.jsonl", "max_examples": None, "seed": 42, "r": 32, "alpha": 64, "dropout": 0.05, "learning_rate": 4e-05, "weight_decay": 0.0,
       "warmup_ratio": 0.03, "max_grad_norm": 1.0, "micro_batch": 2, "accumulation": 8, "max_length": 8192, "max_updates": 400, "save_at": [200, 300, 400],
       "part2_share": 0.339, "gradient_checkpointing": True, "print_every": 10, "schedule_file": f"{S}/plan/E27_schedule.json"}
json.dump(cfg, open(f"{S}/E27_config.json", "x"), indent=1)
plan = {"session": name, "base": {"tag": a.base, "adapter": base_adapter, "public": base_public},
        "eval_sets": ["verify_dev", "dev", "dev_new", "dev_bfcl", "buyer_select_new"], "p1_sets": ["verify_dev", "dev", "dev_new", "dev_bfcl"],
        "buyer_records": f"{E26_F}/buyer_select_new.jsonl", "checkpoints": [200, 300, 400],
        "e27_select": {"p1_min_gain": 2.0, "fill_drop_pp": 2.0, "order_acc_drop_pp": 2.0, "text_acc_drop_pp": 1.0,
                       "metric": "mean over p1_sets of the official-rule weighted score (first-step rule; must-call x10)"},
        "e27_adopt": {"p1_min_gain": 1.0, "fill_drop": 0.01, "price_rise": 0.005, "order_acc_drop": 0.01, "text_acc_drop": 0.01},
        "limits": {"wait_gpu": 3600, "eval_total": 2400, "train": 3000, "train_expect": 2400, "train_stall": 600, "train_init": 900,
                   "judge": 2700, "judge_expect": 900, "judge_stall": 900, "judge_init": 1200, "public": 3600, "public_expect": 1900},
        "poll": 15, "commands": {},
        "hashes": {"base_adapter": sha(f"{base_adapter}/adapter_model.safetensors"), "train_pool": sha(f"{F}/train_pool.jsonl"),
                   "schedule": sha(f"{S}/plan/E27_schedule.json"), "config": sha(f"{S}/E27_config.json"),
                   "code": {p: sha(p) for p in ("code/train.py", "code/ops/phase_e27.py", "code/ops/e27_steps.py", "code/ops/e27_build.py", "code/ops/e27_score_p1.py",
                                                "code/data/gen_p1_verify_e27.py", "code/ops/phase_e25.py", "code/ops/run_supervised.py", "code/ops/eval_e24.py",
                                                "code/ops/e25_steps.py", "code/eval_buyer_full.py")}}}
p = f"{S}/plan.json"
json.dump(plan, open(p, "x"), indent=1, ensure_ascii=False)
open(f"{S}/plan.sha256", "x").write(sha(p) + "  plan.json\n")
os.chmod(p, 0o444)
os.chmod(f"{S}/plan.sha256", 0o444)
print(S, "frozen", sha(p))

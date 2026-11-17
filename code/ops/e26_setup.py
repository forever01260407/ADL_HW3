"""Prepare the E26 session before the controller starts, then freeze plan.json.

    python code/ops/e26_setup.py --session runs/E26_session_X

1. own_eval steps: export buyer_select_new / buyer_confirm_new from the frozen E26 files; copy the
   unchanged dev, dev_new, dev_bfcl, dev_confirm steps from E25 (sha256 checked equal).
2. E23 and C0 predictions on those unchanged dev sets are copied from E25 with their provenance
   files (same steps sha256, adapter sha256, generation settings); eval_e24.py re-checks them.
3. C0 = E25's C0 (the C0 schedule is byte-identical, checked by make_schedule_e26 --expect-c0):
   <session>/C0 is a symlink to E25's C0 run, and jobs/train_C0.json records the reuse so the
   controller verifies it instead of training it. Its verify uses E25's train pool hash.
4. C1 config (lr 3e-5, 200 updates, E26 pool and schedule), plan.json with the E25 gates plus the
   E26 public buyer veto, frozen read-only with plan.sha256.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

E25 = "runs/E25_session_20261109_123750"
E25_POOL = "data_work/frozen/e25_E25_session_20261109_123750/train_pool.jsonl"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument("--session", required=True)
a = ap.parse_args()
S = a.session
name = os.path.basename(S)
F = f"data_work/frozen/e26_{name}"
os.makedirs(f"{S}/own_eval/steps", exist_ok=True)
for n in ("buyer_select_new", "buyer_confirm_new"):
    subprocess.run([sys.executable, "code/eval_own.py", "export", "--data", f"{F}/{n}.jsonl", "--out", f"{S}/own_eval/steps/{n}.jsonl"], check=True)
reuse = {}
for n in ("dev", "dev_new", "dev_bfcl", "dev_confirm"):
    shutil.copyfile(f"{E25}/own_eval/steps/{n}.jsonl", f"{S}/own_eval/steps/{n}.jsonl")
    assert sha(f"{S}/own_eval/steps/{n}.jsonl") == sha(f"{E25}/own_eval/steps/{n}.jsonl")
for tag, sets in (("E23", ("dev", "dev_new", "dev_bfcl", "dev_confirm")), ("C0", ("dev", "dev_new", "dev_bfcl"))):
    os.makedirs(f"{S}/own_eval/{tag}", exist_ok=True)
    shutil.copyfile(f"{E25}/own_eval/{tag}/adapter.json", f"{S}/own_eval/{tag}/adapter.json")
    for n in sets:
        for suffix in (".jsonl", ".provenance.json"):
            shutil.copyfile(f"{E25}/own_eval/{tag}/{n}{suffix}", f"{S}/own_eval/{tag}/{n}{suffix}")
        prov = json.load(open(f"{S}/own_eval/{tag}/{n}.provenance.json"))
        assert prov["steps_sha256"] == sha(f"{S}/own_eval/steps/{n}.jsonl"), (tag, n)
        reuse[f"{tag}/{n}"] = {"pred_sha256": sha(f"{S}/own_eval/{tag}/{n}.jsonl"), "from": f"{E25}/own_eval/{tag}/{n}.jsonl"}
# C0 = E25 C0
os.symlink(os.path.realpath(f"{E25}/C0"), f"{S}/C0")
os.makedirs(f"{S}/jobs", exist_ok=True)
job = json.load(open(f"{E25}/jobs/train_C0.json"))
job["reused_in"] = name
job["note"] = "E25's C0 (identical schedule, same parent, same settings); not retrained in E26"
json.dump(job, open(f"{S}/jobs/train_C0.json", "x"), indent=1)
shutil.copyfile(f"{E25}/C0_config.json", f"{S}/C0_config.json")
cfg = json.load(open(f"{E25}/C1_config.json"))
cfg.update({"train_file": f"{F}/train_pool.jsonl", "schedule_file": f"{S}/plan/C1_schedule.json"})
json.dump(cfg, open(f"{S}/C1_config.json", "x"), indent=1)
json.dump({"reused": reuse, "C0": os.path.realpath(f"{E25}/C0")}, open(f"{S}/audit/reuse_report.json", "x"), indent=1)

plan = json.load(open(f"{E25}/plan.json"))
plan.update({
    "session": name, "frozen": F,
    "only_difference_C0_C1": "all 1,084 Part 2 positions hold E26 buyer steps in C1 (C0 = E25's C0: original data_v5 Part 2); Part 1 positions, settings and parent identical",
    "user_settings_before_T0": json.load(open(f"{S}/session.json"))["user_decisions"],
    "deviations": ["new round after E25 (user decision): public used as a design reference (TA's spoken guidance); see records/synthesis_spec_e26.md",
                   "public collision check treats the Offer line prefix as format (--offer-frame-as-format); exact and 0.6 near unchanged; whole-family exclusion unchanged",
                   "C0 is E25's C0 (schedule byte-identical); E23/C0 dev predictions reused from E25 with provenance",
                   "selection needs select gates AND confirm (the policy was chosen after looking at public, not at select)"],
    "public_veto_p1": 60.3,
    "public_veto_buyer": {"fill_rate_min": 0.930, "relative_price_max": 1.027, "order_accuracy_min": 0.966, "text_accuracy_min": 0.986,
                          "E23_public": {"fill_rate": 0.91983, "order_accuracy": 0.97075, "relative_price": 1.02675, "text_accuracy": 0.99133}},
})
plan["hashes"] = {"E23_adapter": plan["hashes"]["E23_adapter"], "train_pool": sha(f"{F}/train_pool.jsonl"), "C0_train_pool": sha(E25_POOL),
                  "patch": sha(f"{F}/buyer_patch_train.jsonl"), "select_new": sha(f"{F}/buyer_select_new.jsonl"), "confirm_new": sha(f"{F}/buyer_confirm_new.jsonl"),
                  "C0_schedule": sha(f"{S}/plan/C0_schedule.json"), "C1_schedule": sha(f"{S}/plan/C1_schedule.json"), "C1_config": sha(f"{S}/C1_config.json"),
                  "code": {p: sha(p) for p in ("code/train.py", "code/ops/run_supervised.py", "code/ops/guardian.py", "code/ops/eval_e24.py", "code/ops/phase_e25.py",
                                               "code/ops/e25_steps.py", "code/ops/e25_gates.py", "code/eval_buyer_full.py", "code/eval_own.py",
                                               "code/data/gen_buyer_e26.py", "code/data/check_buyer_e26.py", "code/data/dedupe_against_public.py",
                                               "code/ops/make_schedule_e26.py", "code/ops/e25_isolation.py", "code/ops/e26_setup.py")}}
plan["commands"]["final_public"] = ["bash", "code/eval_full.sh", "{S}/final_public", "release/candidates/" + name + "/adapter_checkpoint"]
plan["gates"]["public"] = "selected new model only, once, after selection.json: Part 1 >= 60.3 and buyer fill >= 0.930, price <= 1.027, order acc >= 0.966, text >= 0.986; else keep E23"
assert plan["hashes"]["C0_schedule"] == "e78ca78f6cb95c0d48b1ab8ee75c220fdfb0177333fd554373d9bab9067c8835"
p = f"{S}/plan.json"
json.dump(plan, open(p, "x"), indent=1, ensure_ascii=False)
open(f"{S}/plan.sha256", "x").write(sha(p) + "  plan.json\n")
os.chmod(p, 0o444)
os.chmod(f"{S}/plan.sha256", 0o444)
print("frozen", sha(p))

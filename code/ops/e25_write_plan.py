"""Write the E25 C0/C1 configs and plan.json, then freeze it (plan.sha256, read-only). Run once, before C0.

    python code/ops/e25_write_plan.py --session runs/E25_session_X
"""
import argparse, hashlib, json, os

ap = argparse.ArgumentParser()
ap.add_argument("--session", required=True)
a = ap.parse_args()
S = a.session
name = os.path.basename(S)
F = f"data_work/frozen/e25_{name}"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


sched = json.load(open(f"{S}/plan/schedule_report.json"))
for m in ("C0", "C1"):
    cfg = {"train_file": f"{F}/train_pool.jsonl", "max_examples": None, "seed": 42, "r": 32, "alpha": 64, "dropout": 0.05, "learning_rate": 3e-05,
           "weight_decay": 0.0, "warmup_ratio": 0.03, "max_grad_norm": 1.0, "micro_batch": 2, "accumulation": 8, "max_length": 8192, "max_updates": 200,
           "save_at": [100, 200], "part2_share": 0.339, "gradient_checkpointing": True, "print_every": 10, "schedule_file": f"{S}/plan/{m}_schedule.json"}
    json.dump(cfg, open(f"{S}/{m}_config.json", "x"), indent=1)
PY = "{PY}"
plan = {
    "session": name, "parent": "runs/E23/checkpoints/step_1475", "updates": 200, "frozen": F,
    "dev_confirm_records": "data_work/frozen/e24_session_20261109_031718/dev_confirm.jsonl",
    "select_sets_new": ["buyer_select_new", "buyer_confirm_new"], "select_sets_all": ["buyer_select_new", "dev", "dev_new", "dev_bfcl"],
    "confirm_sets": ["buyer_confirm_new", "dev_confirm"], "dev_sets": ["dev", "dev_new", "dev_bfcl"], "public_veto_p1": 60.3,
    "only_difference_C0_C1": "324 of the 1,084 Part 2 positions (162 per half) hold E25 patch steps in C1; every other position, all settings and the parent are identical",
    "user_settings_before_T0": json.load(open(f"{S}/session.json"))["user_overrides_before_T0"],
    "deviations": ["patch has 4 flow families, not the plan's 6: whole-family public exclusion removed deadline, first_at_target, price_up (5 near records); user approved continuing with 4 (~13:00)",
                   "confirm_new regenerated once with its own seller wording after the same-source check found it shared select's wording (public check was clean); no public-driven rewording",
                   "E23 predictions on dev/dev_new/dev_bfcl/dev_confirm reused from E24 after hash/settings verification (own_eval/E23/reuse_report.json)",
                   "first saved checkpoint is 100 (plan said 50) because the user set 200 updates; selection point fixed at 200"],
    "hashes": {"E23_adapter": sha("runs/E23/checkpoints/step_1475/adapter_model.safetensors"), "train_pool": sha(f"{F}/train_pool.jsonl"),
               "patch": sha(f"{F}/buyer_patch_train.jsonl"), "select_new": sha(f"{F}/buyer_select_new.jsonl"), "confirm_new": sha(f"{F}/buyer_confirm_new.jsonl"),
               "C0_schedule": sha(f"{S}/plan/C0_schedule.json"), "C1_schedule": sha(f"{S}/plan/C1_schedule.json"),
               "C0_config": sha(f"{S}/C0_config.json"), "C1_config": sha(f"{S}/C1_config.json"),
               "code": {p: sha(p) for p in ("code/train.py", "code/ops/run_supervised.py", "code/ops/guardian.py", "code/ops/eval_e24.py", "code/ops/phase_e25.py",
                                            "code/ops/e25_steps.py", "code/ops/e25_gates.py", "code/eval_buyer_full.py", "code/eval_own.py",
                                            "code/data/gen_buyer_patch_e25.py", "code/data/gen_buyer_eval_e25.py", "code/data/check_buyer_patch_e25.py",
                                            "code/ops/make_schedule_e25.py", "code/ops/e25_isolation.py")}},
    "gates": {
        "F2": {"p1_drop_pp": 0.5, "after_tool_drop_pp": 1.0, "problem_rise_pp": 0.5, "mean3": "not below E23"},
        "F3": {"order_acc_drop_pp": 0.5, "common_price_worse": 0.002, "min_orders": 50, "min_common": 50, "text_acc_drop_pp": 0.5, "text_coverage_drop_pp": 1.0,
               "fill_all": "not below E23", "fill_opportunity": "not below E23", "invalid_per_negotiation": "not above E23"},
        "F4": {"metric": "fill_rate over all buyer_select_new negotiations (pre-registered, the only improvement metric)", "min_pp": 2.0,
               "interval": "paired family bootstrap, 95% two-sided, lower end > 0", "draws": 3000, "seed": 42,
               "C1_credit": "C1 also needs the same improvement over C0; otherwise C0 may be provisional only by passing everything vs E23"},
        "F5": {"min_orders": 30, "min_common": 30, "order_acc_drop_pp": 1.0, "common_price_worse": 0.002, "text_acc_drop_pp": 1.0, "fill": "not below E23",
               "problem_rise_pp": 0.5, "p1_min_steps": 80, "p1_min_families": 4, "p1": "dev_confirm exact not below E23"},
        "public": "selected new model only, once, after selection.json: Part 1 headline < 60.3 -> keep E23"},
    "limits": {"wait_gpu": 1800, "eval_total": 2400, "train": 1800, "train_expect": 1500, "train_stall": 600, "train_init": 900,
               "judge": 2700, "judge_expect": 1500, "judge_confirm_expect": 900, "judge_stall": 900, "judge_init": 1200, "public": 3600, "public_expect": 2100},
    "poll": 15,
    "commands": {
        "eval": [PY, "code/ops/eval_e24.py", "--session", "{S}", "--tag", "{tag}", "--adapter", "{adapter}", "--sets", "{sets}", "--max-seconds-per-set", "1200"],
        "train": [PY, "code/train.py", "--config", "{S}/{m}_config.json", "--run", "{S}/{m}", "--init-adapter", "runs/E23/checkpoints/step_1475"],
        "verify": [PY, "code/ops/e25_steps.py", "verify", "--session", "{S}", "--model", "{m}"],
        "judge_build": [PY, "code/ops/e25_steps.py", "judge-build", "--session", "{S}", "--split", "{split}", "--models", "{models}"],
        "judge_run": [PY, "hw3/eval.py", "judge", "run", "--requests", "{S}/judge/{split}/requests.jsonl", "--out", "{S}/judge/{split}/judgments.jsonl", "--gpu", "0"],
        "judge_check": [PY, "code/ops/e25_steps.py", "judge-check", "--session", "{S}", "--split", "{split}"],
        "score": [PY, "code/ops/e25_steps.py", "score", "--session", "{S}", "--split", "{split}", "--models", "{models}"],
        "gates_select": [PY, "code/ops/e25_gates.py", "select", "--session", "{S}", "--out", "{S}/scores/provisional.json"],
        "gates_confirm": [PY, "code/ops/e25_gates.py", "confirm", "--session", "{S}", "--candidate", "{cand}", "--out", "{S}/selection_gates.json"],
        "copy_candidate": [PY, "code/ops/e25_steps.py", "copy", "--session", "{S}", "--model", "{m}"],
        "final_public": ["bash", "code/eval_full.sh", "{S}/final_public", "release/candidates/" + name + "/adapter_checkpoint"],
        "public_check": [PY, "code/ops/e25_steps.py", "public-check", "--session", "{S}"],
        "public_veto": [PY, "code/ops/e25_steps.py", "veto", "--session", "{S}"]},
}
p = f"{S}/plan.json"
json.dump(plan, open(p, "x"), indent=1, ensure_ascii=False)
open(f"{S}/plan.sha256", "x").write(sha(p) + "  plan.json\n")
os.chmod(p, 0o444)
os.chmod(f"{S}/plan.sha256", 0o444)
print("frozen", sha(p))

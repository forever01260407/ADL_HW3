"""E27 controller: one training run with checkpoints, own evaluation under the official rules, one
candidate, one full public evaluation, adopt or keep the base. Runs unattended.

    python code/ops/phase_e27.py --session runs/E27_session_X

Uses phase_e25.Run (frozen plan check, supervised jobs, status/events, handoff, E-base fallback).
Order: base on the new P1 dev set -> train (save_at checkpoints) -> each checkpoint: own P1 dev sets +
buyer_select_new -> judge (base + checkpoints, one batch) -> scores -> selection (plan "e27_select")
-> candidate copy -> full public -> adopt rule (plan "e27_adopt") -> handoff.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from phase_e25 import PY, Run  # noqa: E402


def sequence(R):
    P = R.plan
    L = P["limits"]
    base_tag, base_adapter = P["base"]["tag"], P["base"]["adapter"]
    R.fail = lambda what: R.handoff("failed_fallback_base", base_tag, failure=what, code=1)  # the base, not always E23
    R.log("controller", "started", base=base_tag)
    ev = lambda tag, adapter, sets: [PY, "code/ops/eval_e24.py", "--session", R.S, "--tag", tag, "--adapter", adapter, "--sets", *sets, "--max-seconds-per-set", "1200"]  # noqa: E731
    if not R.wait_gpu_free(L["wait_gpu"]):
        R.fail("GPU still held by another supervised job")
    if not R.cpu("eval_base", ev(base_tag, base_adapter, P["eval_sets"]), timeout=L["eval_total"]):
        R.fail("base evaluation")
    log = f"{R.S}/E27/train_log.jsonl"
    prev = f"{R.S}/jobs/train_E27.json"
    if not (os.path.exists(prev) and json.load(open(prev))["status"] == "completed"):
        ok = R.job("train_E27", [PY, "code/train.py", "--config", f"{R.S}/E27_config.json", "--run", f"{R.S}/E27", "--init-adapter", base_adapter],
                   progress=f"update:{log}", deadline_key="D_train", max_seconds=L["train"], expect=L["train_expect"], stall=L["train_stall"], init_grace=L["train_init"])
        if not ok:
            done = [c for c in P["checkpoints"] if os.path.exists(f"{R.S}/E27/checkpoints/step_{c}/done.json")]
            if not done:
                R.fail("train E27 (no checkpoint)")
            R.log("train_E27", "partial", checkpoints=done)
    ckpts = [c for c in P["checkpoints"] if os.path.exists(f"{R.S}/E27/checkpoints/step_{c}/done.json")]
    for c in ckpts:
        if time.time() > R.deadline("D_select"):
            R.log(f"eval_E27_{c}", "skipped", reason="D_select passed")
            continue
        if not R.cpu(f"verify_E27_{c}", [PY, "code/ops/e27_steps.py", "verify", "--session", R.S, "--step", str(c)]):
            R.fail(f"verify E27 step {c}")
        if not R.cpu(f"eval_E27_{c}", ev(f"E27_{c}", f"{R.S}/E27/checkpoints/step_{c}", P["eval_sets"]), timeout=L["eval_total"]):
            R.fail(f"eval E27 step {c}")
    models = [base_tag] + [f"E27_{c}" for c in ckpts if os.path.isdir(f"{R.S}/own_eval/E27_{c}")]
    if not R.cpu("judge_build", [PY, "code/ops/e25_steps.py", "judge-build", "--session", R.S, "--split", "select", "--models", *models]):
        R.fail("judge build")
    if not R.job("judge_select", [PY, "hw3/eval.py", "judge", "run", "--requests", f"{R.S}/judge/select/requests.jsonl", "--out", f"{R.S}/judge/select/judgments.jsonl", "--gpu", "0"],
                 progress=f"rows:{R.S}/judge/select/judgments.jsonl", deadline_key="D_select", max_seconds=L["judge"], expect=L["judge_expect"],
                 stall=L["judge_stall"], init_grace=L["judge_init"], done_check=[PY, "code/ops/e25_steps.py", "judge-check", "--session", R.S, "--split", "select"]):
        R.fail("judge")
    if not R.cpu("score_and_select", [PY, "code/ops/e27_steps.py", "select", "--session", R.S, "--models", *models]):
        R.fail("scoring / selection")
    sel = json.load(open(f"{R.S}/selection.json"))
    R.log("selection", sel["selected"])
    if sel["selected"] == base_tag:
        R.handoff("completed_kept_base", base_tag)
    if not R.cpu("candidate_copy", [PY, "code/ops/e27_steps.py", "copy", "--session", R.S]):
        R.fail("candidate copy")
    cand = f"release/candidates/{os.path.basename(R.S)}/adapter_checkpoint"
    if not R.job("final_public", ["bash", "code/eval_full.sh", f"{R.S}/final_public", cand], progress="none", deadline_key="D_gpu", max_seconds=L["public"],
                 expect=L["public_expect"], done_check=[PY, "code/ops/e25_steps.py", "public-check", "--session", R.S]):
        R.fail("full public evaluation")
    if not R.cpu("adopt", [PY, "code/ops/e27_steps.py", "adopt", "--session", R.S]):
        R.fail("adopt check")
    v = json.load(open(f"{R.S}/final_public/adopt.json"))
    R.handoff("completed_new_candidate" if v["adopt"] else "completed_public_kept_base", sel["selected"] if v["adopt"] else base_tag)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    a = ap.parse_args()
    R = Run(a.session, None)
    try:
        sequence(R)
    except SystemExit:
        raise
    except Exception as e:
        R.handoff("failed_fallback_base", R.plan["base"]["tag"], failure=f"controller exception: {type(e).__name__}: {e}", code=1)


if __name__ == "__main__":
    main()

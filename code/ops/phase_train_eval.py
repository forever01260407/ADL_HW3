"""E24 phase driver: E21 buyer/select baseline -> B0 train -> verify -> B0 eval -> B1 train -> verify -> B1 eval.

    python code/ops/phase_train_eval.py --session runs/E24_session_X

Runs on its own (nohup) so the sequence continues when nobody is watching. Every GPU job goes
through run_supervised.py (GPU lock, progress, stall and time limits); training jobs use the
session's D_train deadline and are refused if they cannot finish before it. A step that fails
stops the driver: no later GPU work starts. Checkpoints are used only when done.json exists,
names the right update and parent, and the adapter hash matches it. Writes <session>/phase.json.
"""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = "runs/E23/checkpoints/step_1475"
EVAL_SETS = ["dev_select", "buyer_select", "dev", "dev_new", "dev_bfcl"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def log(session, state, **kw):
    path = f"{session}/phase.json"
    data = json.load(open(path)) if os.path.exists(path) else {"steps": []}
    data["steps"].append({"time": time.strftime("%H:%M:%S"), "state": state, **kw})
    data["current"] = state
    tmp = path + ".tmp"
    json.dump(data, open(tmp, "w"), indent=1)
    os.replace(tmp, path)
    print(time.strftime("%H:%M:%S"), state, kw, flush=True)


def wait_gpu(session, limit=1800):
    end = time.time() + limit
    while time.time() < end:
        with open(f"{session}/gpu.lock", "a") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(f, fcntl.LOCK_UN)
                return True
            except BlockingIOError:
                pass
        time.sleep(20)
    return False


def verify(run, update):
    d = f"{run}/checkpoints/step_{update}"
    try:
        done = json.load(open(f"{d}/done.json"))
    except (OSError, ValueError):
        return f"no done.json at {d}"
    if done.get("update") != update or done.get("parent_sha256") != sha(f"{PARENT}/adapter_model.safetensors"):
        return "done.json does not match the update or parent"
    if done.get("adapter_sha256") != sha(f"{d}/adapter_model.safetensors"):
        return "adapter hash differs from done.json"
    from safetensors import safe_open
    with safe_open(f"{d}/adapter_model.safetensors", "pt") as f:
        if not list(f.keys()):
            return "empty adapter"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--skip-e21", action="store_true")
    a = ap.parse_args()
    S = os.path.realpath(a.session)
    py = sys.executable
    if not a.skip_e21:
        if not wait_gpu(S):
            return log(S, "failed", why="GPU lock not free for 30 min before the E21 baseline")
        r = subprocess.run([py, f"{HERE}/eval_e24.py", "--session", S, "--tag", "E21", "--adapter", "runs/E21/checkpoints/step_1375", "--sets", "buyer_select", "dev_select"])
        if r.returncode:
            return log(S, "failed", why="E21 baseline evaluation failed")
        log(S, "e21_baseline_done")
    for name in ("B0", "B1"):
        run = f"{S}/{name}"
        if not wait_gpu(S):
            return log(S, "failed", why=f"GPU lock not free before {name} training")
        log(S, f"{name}_training_start")
        r = subprocess.run([py, f"{HERE}/run_supervised.py", "--session", S, "--name", f"train_{name}", "--gpu", "--deadline-key", "D_train",
                            "--progress", f"update:{run}/train_log.jsonl", "--init-grace", "900", "--stall", "600", "--max-seconds", "3600",
                            "--expect-seconds", "2700", "--poll", "30", "--",
                            py, "code/train.py", "--config", f"{S}/{name}_config.json", "--run", run, "--init-adapter", PARENT])
        problems = [p for p in (verify(run, 150), verify(run, 300)) if p]
        if r.returncode or problems:
            return log(S, "failed", why=f"{name} training", exit=r.returncode, problems=problems)
        log(S, f"{name}_trained", adapter_sha256=sha(f"{run}/checkpoints/step_300/adapter_model.safetensors"))
        if not wait_gpu(S):
            return log(S, "failed", why=f"GPU lock not free before {name} evaluation")
        r = subprocess.run([py, f"{HERE}/eval_e24.py", "--session", S, "--tag", f"{name}_300", "--adapter", f"{run}/checkpoints/step_300", "--sets", *EVAL_SETS])
        if r.returncode:
            return log(S, "failed", why=f"{name} evaluation failed")
        log(S, f"{name}_evaluated")
    log(S, "completed")


if __name__ == "__main__":
    main()

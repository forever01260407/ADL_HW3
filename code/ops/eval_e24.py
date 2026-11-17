"""Generate predictions for one adapter on E24's own evaluation sets, every job supervised.

    python code/ops/eval_e24.py --session runs/E24_session_X --tag E23 --adapter runs/E23/checkpoints/step_1475 \
        --sets dev_select buyer_select [--attempt 1] [--max-seconds-per-set 1800]

Steps files are <session>/own_eval/steps/<set>.jsonl (made by `code/eval_own.py export`). For each
set: official `hw3/eval.py generate` (vLLM, greedy, official settings) through run_supervised.py
with the GPU lock, progress = distinct prediction ids, and a done-check by
code/audit/check_predictions.py (every id present, no structural error). The merged model is
kept in <session>/own_eval/<tag>/merged while the sets run and removed afterwards (only that
path, after checking it lies inside the session). Predictions already complete are not redone;
a partial file of the same adapter is resumed by the official generate (it skips done ids).
Exit 1 if any set is not completed. E25: each prediction file has <set>.provenance.json (steps
sha256, adapter sha256, generation settings); an existing prediction whose provenance differs or is
missing is refused, not reused or resumed.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def complete(steps, pred):
    if not os.path.exists(pred):
        return False
    r = subprocess.run([sys.executable, "code/audit/check_predictions.py", "--data", steps, "--pred", pred], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--attempt", default="1")
    ap.add_argument("--max-seconds-per-set", type=float, default=1800)
    ap.add_argument("--gpu-util", default="0.85")
    a = ap.parse_args()
    session = os.path.realpath(a.session)
    out = f"{session}/own_eval/{a.tag}"
    merged = f"{out}/merged"
    os.makedirs(out, exist_ok=True)
    marker = f"{out}/adapter.json"
    import hashlib
    adapter_hash = hashlib.sha256(open(f"{a.adapter}/adapter_model.safetensors", "rb").read()).hexdigest()
    if os.path.exists(marker):
        if json.load(open(marker))["adapter_sha256"] != adapter_hash:
            raise SystemExit(f"{out} holds predictions of another adapter")
    else:
        json.dump({"adapter": a.adapter, "adapter_sha256": adapter_hash}, open(marker, "w"), indent=1)
    failed = []
    fmt_sha = hashlib.sha256(open("hw3/eval/format.py", "rb").read()).hexdigest()
    for name in a.sets:
        steps = f"{session}/own_eval/steps/{name}.jsonl"
        pred = f"{out}/{name}.jsonl"
        # E25: a prediction file is only reused or resumed when it was made from the same steps file,
        # adapter and generation settings (same ids with changed data must not pass as done).
        prov_path = f"{out}/{name}.provenance.json"
        prov = {"steps_sha256": hashlib.sha256(open(steps, "rb").read()).hexdigest(), "adapter_sha256": adapter_hash,
                "settings": {"backend": "vllm", "gpu_util": a.gpu_util, "hw3_eval_format_sha256": fmt_sha}}
        if os.path.exists(pred):
            if not os.path.exists(prov_path) or json.load(open(prov_path)) != prov:
                print(f"{name}: existing predictions do not match this steps file / adapter / settings; refusing to reuse", flush=True)
                failed.append(name)
                break
        else:
            json.dump(prov, open(prov_path, "w"), indent=1)
        if complete(steps, pred):
            print(f"{name}: already complete", flush=True)
            continue
        job = f"gen_{a.tag}_{name}_a{a.attempt}"
        argv = [sys.executable, f"{HERE}/run_supervised.py", "--session", session, "--name", job, "--gpu",
                "--progress", f"lines:{pred}", "--init-grace", "900", "--stall", "600", "--max-seconds", str(a.max_seconds_per_set),
                "--expect-seconds", "300", "--poll", "15",
                "--done-check", json.dumps([sys.executable, "code/audit/check_predictions.py", "--data", steps, "--pred", pred]),
                "--", sys.executable, "hw3/eval.py", "generate", "--data", steps, "--adapter", a.adapter, "--merge-dir", merged,
                "--out", pred, "--backend", "vllm", "--gpu-util", a.gpu_util]
        r = subprocess.run(argv)
        print(f"{name}: supervisor exit {r.returncode}", flush=True)
        if r.returncode != 0:
            failed.append(name)
            break  # do not start more GPU work after a failure
    if os.path.realpath(merged).startswith(session + "/") and os.path.isdir(merged):
        shutil.rmtree(merged)
    print(json.dumps({"tag": a.tag, "failed": failed}))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

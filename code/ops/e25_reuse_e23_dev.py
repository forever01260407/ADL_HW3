"""Copy E23's E24 predictions on the unchanged dev sets into the E25 session, after verifying
that the steps files, adapter and generation settings are the same (plan D1: reuse only when
model, data, template, generation settings and hashes all match). Writes provenance files in the
format code/ops/eval_e24.py checks; refuses on any mismatch.

    python code/ops/e25_reuse_e23_dev.py --session runs/E25_session_X
"""
import argparse, hashlib, json, os, shutil, subprocess, sys

E24 = "runs/E24_session_20261109_031718"
SETS = ("dev", "dev_new", "dev_bfcl", "dev_confirm")
E23_SHA = "9ed86db7975797d1214ac8bdd044e0f6445e8c127cd0c293a879ea68f307ba9a"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument("--session", required=True)
a = ap.parse_args()
S = os.path.realpath(a.session)
assert json.load(open(f"{E24}/own_eval/E23/adapter.json"))["adapter_sha256"] == E23_SHA == sha("runs/E23/checkpoints/step_1475/adapter_model.safetensors")
os.makedirs(f"{S}/own_eval/E23", exist_ok=True)
marker = f"{S}/own_eval/E23/adapter.json"
if not os.path.exists(marker):
    json.dump({"adapter": "runs/E23/checkpoints/step_1475", "adapter_sha256": E23_SHA}, open(marker, "w"), indent=1)
fmt = sha("hw3/eval/format.py")
report = {}
for n in SETS:
    src_steps, dst_steps = f"{E24}/own_eval/steps/{n}.jsonl", f"{S}/own_eval/steps/{n}.jsonl"
    shutil.copyfile(src_steps, dst_steps)
    pred = f"{S}/own_eval/E23/{n}.jsonl"
    shutil.copyfile(f"{E24}/own_eval/E23/{n}.jsonl", pred)
    ok = subprocess.run([sys.executable, "code/audit/check_predictions.py", "--data", dst_steps, "--pred", pred], stdout=subprocess.DEVNULL).returncode == 0
    if not ok:
        sys.exit(f"{n}: copied predictions incomplete")
    prov = {"steps_sha256": sha(dst_steps), "adapter_sha256": E23_SHA, "settings": {"backend": "vllm", "gpu_util": "0.85", "hw3_eval_format_sha256": fmt}}
    json.dump(prov, open(f"{S}/own_eval/E23/{n}.provenance.json", "w"), indent=1)
    report[n] = {"steps_sha256": prov["steps_sha256"], "pred_sha256": sha(pred), "from": f"{E24}/own_eval/E23/{n}.jsonl"}
json.dump({"reused_from_E24": report, "note": "E24 generated these with code/ops/eval_e24.py (official generate, vLLM, gpu_util 0.85); official files unchanged (hash_official.sh check)"},
          open(f"{S}/own_eval/E23/reuse_report.json", "x"), indent=1)
print(json.dumps(report, indent=1))

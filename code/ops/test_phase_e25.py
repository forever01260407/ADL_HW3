"""Drills for the E25 controller with CPU dummy commands (plan 11.3 items 1, 3, 4, 6).

    python code/ops/test_phase_e25.py --out runs/E25_session_X/audit/controller_tests.json

A  full chain with nobody issuing commands: C0 then C1 train automatically, judge, gates, confirm,
   selection, candidate copy, public, veto -> handoff completed_new_candidate, exit 0.
B  verification fails after C0 exits 0 -> controller exits non-zero, handoff failed_fallback_E23,
   C1 never starts.
C  D_train already passed -> training refused, exit non-zero; a restart in the same session gets no
   new time (deadlines come from session.json; finished job names cannot be reused).
D  eval_e24.py refuses a prediction file whose steps file changed while ids stayed the same.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

TRAIN = "import json,os,sys,time\nos.makedirs(sys.argv[1],exist_ok=True)\nfor i in range(1,4):\n    open(sys.argv[1]+'/train_log.jsonl','a').write(json.dumps({'update':i,'loss':0.1})+'\\n'); time.sleep(0.3)\n"
WRITE = "import json,os,sys\nos.makedirs(os.path.dirname(sys.argv[1]),exist_ok=True)\njson.dump(json.loads(sys.argv[2]),open(sys.argv[1],'w'))\n"
OK = [PY, "-c", "pass"]


def session(root, name, d_train_in=3600):
    S = f"{root}/{name}"
    shutil.rmtree(S, ignore_errors=True)
    os.makedirs(S)
    t = time.time()
    dl = {k: {"epoch": t + v} for k, v in (("D_ready", 3600), ("D_train", d_train_in), ("D_select", 3600), ("D_gpu", 3600))}
    json.dump({"deadlines": dl}, open(f"{S}/session.json", "w"))
    return S


def plan(S, verify_ok=True, provisional="C1"):
    w = lambda path, obj: [PY, "-c", WRITE, path, json.dumps(obj)]  # noqa: E731
    p = {"parent": "x", "updates": 3, "select_sets_new": ["a"], "select_sets_all": ["a"], "confirm_sets": ["a"], "poll": 0.5,
         "limits": {"wait_gpu": 5, "eval_total": 30, "train": 30, "train_expect": 1, "train_stall": 5, "train_init": 5, "judge": 30, "judge_expect": 1,
                    "judge_confirm_expect": 1, "judge_stall": 5, "judge_init": 5, "public": 30, "public_expect": 1},
         "commands": {"eval": OK, "train": [PY, "-c", TRAIN, "{S}/{m}"], "verify": OK if verify_ok else [PY, "-c", "import sys; sys.exit(1)"],
                      "judge_build": OK, "judge_run": [PY, "-c", WRITE, "{S}/judge/{split}/judgments.jsonl", json.dumps({"id": "k"})], "judge_check": OK,
                      "score": OK, "gates_select": w("{S}/scores/provisional.json", {"provisional_candidate": provisional}),
                      "gates_confirm": w("{S}/selection_gates.json", {"selected": "{cand}"}), "copy_candidate": OK, "final_public": OK, "public_check": OK,
                      "public_veto": w("{S}/final_public/veto.json", {"passes": True})}}
    path = f"{S}/plan.json"
    json.dump(p, open(path, "w"))
    open(f"{S}/plan.sha256", "w").write(hashlib.sha256(open(path, "rb").read()).hexdigest() + "  plan.json\n")


def drive(S):
    t = time.monotonic()
    r = subprocess.run([PY, f"{HERE}/phase_e25.py", "--session", S], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=300)
    h = json.load(open(f"{S}/handoff.json")) if os.path.exists(f"{S}/handoff.json") else {}
    jobs = sorted(os.listdir(f"{S}/jobs")) if os.path.isdir(f"{S}/jobs") else []
    return r.returncode, round(time.monotonic() - t, 1), h, jobs, r.stdout[-800:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = os.path.dirname(os.path.realpath(a.out)) + "/controller_drills"
    os.makedirs(root, exist_ok=True)
    res = []

    S = session(root, "A")
    plan(S)
    code, el, h, jobs, out = drive(S)
    order = [s["step"] for s in h.get("steps", [])]
    ok = (code == 0 and h.get("outcome") == "completed_new_candidate" and h.get("selected") == "C1"
          and order.index("train_C0") < order.index("verify_C0") < order.index("train_C1") < order.index("judge_select") < order.index("final_public"))
    res.append({"drill": "A_full_chain_unattended", "passed": ok, "exit": code, "elapsed": el, "outcome": h.get("outcome"), "order": order})

    S = session(root, "B")
    plan(S, verify_ok=False)
    code, el, h, jobs, out = drive(S)
    ok = code != 0 and h.get("outcome") == "failed_fallback_E23" and "train_C1.json" not in jobs
    res.append({"drill": "B_verify_fails_nonzero", "passed": ok, "exit": code, "outcome": h.get("outcome"), "failure": h.get("failure"), "jobs": jobs})

    S = session(root, "C", d_train_in=-10)
    plan(S)
    code, el, h, jobs, out = drive(S)
    refused = json.load(open(f"{S}/jobs/train_C0.json"))["status"] == "refused" if os.path.exists(f"{S}/jobs/train_C0.json") else False
    os.remove(f"{S}/handoff.json")
    code2, _, h2, _, _ = drive(S)
    ok = code != 0 and refused and code2 != 0 and h2.get("outcome") == "failed_fallback_E23"
    res.append({"drill": "C_deadline_and_restart", "passed": ok, "exit": code, "train_refused": refused, "restart_exit": code2, "restart_failure": h2.get("failure")})

    S = session(root, "D")
    os.makedirs(f"{S}/own_eval/steps")
    os.makedirs(f"{S}/adapter")
    open(f"{S}/adapter/adapter_model.safetensors", "wb").write(b"dummy")
    steps = f"{S}/own_eval/steps/x.jsonl"
    open(steps, "w").write(json.dumps({"id": "s1", "messages": []}) + "\n")
    os.makedirs(f"{S}/own_eval/T")
    open(f"{S}/own_eval/T/x.jsonl", "w").write(json.dumps({"id": "s1", "output": "hi", "finish_reason": "stop"}) + "\n")
    json.dump({"steps_sha256": "0" * 64, "adapter_sha256": "x", "settings": {}}, open(f"{S}/own_eval/T/x.provenance.json", "w"))
    r = subprocess.run([PY, f"{HERE}/eval_e24.py", "--session", S, "--tag", "T", "--adapter", f"{S}/adapter", "--sets", "x"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
    ok = r.returncode != 0 and "refusing to reuse" in r.stdout and not os.path.isdir(f"{S}/jobs")
    res.append({"drill": "D_changed_steps_refused", "passed": ok, "exit": r.returncode, "out": r.stdout[-300:]})

    for x in res:
        print("PASS" if x["passed"] else "FAIL", x["drill"], {k: v for k, v in x.items() if k in ("exit", "outcome", "failure")}, flush=True)
    passed = all(x["passed"] for x in res)
    json.dump({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "passed": passed, "drills": res}, open(a.out, "x"), indent=1)
    print("ALL PASSED" if passed else "SOME FAILED")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

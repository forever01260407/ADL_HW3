"""Helper steps of the E25 controller (code/ops/phase_e25.py). Each exits non-zero on any problem.

    python code/ops/e25_steps.py verify --session S --model C0
    python code/ops/e25_steps.py judge-build --session S --split select --models E23 C0 C1
    python code/ops/e25_steps.py judge-check --session S --split select
    python code/ops/e25_steps.py score --session S --split select --models E23 C0 C1
    python code/ops/e25_steps.py copy --session S --model C1
    python code/ops/e25_steps.py public-check --session S
    python code/ops/e25_steps.py veto --session S

verify: checkpoint step_<updates>/done.json names update=<updates>, the E23 parent and its hash,
the frozen train pool and this model's frozen schedule; the adapter hash matches; LoRA r/alpha,
target modules and trainable chat rows equal E23's. Writes scores/<M>_integrity.json.
judge-build: the official `hw3/eval.py judge build` on the split's buyer steps, one prediction file
per model (named <M>.jsonl so each model's replies keep their own source), one batch.
judge-check: every request key has a judgment row.
score: eval_own score (--by-source) on the dev sets and eval_buyer_full score with the judgments.
copy: the selected adapter into release/candidates/<session>/adapter_checkpoint, hashes checked.
veto: final_public Part 1 headline (first-step rule, lenient) >= plan veto -> final_public/veto.json.
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys

PY = sys.executable
E23 = "runs/E23/checkpoints/step_1475"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def run(argv):
    r = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(r.stdout[-1500:], flush=True)
    if r.returncode != 0:
        sys.exit(f"failed: {' '.join(argv[:4])}")


def arch(path):
    c = json.load(open(f"{path}/adapter_config.json"))
    return {"r": c["r"], "alpha": c["lora_alpha"], "dropout": c["lora_dropout"], "targets": sorted(c["target_modules"]), "rows": c.get("trainable_token_indices")}


def verify(a, P):
    ck = f"{a.session}/{a.model}/checkpoints/step_{P['updates']}"
    done = json.load(open(f"{ck}/done.json"))
    want = {"update": P["updates"], "parent_adapter": E23, "parent_sha256": P["hashes"]["E23_adapter"],
            "train_sha256": P["hashes"].get(f"{a.model}_train_pool", P["hashes"]["train_pool"]),  # E26: C0 is E25's C0 (E25 pool)
            "schedule_sha256": P["hashes"][f"{a.model}_schedule"]}
    checks = {k: done.get(k) == v for k, v in want.items()}
    checks["adapter_hash"] = done["adapter_sha256"] == sha(f"{ck}/adapter_model.safetensors")
    checks["architecture_same_as_E23"] = arch(ck) == arch(E23)
    log = [json.loads(l) for l in open(f"{a.session}/{a.model}/train_log.jsonl") if l.strip()]
    checks["log_reaches_last_update"] = bool(log) and max(r.get("update", 0) for r in log) == P["updates"]
    out = {"checkpoint": ck, "checks": checks, "passed": all(checks.values()), "done": done}
    os.makedirs(f"{a.session}/scores", exist_ok=True)  # amendment 1: the folder did not exist yet at the first verify
    json.dump(out, open(f"{a.session}/scores/{a.model}_integrity.json", "x"), indent=1)
    print(json.dumps(checks))
    sys.exit(0 if out["passed"] else 1)


def judge_build(a, P):
    d = f"{a.session}/judge/{a.split}"
    os.makedirs(f"{d}/preds", exist_ok=True)
    steps = f"{a.session}/own_eval/steps/buyer_{a.split}_new.jsonl"
    preds = []
    for m in a.models:
        src = f"{a.session}/own_eval/{m}/buyer_{a.split}_new.jsonl"
        dst = f"{d}/preds/{m}.jsonl"
        shutil.copyfile(src, dst)
        preds.append(dst)
    run([PY, "hw3/eval.py", "judge", "build", "--data", steps, "--pred", *preds, "--out", f"{d}/requests.jsonl"])
    json.dump({"steps": steps, "steps_sha256": sha(steps), "preds": {p: sha(p) for p in preds}, "requests_sha256": sha(f"{d}/requests.jsonl")},
              open(f"{d}/build.json", "x"), indent=1)


def judge_check(a, P):
    d = f"{a.session}/judge/{a.split}"
    keys = {json.loads(l)["key"] for l in open(f"{d}/requests.jsonl") if l.strip()}
    got = {}
    if os.path.exists(f"{d}/judgments.jsonl"):
        for l in open(f"{d}/judgments.jsonl"):
            if l.strip():
                r = json.loads(l)
                got[r["key"]] = r["label"]
    missing = keys - set(got)
    print(json.dumps({"requests": len(keys), "judged": len(keys & set(got)), "missing": len(missing),
                      "labels": {x: sum(1 for k in keys if got.get(k) == x) for x in ("Yes", "No", "unparsed")}}))
    sys.exit(0 if not missing else 1)


def score(a, P):
    S = a.session
    os.makedirs(f"{S}/scores", exist_ok=True)
    dev_sets = P["dev_sets"] if a.split == "select" else ["dev_confirm"]
    for m in a.models:
        for n in dev_sets:
            out = f"{S}/scores/{m}_{n}.metrics.json"
            if not os.path.exists(out):
                run([PY, "code/eval_own.py", "score", "--steps", f"{S}/own_eval/steps/{n}.jsonl", "--pred", f"{S}/own_eval/{m}/{n}.jsonl", "--out", out, "--by-source", "--show", "0"])
        name = f"buyer_{a.split}_new"
        run([PY, "code/eval_buyer_full.py", "score", "--records", f"{P['frozen']}/{name}.jsonl", "--steps", f"{S}/own_eval/steps/{name}.jsonl",
             "--pred", f"{S}/judge/{a.split}/preds/{m}.jsonl", "--judgments", f"{S}/judge/{a.split}/judgments.jsonl", "--out", f"{S}/scores/{m}_{name}.json"])
    if a.split == "confirm":
        fams = {json.loads(l)["family"] for l in open(P["dev_confirm_records"]) if l.strip()}
        json.dump({"families": len(fams), "names": sorted(fams), "source": P["dev_confirm_records"]}, open(f"{S}/scores/dev_confirm_families.json", "x"), indent=1)


def copy(a, P):
    src = f"{a.session}/{a.model}/checkpoints/step_{P['updates']}"
    dst = f"release/candidates/{os.path.basename(a.session)}/adapter_checkpoint"
    if os.path.exists(dst):
        sys.exit(f"{dst} exists")
    os.makedirs(dst)
    for n in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copyfile(f"{src}/{n}", f"{dst}/{n}")
        if sha(f"{src}/{n}") != sha(f"{dst}/{n}"):
            sys.exit(f"copy of {n} differs")
    json.dump({"from": src, "model": a.model, "adapter_sha256": sha(f"{dst}/adapter_model.safetensors"), "config_sha256": sha(f"{dst}/adapter_config.json")},
              open(f"{a.session}/candidate.json", "x"), indent=1)


def public_check(a, P):
    d = f"{a.session}/final_public"
    run([PY, "code/audit/check_predictions.py", "--data", "hw3/data/public_test.jsonl.gz", "--pred", f"{d}/predictions/tools.jsonl"])
    run([PY, "code/audit/check_predictions.py", "--data", "hw3/data/public_test_negotiation.jsonl.gz", "--pred", f"{d}/predictions/negotiation.jsonl"])
    for n in ("tools.json", "negotiation.json"):
        if not os.path.exists(f"{d}/metrics/{n}"):
            sys.exit(f"missing metrics/{n}")


def veto(a, P):
    d = f"{a.session}/final_public"
    t = json.load(open(f"{d}/metrics/tools.json"))["headline"]
    p1 = 100 * t["with_first_step_rule"]["all"]["weighted_step_acc"]
    neg = json.load(open(f"{d}/metrics/negotiation.json"))["metrics"]
    checks = {"part1": p1 >= P["public_veto_p1"]}
    buyer = P.get("public_veto_buyer")  # E26: the buyer must not be worse than E23 on public either
    if buyer:
        checks.update({"fill_rate": neg["fill_rate"] >= buyer["fill_rate_min"], "relative_price": neg["relative_price"] <= buyer["relative_price_max"],
                       "order_accuracy": neg["order_accuracy"] >= buyer["order_accuracy_min"],
                       "text_accuracy": isinstance(neg.get("text_accuracy"), (int, float)) and neg["text_accuracy"] >= buyer["text_accuracy_min"]})
    out = {"part1_headline": p1, "veto_threshold": P["public_veto_p1"], "buyer_thresholds": buyer, "checks": checks, "passes": all(checks.values()),
           "negotiation": {k: neg.get(k) for k in ("fill_rate", "order_accuracy", "relative_price", "text_accuracy", "n_model_orders", "n_negotiations")},
           "note": "one look at public, after selection.json; not used to pick among models"}
    json.dump(out, open(f"{d}/veto.json", "x"), indent=1, default=str)
    print(json.dumps({k: out[k] for k in ("part1_headline", "checks", "passes", "negotiation")}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("verify", "judge-build", "judge-check", "score", "copy", "public-check", "veto"))
    ap.add_argument("--session", required=True)
    ap.add_argument("--model")
    ap.add_argument("--split", choices=("select", "confirm"))
    ap.add_argument("--models", nargs="+")
    a = ap.parse_args()
    P = json.load(open(f"{a.session}/plan.json"))
    {"verify": verify, "judge-build": judge_build, "judge-check": judge_check, "score": score, "copy": copy,
     "public-check": public_check, "veto": veto}[a.cmd](a, P)


if __name__ == "__main__":
    main()

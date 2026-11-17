"""Helper steps of the E27 controller (code/ops/phase_e27.py). Each exits non-zero on any problem.

    python code/ops/e27_steps.py verify --session S --step 200
    python code/ops/e27_steps.py select --session S --models BASE E27_200 E27_300 ...
    python code/ops/e27_steps.py copy   --session S
    python code/ops/e27_steps.py adopt  --session S

select (rules frozen in plan.json "e27_select"):
  P1 = mean over plan "p1_sets" of the official-rule weighted score (code/ops/e27_score_p1.py);
  buyer = code/eval_buyer_full.py with the official judge on buyer_select_new.
  A checkpoint qualifies when P1 >= base P1 + min_gain and its buyer fill / order accuracy / text
  accuracy do not drop below the base by more than the plan's tolerances. The qualifying checkpoint
  with the highest P1 is selected; none -> the base. Writes scores/*.json and selection.json.
adopt (plan "e27_adopt"): the candidate's full public Part 1 headline must beat the base's public
  headline by min_gain, and its public buyer metrics stay within the tolerances of the base's.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

PY = sys.executable


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def run(argv):
    r = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(r.stdout[-1200:], flush=True)
    if r.returncode != 0:
        sys.exit(f"failed: {' '.join(argv[:4])}")


def verify(a, P):
    ck = f"{a.session}/E27/checkpoints/step_{a.step}"
    done = json.load(open(f"{ck}/done.json"))
    checks = {"update": done.get("update") == a.step, "parent": done.get("parent_sha256") == P["hashes"]["base_adapter"],
              "train": done.get("train_sha256") == P["hashes"]["train_pool"], "schedule": done.get("schedule_sha256") == P["hashes"]["schedule"],
              "adapter_hash": done.get("adapter_sha256") == sha(f"{ck}/adapter_model.safetensors")}
    base = json.load(open(f"{P['base']['adapter']}/adapter_config.json"))
    mine = json.load(open(f"{ck}/adapter_config.json"))
    norm = lambda c: {k: (sorted(c.get(k) or []) if k == "target_modules" else c.get(k)) for k in ("r", "lora_alpha", "lora_dropout", "target_modules", "trainable_token_indices")}  # noqa: E731
    checks["architecture"] = norm(base) == norm(mine)  # amendment 1: PEFT saves target_modules (a set) in arbitrary order
    os.makedirs(f"{a.session}/scores", exist_ok=True)
    json.dump({"checkpoint": ck, "checks": checks, "passed": all(checks.values())}, open(f"{a.session}/scores/E27_{a.step}_integrity.json", "x"), indent=1)
    print(json.dumps(checks))
    sys.exit(0 if all(checks.values()) else 1)


def select(a, P):
    S = a.session
    sys.path.insert(0, "code/ops")
    sys.path.insert(0, "hw3")
    import e27_score_p1 as SP
    from eval import format as F
    rows = {}
    for m in a.models:
        p1 = {}
        for n in P["p1_sets"]:
            steps = list(F.iter_jsonl(f"{S}/own_eval/steps/{n}.jsonl"))
            preds, _ = F.load_predictions([f"{S}/own_eval/{m}/{n}.jsonl"])
            p1[n] = SP.score(steps, preds)
        out = f"{S}/scores/{m}_buyer_select_new.json"
        if not os.path.exists(out):
            run([PY, "code/eval_buyer_full.py", "score", "--records", P["buyer_records"], "--steps", f"{S}/own_eval/steps/buyer_select_new.jsonl",
                 "--pred", f"{S}/judge/select/preds/{m}.jsonl", "--judgments", f"{S}/judge/select/judgments.jsonl", "--out", out])
        b = json.load(open(out))
        om = b["official_metrics"]
        rows[m] = {"p1": {n: {"weighted": v["weighted"], "must_call": v["must_call"]} for n, v in p1.items()},
                   "p1_mean": sum(v["weighted"] for v in p1.values()) / len(p1),
                   "buyer": {"fill": 100 * om["fill_rate"], "order_acc": 100 * om["order_accuracy"] if om["order_accuracy"] is not None else None,
                             "text_acc": 100 * om["text_accuracy"] if isinstance(om.get("text_accuracy"), (int, float)) else None, "price": om["relative_price"]}}
    g = P["e27_select"]
    base = rows[P["base"]["tag"]]
    best, why = None, {}
    for m in a.models[1:]:
        r = rows[m]
        if not json.load(open(f"{S}/scores/{m}_integrity.json"))["passed"]:
            why[m] = "integrity"
            continue
        ok = {"p1_gain": r["p1_mean"] >= base["p1_mean"] + g["p1_min_gain"],
              "fill": r["buyer"]["fill"] >= base["buyer"]["fill"] - g["fill_drop_pp"],
              "order_acc": r["buyer"]["order_acc"] is not None and r["buyer"]["order_acc"] >= base["buyer"]["order_acc"] - g["order_acc_drop_pp"],
              "text_acc": r["buyer"]["text_acc"] is not None and r["buyer"]["text_acc"] >= base["buyer"]["text_acc"] - g["text_acc_drop_pp"]}
        why[m] = ok
        if all(ok.values()) and (best is None or r["p1_mean"] > rows[best]["p1_mean"]):
            best = m
    sel = {"time": __import__("time").strftime("%Y-%m-%d %H:%M:%S"), "rows": rows, "checks": why, "selected": best or P["base"]["tag"],
           "adapter": f"{S}/E27/checkpoints/step_{best.split('_')[1]}" if best else P["base"]["adapter"], "public_used_for_selection": False}
    json.dump(rows, open(f"{S}/scores/summary.json", "x"), indent=1)
    json.dump(sel, open(f"{S}/selection.json", "x"), indent=1)
    print(json.dumps({m: {"p1_mean": round(r["p1_mean"], 2), **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in r["buyer"].items()}} for m, r in rows.items()}, indent=1))
    print("checks", json.dumps(why), "selected", sel["selected"])


def copy(a, P):
    sel = json.load(open(f"{a.session}/selection.json"))
    dst = f"release/candidates/{os.path.basename(a.session)}/adapter_checkpoint"
    if os.path.exists(dst):
        sys.exit(f"{dst} exists")
    os.makedirs(dst)
    for n in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copyfile(f"{sel['adapter']}/{n}", f"{dst}/{n}")
        assert sha(f"{sel['adapter']}/{n}") == sha(f"{dst}/{n}")
    json.dump({"from": sel["adapter"], "adapter_sha256": sha(f"{dst}/adapter_model.safetensors")}, open(f"{a.session}/candidate.json", "x"), indent=1)


def adopt(a, P):
    d = f"{a.session}/final_public"
    t = json.load(open(f"{d}/metrics/tools.json"))["headline"]
    p1 = 100 * t["with_first_step_rule"]["all"]["weighted_step_acc"]
    neg = json.load(open(f"{d}/metrics/negotiation.json"))["metrics"]
    g, base = P["e27_adopt"], P["base"]["public"]
    checks = {"p1": p1 >= base["p1"] + g["p1_min_gain"],
              "fill": neg["fill_rate"] >= base["fill_rate"] - g["fill_drop"],
              "price": neg["relative_price"] <= base["relative_price"] + g["price_rise"],
              "order_acc": neg["order_accuracy"] >= base["order_accuracy"] - g["order_acc_drop"],
              "text_acc": isinstance(neg.get("text_accuracy"), (int, float)) and neg["text_accuracy"] >= base["text_accuracy"] - g["text_acc_drop"]}
    out = {"p1": p1, "negotiation": {k: neg.get(k) for k in ("fill_rate", "order_accuracy", "relative_price", "text_accuracy")}, "base": base, "rules": g,
           "checks": checks, "adopt": all(checks.values())}
    json.dump(out, open(f"{d}/adopt.json", "x"), indent=1)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("verify", "select", "copy", "adopt"))
    ap.add_argument("--session", required=True)
    ap.add_argument("--step", type=int)
    ap.add_argument("--models", nargs="+")
    a = ap.parse_args()
    P = json.load(open(f"{a.session}/plan.json"))
    {"verify": verify, "select": select, "copy": copy, "adopt": adopt}[a.cmd](a, P)


if __name__ == "__main__":
    main()

"""Apply the frozen E24 gates (plan.json, runbook 10.3) to the scores and print the comparison.

    python code/ops/e24_gates.py --session runs/E24_session_X [--out selection_draft.json]

Reads <session>/scores/: <M>_dev.metrics.json, <M>_dev_new.metrics.json, <M>_dev_bfcl.metrics.json
(code/eval_own.py score), <M>_dev_select.json (code/ops/score_part1.py), <M>_buyer_select.json
(code/eval_buyer_full.py) for M in E23, B0, B1, plus the paired comparisons it computes itself.
Text-rule hits are listed, not decided: a hit becomes a defect only after reading it.
"""
import argparse
import json
import os
import subprocess
import sys


def load(S, name):
    p = f"{S}/scores/{name}"
    return json.load(open(p)) if os.path.exists(p) else None


def p1(metrics, measure="exact"):
    t = metrics["totals"]["part1"]
    return 100 * t[measure] / t["steps"]


def bfcl_after_tool(metrics):
    c = [v for k, v in metrics["totals"].items() if k.startswith("part1 bfcl_") and k.endswith("call after tool")]
    return 100 * sum(x["exact"] for x in c) / sum(x["steps"] for x in c)


def pair(S, base, new):
    out = f"{S}/scores/pair_{new}_vs_{base}_select.json"
    if not os.path.exists(out):
        subprocess.run([sys.executable, "code/ops/score_part1.py", "pair", "--base", f"{S}/scores/{base}_dev_select.json",
                        "--new", f"{S}/scores/{new}_dev_select.json", "--out", out], check=True, stdout=subprocess.DEVNULL)
    return json.load(open(out))


def buyer_cmp(S, base, new):
    out = f"{S}/scores/cmp_{new}_vs_{base}_buyer.json"
    if not os.path.exists(out):
        subprocess.run([sys.executable, "code/eval_buyer_full.py", "compare", "--base", f"{S}/scores/{base}_buyer_select.json",
                        "--new", f"{S}/scores/{new}_buyer_select.json", "--out", out], check=True, stdout=subprocess.DEVNULL)
    return json.load(open(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    S = a.session
    rows, gates = {}, {}
    for m in ("E23", "B0", "B1"):
        need = [f"{m}_dev.metrics.json", f"{m}_dev_new.metrics.json", f"{m}_dev_bfcl.metrics.json", f"{m}_dev_select.json", f"{m}_buyer_select.json"]
        got = {n: load(S, n) for n in need}
        if any(v is None for v in got.values()):
            rows[m] = {"missing": [n for n, v in got.items() if v is None]}
            continue
        sel = got[f"{m}_dev_select.json"]["totals"]
        buy = got[f"{m}_buyer_select.json"]
        om = buy["official_metrics"]
        rows[m] = {"old_dev_P1": p1(got[f"{m}_dev.metrics.json"]), "rules_P1": p1(got[f"{m}_dev_new.metrics.json"]),
                   "bfcl_P1": p1(got[f"{m}_dev_bfcl.metrics.json"]), "bfcl_after_tool": bfcl_after_tool(got[f"{m}_dev_bfcl.metrics.json"]),
                   "select_must_call": 100 * sel["must_call"]["rate"], "select_must_call_n": f"{sel['must_call']['right']}/{sel['must_call']['steps']}",
                   "select_non_call": 100 * sel["non_call"]["rate"], "select_missing": got[f"{m}_dev_select.json"]["missing"],
                   "buyer_fill": 100 * om["fill_rate"], "buyer_order_acc": 100 * om["order_accuracy"], "buyer_orders": om["n_model_orders"],
                   "buyer_price_all": om["relative_price"], "buyer_complete": buy["complete"]}
    e = rows.get("E23", {})
    for m in ("B0", "B1"):
        r = rows.get(m, {})
        if "missing" in r or not r or "missing" in e:
            gates[m] = {"status": "incomplete"}
            continue
        bc = buyer_cmp(S, "E23", m)
        g = {"A_complete": r["select_missing"] == 0 and r["buyer_complete"],
             "B_old_dev": e["old_dev_P1"] - r["old_dev_P1"] <= 1.0,
             "B_rules": e["rules_P1"] - r["rules_P1"] <= 2.0,
             "B_non_call": e["select_non_call"] - r["select_non_call"] <= 2.0,
             "B_fill": e["buyer_fill"] - r["buyer_fill"] <= 1.0,
             "B_order_acc": e["buyer_order_acc"] - r["buyer_order_acc"] <= 0.5,
             "B_common_price": bc["common_price_change"] is not None and bc["common_price_change"] <= 0.005,
             "B_denominators": r["buyer_orders"] >= 50 and bc["common_deals"] >= 30}
        pe = pair(S, "E23", m)["must_call"]
        g["C_vs_E23"] = pe["diff"] * 100 >= 2.0 and pe["ci95"][0] > 0
        g["C_must_call_n"] = pe["steps"] >= 100
        detail = {"vs_E23_must_call": pe, "buyer_vs_E23": {k: bc[k] for k in ("n_gained", "n_lost", "common_deals", "common_price_change")}}
        if m == "B1" and "missing" not in rows.get("B0", {"missing": 1}):
            pb = pair(S, "B0", "B1")["must_call"]
            g["C_vs_B0"] = pb["diff"] * 100 >= 2.0 and pb["ci95"][0] > 0
            detail["vs_B0_must_call"] = pb
        g["A_and_B_pass"] = all(v for k, v in g.items() if k.startswith(("A_", "B_")))
        gates[m] = {"gates": g, "detail": detail}
    b1, b0 = gates.get("B1", {}).get("gates", {}), gates.get("B0", {}).get("gates", {})
    if b1 and b1["A_and_B_pass"] and b1.get("C_vs_E23") and b1.get("C_vs_B0") and b1["C_must_call_n"]:
        provisional = "B1"
    elif b0 and b0["A_and_B_pass"] and b0.get("C_vs_E23") and b0["C_must_call_n"]:
        provisional = "B0"
    else:
        provisional = "E23"
    out = {"rows": rows, "gates": gates, "provisional_candidate": provisional,
           "note": "text-rule hits must be read before the candidate is final; dev_confirm decides only if provisional is B0/B1"}
    print(json.dumps(rows, indent=1))
    print(json.dumps({m: v.get("gates", v) for m, v in gates.items()}, indent=1))
    print("provisional:", provisional)
    if a.out:
        json.dump(out, open(a.out, "x"), indent=1)


if __name__ == "__main__":
    main()

"""Apply the frozen E25 gates (plan section 9 with the user's pre-T0 settings) and write the decision.

    python code/ops/e25_gates.py select  --session S --out S/scores/provisional.json
    python code/ops/e25_gates.py confirm --session S --candidate C1 --out S/selection_gates.json

Reads <S>/scores/<M>_dev.metrics.json, <M>_dev_new.metrics.json, <M>_dev_bfcl.metrics.json
(code/eval_own.py score), <M>_buyer_select_new.json (code/eval_buyer_full.py score with the
official judge) and, for confirm, <M>_dev_confirm.metrics.json and <M>_buyer_confirm_new.json.
All thresholds come from <S>/plan.json "gates" (frozen before C0's first update), never from here.

select: F1 integrity, F2 Part 1 guards vs E23, F3 buyer guards vs E23, F4 the one pre-registered
improvement (fill_rate over all negotiations, paired family bootstrap, 95% two-sided interval
whose lower end must be > 0, and the difference >= min_pp). C1 also needs the same improvement
over C0 to be credited; if C1 fails, C0 can be provisional only by passing everything vs E23.
Otherwise the provisional candidate is E23 (no confirm run).
confirm: F5 for the single provisional candidate vs E23 on buyer_confirm_new and dev_confirm.
"""
import argparse
import json
import os
import random
import sys


def load(S, name):
    p = f"{S}/scores/{name}"
    return json.load(open(p)) if os.path.exists(p) else None


def p1(metrics):
    t = metrics["totals"]["part1"]
    return {"exact": 100 * t["exact"] / t["steps"], "right": t["exact"], "steps": t["steps"],
            "unparsed_pct": 100 * (t["steps"] - t["parsed"]) / t["steps"], "capped_pct": 100 * (t["steps"] - t["stopped"]) / t["steps"]}


def after_tool(metrics):
    c = [v for k, v in metrics["totals"].items() if k.startswith("part1 bfcl_") and k.endswith("call after tool")]
    return 100 * sum(x["exact"] for x in c) / sum(x["steps"] for x in c)


def buyer_row(b):
    om = b["official_metrics"]
    return {"fill": 100 * om["fill_rate"], "fill_opp": 100 * b["fill_rate_with_opportunity"], "order_acc": None if om["order_accuracy"] is None else 100 * om["order_accuracy"],
            "orders": om["n_model_orders"], "price_all": om["relative_price"], "invalid_per_neg": b["invalid_orders_per_negotiation"],
            "text_acc": 100 * om["text_accuracy"] if isinstance(om.get("text_accuracy"), (int, float)) else None, "unparsed_judgments": om.get("n_unparsed_judgments"),
            "text_coverage": 100 * b["text_coverage_gold_text_steps"], "complete": b["complete"], "judged": b["response"] == "official judge"}


def common(base, new):
    both = [k for k in base["negotiations"] if base["negotiations"][k]["deal"] and new["negotiations"].get(k, {}).get("deal")]
    if not both:
        return 0, None
    d = sum(new["negotiations"][k]["price_ratio"] - base["negotiations"][k]["price_ratio"] for k in both) / len(both)
    return len(both), d


def bootstrap_fill(base, new, draws, seed):
    """Paired: resample flows (families) with replacement, then negotiations inside each drawn flow;
    the same negotiations for both models. Returns (diff in pp, 2.5% and 97.5% quantiles)."""
    fams = {}
    for k, n in base["negotiations"].items():
        fams.setdefault(n["flow"], []).append(k)
    names = sorted(fams)
    rng = random.Random(seed)
    diffs = []
    for _ in range(draws):
        b = w = cnt = 0
        for f in (rng.choice(names) for _ in names):
            for k in (rng.choice(fams[f]) for _ in fams[f]):
                b += base["negotiations"][k]["deal"]
                w += new["negotiations"][k]["deal"]
                cnt += 1
        diffs.append(100 * (w - b) / cnt)
    diffs.sort()
    point = 100 * (sum(new["negotiations"][k]["deal"] for k in base["negotiations"]) - sum(n["deal"] for n in base["negotiations"].values())) / len(base["negotiations"])
    return point, diffs[int(0.025 * draws)], diffs[int(0.975 * draws) - 1]


def select(S, G):
    rows, buy = {}, {}
    for m in ("E23", "C0", "C1"):
        d = {n: load(S, f"{m}_{n}.metrics.json") for n in ("dev", "dev_new", "dev_bfcl")}
        b = load(S, f"{m}_buyer_select_new.json")
        if any(v is None for v in d.values()) or b is None:
            rows[m] = {"missing": True}
            continue
        rows[m] = {n: p1(v) for n, v in d.items()}
        rows[m]["bfcl_after_tool"] = after_tool(d["dev_bfcl"])
        rows[m]["mean3"] = sum(rows[m][n]["exact"] for n in ("dev", "dev_new", "dev_bfcl")) / 3
        rows[m]["buyer"] = buyer_row(b)
        buy[m] = b
    gates = {}
    g2, g3, g4 = G["F2"], G["F3"], G["F4"]
    e = rows.get("E23", {})
    for m in ("C0", "C1"):
        r = rows.get(m, {})
        if r.get("missing") or e.get("missing") or not r:
            gates[m] = {"status": "incomplete"}
            continue
        integ = json.load(open(f"{S}/scores/{m}_integrity.json")) if os.path.exists(f"{S}/scores/{m}_integrity.json") else {"passed": False, "why": "missing"}
        g = {"F1_integrity": integ["passed"] and r["buyer"]["complete"] and r["buyer"]["judged"] and e["buyer"]["judged"]}
        for n in ("dev", "dev_new", "dev_bfcl"):
            g[f"F2_{n}"] = e[n]["exact"] - r[n]["exact"] <= g2["p1_drop_pp"]
            g[f"F2_{n}_unparsed"] = r[n]["unparsed_pct"] - e[n]["unparsed_pct"] <= g2["problem_rise_pp"]
            g[f"F2_{n}_capped"] = r[n]["capped_pct"] - e[n]["capped_pct"] <= g2["problem_rise_pp"]
        g["F2_bfcl_after_tool"] = e["bfcl_after_tool"] - r["bfcl_after_tool"] <= g2["after_tool_drop_pp"]
        g["F2_mean3"] = r["mean3"] >= e["mean3"]
        rb, eb = r["buyer"], e["buyer"]
        n_common, dprice = common(buy["E23"], buy[m])
        g["F3_fill_all"] = rb["fill"] >= eb["fill"]
        g["F3_fill_opportunity"] = rb["fill_opp"] >= eb["fill_opp"]
        g["F3_order_acc"] = rb["order_acc"] is not None and eb["order_acc"] - rb["order_acc"] <= g3["order_acc_drop_pp"]
        g["F3_invalid_per_neg"] = rb["invalid_per_neg"] <= eb["invalid_per_neg"]
        g["F3_common_price"] = dprice is not None and dprice <= g3["common_price_worse"]
        g["F3_denominators"] = rb["orders"] >= g3["min_orders"] and n_common >= g3["min_common"]
        g["F3_text_acc"] = rb["text_acc"] is not None and eb["text_acc"] is not None and eb["text_acc"] - rb["text_acc"] <= g3["text_acc_drop_pp"]
        g["F3_text_coverage"] = eb["text_coverage"] - rb["text_coverage"] <= g3["text_coverage_drop_pp"]
        diff, lo, hi = bootstrap_fill(buy["E23"], buy[m], g4["draws"], g4["seed"])
        g["F4_fill_vs_E23"] = diff >= g4["min_pp"] and lo > 0
        detail = {"common_deals": n_common, "common_price_change": dprice, "fill_vs_E23": {"diff_pp": diff, "ci95": [lo, hi]}}
        if m == "C1" and not rows.get("C0", {}).get("missing"):
            d2, lo2, hi2 = bootstrap_fill(buy["C0"], buy["C1"], g4["draws"], g4["seed"])
            g["F4_fill_vs_C0"] = d2 >= g4["min_pp"] and lo2 > 0
            detail["fill_vs_C0"] = {"diff_pp": d2, "ci95": [lo2, hi2]}
        g["F1_to_F4_vs_E23"] = all(v for k, v in g.items() if k != "F4_fill_vs_C0")
        gates[m] = {"gates": g, "detail": detail}
    c1, c0 = gates.get("C1", {}).get("gates", {}), gates.get("C0", {}).get("gates", {})
    if c1 and c1["F1_to_F4_vs_E23"] and c1.get("F4_fill_vs_C0"):
        cand = "C1"
    elif c0 and c0["F1_to_F4_vs_E23"]:
        cand = "C0"
    else:
        cand = "E23"
    return {"rows": rows, "gates": gates, "provisional_candidate": cand}


def confirm(S, G, cand):
    g5 = G["F5"]
    e, c = load(S, "E23_buyer_confirm_new.json"), load(S, f"{cand}_buyer_confirm_new.json")
    ed, cd = load(S, "E23_dev_confirm.metrics.json"), load(S, f"{cand}_dev_confirm.metrics.json")
    if None in (e, c, ed, cd):
        return {"status": "incomplete", "selected": "E23"}
    eb, cb = buyer_row(e), buyer_row(c)
    n_common, dprice = common(e, c)
    diff, lo, hi = bootstrap_fill(e, c, G["F4"]["draws"], G["F4"]["seed"])
    ep, cp = p1(ed), p1(cd)
    fams = json.load(open(f"{S}/scores/dev_confirm_families.json"))["families"]
    g = {"complete": cb["complete"] and eb["complete"] and cb["judged"] and eb["judged"],
         "denominators": cb["orders"] >= g5["min_orders"] and n_common >= g5["min_common"],
         "fill_not_down": diff >= 0, "order_acc": cb["order_acc"] is not None and eb["order_acc"] - cb["order_acc"] <= g5["order_acc_drop_pp"],
         "common_price": dprice is not None and dprice <= g5["common_price_worse"],
         "text_acc": cb["text_acc"] is not None and eb["text_acc"] - cb["text_acc"] <= g5["text_acc_drop_pp"],
         "p1_not_negative": cp["exact"] - ep["exact"] >= 0,
         "p1_no_new_illegal": cp["unparsed_pct"] - ep["unparsed_pct"] <= g5["problem_rise_pp"] and cp["capped_pct"] - ep["capped_pct"] <= g5["problem_rise_pp"],
         "p1_denominators": cp["steps"] >= g5["p1_min_steps"] and fams >= g5["p1_min_families"]}
    passed = all(g.values())
    return {"candidate": cand, "gates": g, "passed": passed, "selected": cand if passed else "E23",
            "detail": {"buyer_E23": eb, "buyer_cand": cb, "common_deals": n_common, "common_price_change": dprice,
                       "fill_diff_pp": diff, "fill_ci95": [lo, hi], "p1_E23": ep, "p1_cand": cp, "dev_confirm_families": fams}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("select", "confirm"))
    ap.add_argument("--session", required=True)
    ap.add_argument("--candidate")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    G = json.load(open(f"{a.session}/plan.json"))["gates"]
    out = select(a.session, G) if a.mode == "select" else confirm(a.session, G, a.candidate)
    json.dump(out, open(a.out, "x"), indent=1)
    print(json.dumps(out.get("gates", out), indent=1)[:4000])
    print("decision:", out.get("provisional_candidate", out.get("selected")))


if __name__ == "__main__":
    main()

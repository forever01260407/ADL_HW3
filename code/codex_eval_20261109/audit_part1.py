"""CPU-only Part 1 audit. Imports the unchanged course scorer; never generates predictions.

Official input: preserve acceptable_next/defaults/first_step_eligible exactly.
Own input: select task=part1, use the first recorded gold call (no invented alternative
calls), derive first-step eligibility from the final visible user message, and use
explicit row defaults or top-level parameter defaults in tool schemas. This is an
official-rule adaptation of our labels, not proof that those labels are correct.

Counterfactuals are diagnostic score ceilings, NOT trained-model evaluations. In
particular, the oracle uses gold eligibility and must never be used for inference.
No public examples or substituted predictions are exported.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "hw3"))
from eval import format as F
from eval import score_tools as T


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def visible_after(row):
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{row.get('id')}: nonempty visible messages required")
    return messages[-1]["role"]


def adapt_own(row):
    after = visible_after(row)
    if row.get("after", after) != after:
        raise ValueError(f"{row['id']}: after disagrees with visible messages")
    calls = row["gold"]["calls"]
    acceptable = []
    if calls:
        first = calls[0]
        args = first["arguments"]
        args = json.loads(args) if isinstance(args, str) else args
        if not isinstance(args, dict):
            raise ValueError(f"{row['id']}: gold arguments must be an object")
        acceptable = [{"name": first["name"], "arguments": args}]
    if "defaults" in row:
        defaults = row["defaults"] or {}
    else:
        defaults = {}
        for tool in row.get("tools", []):
            function = tool.get("function", tool)
            props = function.get("parameters", {}).get("properties", {})
            values = {k: v["default"] for k, v in props.items() if "default" in v}
            if values:
                defaults[function["name"]] = values
    return {"id": row["id"], "messages": row["messages"],
            "acceptable_next": acceptable, "defaults": defaults,
            "first_step_eligible": bool(acceptable and after == "user")}


def group_of(row):
    if not row["is_call"]:
        return "no_call"
    return "eligible_first_call" if row["elig"] else "must_call"


def error_of(row, pred, parser):
    if row["step"]:
        return "correct"
    if row["missing"]:
        return "missing"
    if row["capped"]:
        return "token_cap"
    if row["invalid"]:
        return "invalid"
    reply = F.read_reply(pred, parser)
    if row["is_call"]:
        if not reply["calls"]:
            return "text_or_empty_instead_of_call"
        return "wrong_arguments" if row["tool"] else "wrong_tool"
    return "unwanted_call"


def legacy_comparison(raw_rows, preds, parser):
    """Keep the project's old exact metric visible; distinguish default handling from first-step credit."""
    old_right = base_right = credited_right = 0
    changes = collections.Counter()
    for row in raw_rows:
        if row.get("task") != "part1":
            continue
        reply = F.read_reply(preds.get(row["id"]), parser)
        usable = reply is not None and not reply["invalid"]
        calls = reply["calls"] if usable else []
        wanted = row["gold"]["calls"]
        if wanted:
            old = bool(usable and calls and T._same_call(T._canon(wanted[0], {}), T._canon(calls[0], {})))
        else:
            old = bool(usable and not calls and reply["text"].strip())
        new = T.score_step(adapt_own(row), preds.get(row["id"]), parser)
        credited = T.credit(new, new["elig"], new["replied"])
        old_right += old
        base_right += new["step"]
        credited_right += credited["step"]
        if old != new["step"]:
            changes["call_comparison_change" if wanted else "no_call_empty_text_rule_change"] += 1
    n = sum(r.get("task") == "part1" for r in raw_rows)
    return {"n": n, "legacy_exact_pct": 100 * old_right / n,
            "official_comparison_unweighted_without_first_rule_pct": 100 * base_right / n,
            "official_comparison_unweighted_with_first_rule_pct": 100 * credited_right / n,
            "comparison_rule_changes_before_first_credit": dict(changes),
            "note": "Different metrics on the SAME predictions; none is a newly trained model improvement."}


def analyze(rows, preds, parser="lenient"):
    if not rows:
        raise ValueError("no Part 1 rows")
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate data ids")
    # Require visible histories for the deployable-position counterfactual.
    for row in rows:
        visible_after(row)
    result, scored = T.score_tools(rows, preds, parser)
    baseline = result["with_first_step_rule"]["all"]["weighted_step_acc"]
    weight = F.settings()["must_call_weight"]
    denominator = sum(weight if T.must_call(r) else 1 for r in scored)
    groups = collections.defaultdict(list)
    errors = collections.defaultdict(collections.Counter)
    for row in scored:
        group = group_of(row)
        groups[group].append(row)
        errors[group][error_of(row, preds.get(row["id"]), parser)] += 1
    summaries = {}
    for group, items in groups.items():
        lost = sum((weight if T.must_call(r) else 1) for r in items if not r["step"])
        summaries[group] = {"n": len(items), "right": sum(r["step"] for r in items),
                            "accuracy": sum(r["step"] for r in items) / len(items),
                            "recoverable_headline_pp": 100 * lost / denominator,
                            "errors": dict(errors[group])}
    scenarios = {}
    for name in ("oracle_eligible_calls_only", "visible_after_user_all_steps"):
        modified = []
        changed = 0
        for gold, previous in zip(rows, scored):
            replace = (bool(gold.get("first_step_eligible")) and bool(gold["acceptable_next"])) if name.startswith("oracle") else visible_after(gold) == "user"
            if replace:
                raw = T.score_step(gold, {"id": gold["id"], "output": "I will check that.", "finish_reason": "stop"}, parser)
                modified.append(T.credit(raw, raw["elig"], raw["replied"]))
                changed += 1
            else:
                modified.append(previous)
        headline = T.summarize(modified)["all"]["weighted_step_acc"]
        gained = sum(not a["step"] and b["step"] for a, b in zip(scored, modified))
        lost = sum(a["step"] and not b["step"] for a, b in zip(scored, modified))
        remaining = max(0.0, (0.70 - headline) * denominator)
        must_errors = sum(T.must_call(r) and not r["step"] for r in modified)
        needed = math.ceil(remaining / weight - 1e-10)
        scenarios[name] = {"diagnostic_only_not_a_model_result": True,
                           "replaced_steps": changed, "gained_steps": gained, "lost_steps": lost,
                           "headline_pct": 100 * headline, "delta_pp": 100 * (headline - baseline),
                           "must_call_fixes_to_70_if_everything_else_unchanged": needed,
                           "enough_remaining_must_call_errors": needed <= must_errors}
    return {"parser": parser, "official_result": result,
            "headline_pct": 100 * baseline, "weighted_denominator": denominator,
            "groups": summaries, "counterfactuals": scenarios,
            "warnings": ["Counterfactuals replace outputs offline, assume no training spillover, and are not measured model gains.",
                         "Gold first_step_eligible is evaluation metadata, never an inference feature.",
                         "Own data can define fewer acceptable calls than an independently annotated benchmark."]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--format", choices=["official", "own"], required=True)
    ap.add_argument("--parser", choices=["lenient", "strict"], default="lenient")
    ap.add_argument("--out", required=True, help="New JSON path; existing output is never overwritten")
    args = ap.parse_args()
    if Path(args.out).exists():
        raise SystemExit("output exists; choose a new output path")
    before = {k: file_hash(v) for k, v in {"data": args.data, "pred": args.pred}.items()}
    raw = list(F.iter_jsonl(args.data))
    rows = [adapt_own(r) for r in raw if r.get("task") == "part1"] if args.format == "own" else raw
    preds, duplicates = F.load_predictions([args.pred])
    if duplicates:
        raise SystemExit("duplicate prediction ids; refusing ambiguous audit")
    report = analyze(rows, preds, args.parser)
    if args.format == "own":
        report["legacy_comparison"] = legacy_comparison(raw, preds, args.parser)
    after = {k: file_hash(v) for k, v in {"data": args.data, "pred": args.pred}.items()}
    if before != after:
        raise SystemExit("inputs changed during audit; no report written")
    selected = {r["id"] for r in rows}
    report["provenance"] = {"data": str(args.data), "pred": str(args.pred), "format": args.format,
                            "sha256": before, "audit_sha256": file_hash(__file__),
                            "scorer_sha256": file_hash(T.__file__), "reader_sha256": file_hash(F.__file__),
                            "input_rows": len(raw), "selected_part1_rows": len(rows),
                            "missing_predictions": len(selected - set(preds)),
                            "predictions_outside_selected_rows": len(set(preds) - selected)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({"out": args.out, "headline_pct": report["headline_pct"],
                      "groups": report["groups"], "counterfactuals": report["counterfactuals"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

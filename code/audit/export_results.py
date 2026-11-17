"""Export aggregate historical metrics and hashes, never benchmark conversations.

python code/audit/export_results.py --root /path/to/HW3_ADL \
  --submission-zip /path/to/submission.zip --out /path/to/docs/results.json

Only existing, complete metrics are exported. This runs on CPU and never starts
generation, training, or the judge. Absolute local paths are not published.
"""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

RUNS = {
    "E02": "runs/E02",
    "E10": "runs/E10/full_1250",
    "E23": "runs/E23/full_1475",
    "E27": "runs/E27_session_20261109_165030/final_public",
    "E28": "runs/E28_session_20261109_182859/info_public",
    "E29": "runs/E29_session_20261109_202924/final_public",
    "E30_template": "runs/E30_session_20261109_215120/public",
    "E30_written_policy": "runs/E30_session_20261109_220121/public",
    "E30_late": "runs/E30_session_20261109_230529/public",
}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def export(root, zip_path=None):
    result = {"schema_version": 1, "evaluation": "course-provided public benchmarks; gold-history per-step evaluation",
              "expected_part1_steps": 13021, "expected_negotiation_steps": 8988,
              "expected_negotiations": 2607, "submitted_model": "E29",
              "private_results": "not verified", "runs": {}}
    for tag, relative in RUNS.items():
        folder = root / relative
        tools = folder / "metrics/tools.json"
        if not tools.exists():
            continue
        raw = json.loads(tools.read_text(encoding="utf-8"))
        scorer = raw.get("headline", raw)
        values = scorer["with_first_step_rule"]["all"]
        if values["n"] != 13021 or scorer.get("missing_predictions", 0):
            continue
        row = {"part1_weighted_accuracy": values["weighted_step_acc"],
               "part1_steps": values["n"], "tools_metrics_sha256": digest(tools),
               "provenance": str(Path(relative) / "metrics/tools.json")}
        neg = folder / "metrics/negotiation.json"
        if not neg.exists():
            neg = folder / "metrics/negotiation_no_judge.json"
        if neg.exists():
            n = json.loads(neg.read_text(encoding="utf-8"))
            m = n["metrics"]
            if n.get("n_missing_predictions", 0) == 0 and m["n_steps"] == 8988:
                row["negotiation"] = {key: m.get(key) for key in ("n_negotiations", "n_steps", "n_model_orders", "order_accuracy", "fill_rate", "relative_price", "text_accuracy")}
                row["negotiation_metrics_sha256"] = digest(neg)
        result["runs"][tag] = row
    if zip_path:
        with ZipFile(zip_path) as archive:
            names = [n for n in archive.namelist() if n.endswith("adapter_checkpoint/adapter_model.safetensors")]
            if len(names) != 1:
                raise ValueError("Expected exactly one submitted adapter")
            h = hashlib.sha256()
            with archive.open(names[0]) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
            result["submission_adapter_sha256"] = h.hexdigest()
            result["submission_archive_sha256"] = digest(zip_path)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--submission-zip", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = export(args.root, args.submission_zip)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

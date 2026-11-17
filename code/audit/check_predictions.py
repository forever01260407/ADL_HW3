"""Read-only structural audit of official generate JSONL output; standard library only.

Use --in-progress while generation is running. Missing IDs and an unfinished
last line are then pending, not failures. This does not score answers or prove
which model produced them. No conversation, answer text, or IDs are reported.
"""
import argparse
import collections
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit(data, pred, in_progress=False):
    data, pred = Path(data), Path(pred)
    opener = gzip.open if data.suffix == ".gz" else open
    expected = set()
    with opener(data, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = row.get("id")
            if not isinstance(key, str) or not key or key in expected:
                raise ValueError("Input data must have unique, nonempty string IDs")
            expected.add(key)
    if not expected:
        raise ValueError("Input data has no records")

    # Bound the read to the size seen at open; no locks or changes to the writer.
    try:
        with open(pred, "rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            raw = stream.read(size)
        exists = True
    except FileNotFoundError:
        size, raw, exists = 0, b"", False
    snapshot = raw
    pending_tail = 0
    if in_progress and raw and not raw.endswith(b"\n"):
        boundary = raw.rfind(b"\n") + 1
        pending_tail = len(raw) - boundary
        raw = raw[:boundary]

    errors = collections.Counter()
    counts = collections.Counter()
    reasons = collections.Counter()
    parsed = 0
    for line in raw.splitlines():
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            errors["malformed_json_lines"] += 1
            continue
        parsed += 1
        if not isinstance(row, dict):
            errors["non_object_rows"] += 1
            continue
        key = row.get("id")
        if not isinstance(key, str) or not key:
            errors["invalid_id_rows"] += 1
        else:
            counts[key] += 1
        if not isinstance(row.get("output"), str):
            errors["invalid_output_rows"] += 1
        reason = row.get("finish_reason")
        if not isinstance(reason, str) or reason not in {"stop", "length", "overlength"}:
            errors["invalid_finish_reason_rows"] += 1
        else:
            reasons[reason] += 1
    duplicates = sum(n - 1 for n in counts.values())
    unknown = sum(n for key, n in counts.items() if key not in expected)
    if duplicates:
        errors["duplicate_id_rows"] = duplicates
    if unknown:
        errors["unknown_id_rows"] = unknown
    missing = len(expected - counts.keys())
    if not in_progress:
        if not exists:
            errors["prediction_file_missing"] = 1
        if missing:
            errors["missing_ids"] = missing
    if len(snapshot) != size:
        errors["file_shrank_during_read"] = 1
    status = "fail" if errors else "pending" if missing or pending_tail else "pass"
    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "structure_and_id_coverage_only",
        "mode": "in_progress" if in_progress else "complete",
        "status": status,
        "data_path": str(data.resolve()),
        "data_sha256": digest(data),
        "prediction_path": str(pred.resolve()),
        "prediction_file_exists": exists,
        "snapshot_bytes": len(snapshot),
        "snapshot_sha256": hashlib.sha256(snapshot).hexdigest(),
        "deferred_tail_bytes": pending_tail,
        "expected_ids": len(expected),
        "parsed_rows": parsed,
        "covered_ids": len(expected & counts.keys()),
        "missing_ids": missing,
        "finish_reasons": dict(reasons),
        "errors": dict(errors),
        "limitations": [
            "Does not verify model identity, scoring, or training-data compliance.",
            "Snapshot assumes append-only writes; concurrent replacement is not supported.",
            "Length and overlength are structurally valid but may be scored wrong.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--in-progress", action="store_true")
    parser.add_argument("--out", help="New audit report path; never overwrites an existing file")
    args = parser.parse_args()
    result = audit(args.data, args.pred, args.in_progress)
    result["checker_sha256"] = digest(__file__)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    print(payload, end="")
    return 1 if result["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())

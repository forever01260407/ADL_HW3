"""Audit BFCL family exclusion without reading or exposing public examples.

The collision report is produced by the isolated public deduplicator. A collision
in any multi-turn variant excludes its entire task family, including plain/state
variants and individual expanded steps. Non-multi-turn IDs are not grouped by
their trailing number.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


FAMILY_ID = re.compile(r"multi_turn_(?:base|miss_param|miss_func|long_context|composite)_(\d+)(?::(?:plain|state))?(?:#\d+)?\Z")


def family_of(record_id):
    match = FAMILY_ID.fullmatch(str(record_id))
    return f"bfcl_task_{int(match[1])}" if match else None


def excluded_families(collision_report):
    excluded = set()
    for record_id in collision_report["drop_ids"]:
        family = family_of(record_id)
        if str(record_id).startswith("multi_turn_") and family is None:
            raise ValueError(f"unrecognized BFCL multi-turn ID: {record_id}")
        if family:
            excluded.add(family)
    return excluded


def target_count(record):
    skip = set(record.get("no_target", []))
    return sum(i > 0 and m["role"] == "assistant" and i not in skip
               for i, m in enumerate(record["messages"]))


def audit_file(path, blocked):
    path = Path(path)
    raw = path.read_bytes()
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    hits, mismatches = [], []
    families = set()
    for record in records:
        family = family_of(record["id"])
        if family is None:
            if str(record.get("source", "")).startswith("bfcl_"):
                raise ValueError(f"unrecognized BFCL record ID: {record['id']}")
            continue
        families.add(family)
        if record.get("family") != family:
            mismatches.append(record["id"])
        if family in blocked:
            hits.append({"id": record["id"], "family": family,
                         "target_steps": target_count(record)})
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "records": len(records), "bfcl_families": sorted(families),
            "family_metadata_mismatches": mismatches,
            "excluded_family_records": hits,
            "excluded_family_target_steps": sum(r["target_steps"] for r in hits)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--collisions", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    raw = Path(args.collisions).read_bytes()
    blocked = excluded_families(json.loads(raw))
    reports = [audit_file(path, blocked) for path in args.files]
    result = {"collision_report": args.collisions,
              "collision_report_sha256": hashlib.sha256(raw).hexdigest(),
              "excluded_families": sorted(blocked), "files": reports,
              "public_content_read": False,
              "limitations": "Family closure of the existing collision list only; not a semantic deduplication or a label-quality certificate."}
    with Path(args.out).open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, ensure_ascii=False)
        out.write("\n")
    failed = any(r["excluded_family_records"] or r["family_metadata_mismatches"] for r in reports)
    print(json.dumps({"passed": not failed, "excluded_families": sorted(blocked),
                      "files": [{k: r[k] for k in ("path", "excluded_family_records", "excluded_family_target_steps")}
                                for r in reports]}, ensure_ascii=False))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()

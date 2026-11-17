"""Make a NEW candidate copy with all public-collision task families removed.

This fixes family exclusion only. It does not certify parameter semantics, hidden
state, or the labels after state augmentation. Existing frozen files stay intact.
The destination must not exist. No third-party tool code is imported or executed.
"""
import argparse
import hashlib
import json
from pathlib import Path

from bfcl_family_guard import excluded_families, family_of, target_count


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--collisions", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    source, dest = Path(args.source).resolve(), Path(args.out_dir).resolve()
    if dest.exists():
        raise SystemExit(f"refusing existing destination: {dest}")
    collision_raw = Path(args.collisions).read_bytes()
    blocked = excluded_families(json.loads(collision_raw))
    prepared, manifest = {}, {}
    for name in ("train.jsonl", "dev.jsonl", "dev_new.jsonl", "dev_bfcl.jsonl"):
        original = (source / name).read_bytes()
        kept, removed = [], []
        for line in original.splitlines(keepends=True):
            if not line.strip():
                kept.append(line)
                continue
            record = json.loads(line)
            family = family_of(record["id"])
            if family is None and str(record.get("source", "")).startswith("bfcl_"):
                raise ValueError(f"unrecognized BFCL ID: {record['id']}")
            if family is not None and record.get("family") != family:
                raise ValueError(f"inconsistent family metadata: {record['id']}")
            if family in blocked:
                removed.append({"id": record["id"], "family": family,
                                "target_steps": target_count(record)})
            else:
                kept.append(line)
        candidate = b"".join(kept)
        prepared[name] = candidate
        manifest[name] = {"source_sha256": sha(original), "candidate_sha256": sha(candidate),
                          "removed_records": removed,
                          "removed_target_steps": sum(r["target_steps"] for r in removed)}
    # Prepare and validate every input before creating any destination files.
    dest.mkdir(parents=True, exist_ok=False)
    for name, raw in prepared.items():
        with (dest / name).open("xb") as out:
            out.write(raw)
    report = {"status": "candidate_only_pending_semantic_and_state_label_review",
              "source": str(source), "collision_report": args.collisions,
              "collision_report_sha256": sha(collision_raw),
              "excluded_families": sorted(blocked), "files": manifest,
              "public_content_read": False, "third_party_tools_executed": False}
    with (dest / "manifest.json").open("x", encoding="utf-8") as out:
        json.dump(report, out, indent=2)
        out.write("\n")
    print(json.dumps({"out_dir": str(dest), "status": report["status"],
                      "removed_targets": {name: item["removed_target_steps"] for name, item in manifest.items()}}))


if __name__ == "__main__":
    main()

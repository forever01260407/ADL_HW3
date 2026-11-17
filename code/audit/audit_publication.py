"""CPU-only review of an explicit Git file list before publication.

This never reads credential stores. It reports paths, not matched secret contents.
It checks Python syntax, JSON, local Markdown links, disallowed artifacts and common
credential prefixes. It is a bounded check, not a guarantee against every secret.
"""
import argparse
import ast
import json
from pathlib import Path
import re

SECRET = re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{25,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
FORBIDDEN = {"hw3", "runs", "release", "data_work", "submission", "submission_draft", "tmp", "scratchpad", ".claude", ".codex", ".vscode", "__pycache__"}
BINARY_SUFFIXES = {".safetensors", ".pt", ".pth", ".bin", ".ckpt", ".zip", ".webm", ".mp4", ".pyc", ".pdf"}


def audit(root, files):
    errors = []
    total = 0
    largest = (0, "")
    checked = {"python": 0, "json": 0, "local_links": 0}
    for name in files:
        rel = Path(name)
        path = root / rel
        if rel.is_absolute() or ".." in rel.parts or set(rel.parts) & FORBIDDEN or rel.suffix.lower() in BINARY_SUFFIXES:
            errors.append({"path": name, "reason": "excluded path or artifact"})
            continue
        if not path.is_file() or path.is_symlink():
            errors.append({"path": name, "reason": "not a regular file"})
            continue
        raw = path.read_bytes()
        total += len(raw)
        largest = max(largest, (len(raw), name))
        if len(raw) > 2 * 1024 * 1024:
            errors.append({"path": name, "reason": "file exceeds 2 MiB publication budget"})
        try:
            text = raw.decode("utf-8-sig")
            if SECRET.search(text):
                errors.append({"path": name, "reason": "potential credential; inspect locally"})
            if path.suffix == ".py":
                ast.parse(text, filename=name)
                checked["python"] += 1
            if path.suffix == ".json":
                json.loads(text)
                checked["json"] += 1
            if name == "README.md" or name.startswith("docs/") and path.suffix == ".md":
                for target in re.findall(r"\]\(([^)]+)\)", text):
                    if "://" in target or target.startswith(("#", "mailto:")):
                        continue
                    target = target.split("#", 1)[0]
                    linked = path.parent / target
                    if not linked.exists():
                        errors.append({"path": name, "reason": "missing local link", "target": target})
                    else:
                        checked["local_links"] += 1
        except (UnicodeError, SyntaxError, ValueError) as error:
            errors.append({"path": name, "reason": type(error).__name__})
    return {"status": "passed" if not errors else "failed", "files": len(files), "bytes": total,
            "largest_file": {"path": largest[1], "bytes": largest[0]}, "checked": checked, "errors": errors}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--file-list", type=Path, required=True, help="UTF-8 JSON array from git ls-files")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    result = audit(args.root, json.loads(args.file_list.read_text(encoding="utf-8-sig")))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    print(json.dumps(result))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()

"""Download the three models at a fixed revision and record the revisions.

    python code/env/download_models.py [--skip-judge]

Only the tokenizer and config files of Qwen/Qwen3-0.6B are needed: the model we
train on is Qwen/Qwen3-0.6B-Base.
"""
import argparse
import json
import os

from huggingface_hub import HfApi, snapshot_download

TOKENIZER_FILES = ["*.json", "*.txt", "*.jinja"]
MODELS = [
    ("Qwen/Qwen3-0.6B-Base", None),
    ("Qwen/Qwen3-0.6B", TOKENIZER_FILES),
    ("Qwen/Qwen3-8B", None),
]
OUTPUT = "records/model_revisions.json"


def size_gib(path):
    total = 0
    for folder, _, names in os.walk(path):
        for name in names:
            # Snapshot entries are links into the blob store, so follow them.
            total += os.path.getsize(os.path.join(folder, name))
    return total / 2**30


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-judge", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    revisions = {}
    if os.path.exists(OUTPUT):
        with open(OUTPUT, encoding="utf-8") as f:
            revisions = json.load(f)

    for repo, patterns in MODELS:
        if args.skip_judge and repo == "Qwen/Qwen3-8B":
            continue
        pinned = revisions.get(repo, {}).get("revision")
        revision = pinned or api.model_info(repo).sha
        path = snapshot_download(repo, revision=revision, allow_patterns=patterns)
        revisions[repo] = {"revision": revision, "size_gib": round(size_gib(path), 2), "partial": patterns is not None}
        print(repo, revisions[repo], flush=True)
        with open(OUTPUT, "w", encoding="utf-8") as f:
            json.dump(revisions, f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()

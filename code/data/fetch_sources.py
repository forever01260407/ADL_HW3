"""Download the candidate public tool-calling datasets at a fixed revision.

    python code/data/fetch_sources.py

The files go to data_work/raw/<name>/ and the revisions to records/data_revisions.json.
"""
import json
import os

from huggingface_hub import HfApi, snapshot_download

SOURCES = {
    "bfcl": "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
    "glaive": "glaiveai/glaive-function-calling-v2",
    "hermes": "NousResearch/hermes-function-calling-v1",
    "toolace": "Team-ACE/ToolACE",
}
OUTPUT = "records/data_revisions.json"


def main():
    api = HfApi()
    revisions = {}
    if os.path.exists(OUTPUT):
        with open(OUTPUT, encoding="utf-8") as f:
            revisions = json.load(f)

    for name, repo in SOURCES.items():
        revision = revisions.get(name, {}).get("revision") or api.dataset_info(repo).sha
        path = snapshot_download(repo, repo_type="dataset", revision=revision, local_dir=f"data_work/raw/{name}")
        revisions[name] = {"repo": repo, "revision": revision}
        print(name, revision, path, flush=True)
        with open(OUTPUT, "w", encoding="utf-8") as f:
            json.dump(revisions, f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()

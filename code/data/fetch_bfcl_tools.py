"""Download BFCL's executable tool classes at a pinned commit.

    python -I code/data/fetch_bfcl_tools.py

The Hugging Face dataset holds only the tool documentation. Running the benchmark's
own tools is what gives the multi-turn conversations real tool results. The code stays
in data_work/raw/bfcl_tools and is not part of the submission: anyone rebuilding the
data fetches this same commit.
"""
import json
import os
import urllib.request

REPO = "ShishirPatil/gorilla"
COMMIT = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
FOLDER = "berkeley-function-call-leaderboard/bfcl_eval/eval_checker/multi_turn_eval/func_source_code"
OUT = "data_work/raw/bfcl_tools"
REVISIONS = "records/data_revisions.json"


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "hw3-data-fetch"}), timeout=60) as response:
        return response.read()


def main():
    os.makedirs(OUT, exist_ok=True)
    listing = json.loads(fetch(f"https://api.github.com/repos/{REPO}/contents/{FOLDER}?ref={COMMIT}"))
    for item in listing:
        if item["type"] == "file" and item["name"].endswith(".py"):
            body = fetch(f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{FOLDER}/{item['name']}")
            with open(os.path.join(OUT, item["name"]), "wb") as f:
                f.write(body)
            print(f"{len(body):8d} {item['name']}")

    with open(REVISIONS, encoding="utf-8") as f:
        revisions = json.load(f)
    revisions["bfcl_tools"] = {"repo": f"https://github.com/{REPO}", "commit": COMMIT, "folder": FOLDER, "license": "Apache-2.0"}
    with open(REVISIONS, "w", encoding="utf-8") as f:
        json.dump(revisions, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()

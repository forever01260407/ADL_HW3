"""Summarize the structure of the downloaded datasets.

    python code/data/inspect_sources.py

Prints file sizes, record counts and field names.
"""
import json
import os

RAW = "data_work/raw"
EXTENSIONS = (".json", ".jsonl", ".parquet", ".txt")


def read_records(path):
    """Return (records, note). Handles JSON lines, a JSON array and parquet."""
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist(), "parquet"
    with open(path, encoding="utf-8") as f:
        text = f.read()
    try:
        data = json.loads(text)
        return (data if isinstance(data, list) else [data]), "json"
    except json.JSONDecodeError:
        pass
    records, bad = [], 0
    for line in text.splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    return records, f"jsonl, {bad} bad lines"


def main():
    for source in sorted(os.listdir(RAW)):
        print(f"\n########## {source}")
        for folder, dirs, names in os.walk(os.path.join(RAW, source)):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for name in sorted(names):
                if not name.endswith(EXTENSIONS):
                    continue
                path = os.path.join(folder, name)
                size = os.path.getsize(path) / 1e6
                try:
                    records, note = read_records(path)
                except Exception as error:
                    print(f"{path[len(RAW) + 1:]}: {size:.1f} MB, unreadable: {type(error).__name__}")
                    continue
                keys = sorted(records[0]) if records and isinstance(records[0], dict) else type(records[0]).__name__ if records else "-"
                print(f"{path[len(RAW) + 1:]}: {size:.1f} MB, {len(records)} records ({note}), fields {keys}")


if __name__ == "__main__":
    main()

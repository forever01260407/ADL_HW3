"""Find own training records whose user messages also appear in the public test.

    python -I code/data/dedupe_against_public.py --out data_work/processed/collisions.json FILE [FILE ...]

Each FILE is JSON lines holding `messages` (a flat list) or BFCL's `question` (a list
of turns). Public text stays inside this process: the output is collision counts and
the ids of OUR records to drop, nothing else. Passing does not prove the absence of
paraphrases, it only rules out identical and heavily overlapping messages.
"""
import argparse
import collections
import gzip
import json
import os
import re
import unicodedata

PUBLIC = ("hw3/data/public_test.jsonl.gz", "hw3/data/public_test_negotiation.jsonl.gz")
SHINGLE = 5
NEAR = 0.6


def words(text):
    return re.findall(r"\w+", unicodedata.normalize("NFKC", text).lower())


def shingles(tokens):
    return {hash(tuple(tokens[i:i + SHINGLE])) for i in range(len(tokens) - SHINGLE + 1)}


OFFER_FRAME = None  # E26 (--offer-frame-as-format): the interface's Offer line prefix is format, not content


def without_engine_lines(text):
    """Part 2 wraps seller text in fixed JSON lines. They are format, not content, and would
    match on every negotiation, so they are left out on both sides."""
    kept = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith(('{"phase"', '{"from_id"')):
            continue
        if OFFER_FRAME is not None:
            line = OFFER_FRAME.sub("", line)
        kept.append(line)
    return "\n".join(kept)


def user_texts(messages):
    texts = (without_engine_lines(m["content"]) for m in messages if m.get("role") == "user" and isinstance(m.get("content"), str))
    return [text for text in texts if text.strip()]


def load_public():
    exact, grams = set(), set()
    for path in PUBLIC:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                for text in user_texts(json.loads(line)["messages"]):
                    tokens = words(text)
                    exact.add(hash(tuple(tokens)))
                    grams |= shingles(tokens)
    return exact, grams


def flatten(question):
    """BFCL's `question` is a string, a list of messages, or a list of turns."""
    if isinstance(question, str):
        return [{"role": "user", "content": question}]
    if isinstance(question, dict):
        return [question]
    return [message for item in question for message in flatten(item)]


def own_records(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    with open(path, encoding="utf-8") as f:
        for index, line in enumerate(f):
            row = json.loads(line)
            messages = row.get("messages")
            if messages is None:
                messages = flatten(row["question"])
            yield str(row.get("id", f"{stem}_{index}")), user_texts(messages)


def verdict(texts, exact, grams):
    """Worst match over a record's user messages."""
    result = "clean"
    for text in texts:
        tokens = words(text)
        if hash(tuple(tokens)) in exact:
            return "exact"
        mine = shingles(tokens)
        if mine and len(mine & grams) / len(mine) >= NEAR:
            result = "near"
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--offer-frame-as-format", action="store_true",
                    help="E26: drop the fixed 'Offer: <id> at $<price> per unit, qty <n>.' prefix on both sides (text after it is still compared)")
    args = ap.parse_args()
    if args.offer_frame_as_format:
        global OFFER_FRAME
        OFFER_FRAME = re.compile(r"^\s*Offer: \S+ at \$\d+\.\d{2} per unit, qty \d+\.\s*")

    exact, grams = load_public()
    print(f"public: {len(exact)} distinct user messages")

    report, drop = {}, []
    print(f"{'file':45s} {'records':>7s} {'exact':>6s} {'near':>6s}")
    for path in args.files:
        counts = collections.Counter()
        for record_id, texts in own_records(path):
            kind = verdict(texts, exact, grams)
            counts[kind] += 1
            if kind != "clean":
                drop.append(record_id)
        total = sum(counts.values())
        name = os.path.basename(path)
        report[name] = {"records": total, "exact": counts["exact"], "near": counts["near"]}
        print(f"{name:45s} {total:7d} {counts['exact']:6d} {counts['near']:6d}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"shingle": SHINGLE, "near_threshold": NEAR, "offer_frame_as_format": bool(args.offer_frame_as_format), "files": report, "drop_ids": sorted(set(drop))}, f, indent=1)


if __name__ == "__main__":
    main()

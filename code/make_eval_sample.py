"""Write the fixed public id lists used for the learning curve and quick checks.

    python code/make_eval_sample.py

Whole conversations are drawn, since a negotiation cut short changes deal price and deal
volume, and every point of a curve must use the same list. Prints counts only, and compares
the mix of step types with the full file because must-call steps carry most of the score.
The same seed gives the same lists. Run from the project root.
"""
import collections
import random
import sys

sys.path.insert(0, "hw3")

from eval import format as F  # noqa: E402

SEED = 20261107
PART1_EPISODES = 200
PART2_NEGOTIATIONS = 400
OUT = "data_work/frozen"


def group_by(rows, key):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return groups


def draw(rng, groups, count):
    names = sorted(groups)
    return [row for name in rng.sample(names, count) for row in groups[name]]


def part1_mix(rows):
    """Share of the score weight on must-call steps (score_tools.must_call), and of text steps."""
    must = sum(row["kind"] == "call" and not row["first_step_eligible"] for row in rows)
    text = sum(row["kind"] == "text" for row in rows)
    weight = F.settings()["must_call_weight"]
    total = must * weight + len(rows) - must
    return f"{len(rows)} steps, must-call {must} ({must * weight / total:.1%} of the weight), text {text / len(rows):.1%}"


def write_ids(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(row["id"] + "\n")


def main():
    rng = random.Random(SEED)
    part1 = F.read_jsonl(F.DATA_TOOLS)
    sample1 = draw(rng, group_by(part1, "episode"), PART1_EPISODES)
    print("part 1 full  :", part1_mix(part1))
    print("part 1 sample:", part1_mix(sample1), f"from {PART1_EPISODES} episodes")
    write_ids(f"{OUT}/eval_ids_tools.txt", sample1)

    part2 = F.read_jsonl(F.DATA_NEGOTIATION)
    sample2 = draw(rng, group_by(part2, "negotiation"), PART2_NEGOTIATIONS)
    print(f"part 2 full  : {len(part2)} steps")
    print(f"part 2 sample: {len(sample2)} steps from {PART2_NEGOTIATIONS} negotiations")
    write_ids(f"{OUT}/eval_ids_negotiation.txt", sample2)
    print("seed", SEED, "->", OUT)


if __name__ == "__main__":
    main()

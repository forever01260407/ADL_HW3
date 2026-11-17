"""STEP 05 acceptance check for the training code.

    python code/check_train_setup.py --out runs/E01_setup

Uses the same functions as train.py. Checks prompts against the official renderer, labels,
padding, which parameters train, that one update changes what it should and nothing else,
and that the saved adapter survives a fresh load and the official merge. Prints 12 examples
for reading. Exits non-zero on the first failed check. Run from the project root.
"""
import argparse
import json
import random
import sys

import torch
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoModelForCausalLM

import model_setup as M

CONFIG = {"r": 32, "alpha": 64, "dropout": 0.05, "seed": 42, "gradient_checkpointing": True}
MAX_LENGTH = 4096
THINK = "<think>\n\n</think>\n\n"


def require(condition, what):
    print(f"  {'ok  ' if condition else 'FAIL'} {what}", flush=True)
    if not condition:
        sys.exit(1)


def show(step, tok):
    input_ids, labels = M.encode(step, tok, MAX_LENGTH)
    first = next(i for i, label in enumerate(labels) if label != M.IGNORE)
    print(f"\n--- {step['id']} ({step['task']}) tokens {len(input_ids)}, prompt {first}, target {len(input_ids) - first}")
    print("  prompt ends:", repr(tok.decode(input_ids[max(0, first - 24):first])))
    print("  target     :", repr(tok.decode(input_ids[first:])))
    print("  labels     :", labels[first - 2:first + 3], "...", labels[-2:])


def check_examples(steps, tok):
    print("examples")
    bad_prefix = bad_boundary = bad_end = bad_mask = 0
    for step in steps:
        official = M.F.render_prompt(step["messages"][:-1], step["tools"], tok)
        input_ids, labels = M.encode(step, tok, MAX_LENGTH)
        first = next(i for i, label in enumerate(labels) if label != M.IGNORE)
        prompt_ids = tok(official, add_special_tokens=False)["input_ids"]
        bad_prefix += input_ids[:first] != prompt_ids
        bad_boundary += not official.endswith(THINK)
        bad_end += input_ids[-1] != M.IM_END_ID or labels[-1] != M.IM_END_ID
        bad_mask += any(label != M.IGNORE for label in labels[:first]) or labels[first:] != input_ids[first:]
    require(bad_prefix == 0, f"prompt tokens equal the official renderer on all {len(steps)} examples")
    require(bad_boundary == 0, "every prompt ends with the empty think block")
    require(bad_end == 0, "every target ends with <|im_end|>, and it is scored")
    require(bad_mask == 0, "prompt labels are -100 and target labels equal the input")


def check_padding(steps, tok):
    print("padding")
    short = min(steps, key=lambda step: len(M.encode(step, tok, MAX_LENGTH)[0]))
    long = max(steps, key=lambda step: len(M.encode(step, tok, MAX_LENGTH)[0]))
    pairs = [M.encode(short, tok, MAX_LENGTH), M.encode(long, tok, MAX_LENGTH)]
    batch = M.collate(pairs, tok.pad_token_id)
    gap = len(pairs[1][0]) - len(pairs[0][0])
    require(gap > 0, f"the two examples differ in length by {gap}")
    require(int(batch["attention_mask"][0].sum()) == len(pairs[0][0]), "attention covers exactly the real tokens")
    require(bool((batch["labels"][0, -gap:] == M.IGNORE).all()), "padding is not scored")
    require(int(batch["labels"][0, len(pairs[0][0]) - 1]) == M.IM_END_ID, "the real <|im_end|> stays scored next to the padding")
    print(f"  pad id {tok.pad_token_id}, <|im_end|> id {M.IM_END_ID}, <|endoftext|> id {tok.convert_tokens_to_ids('<|endoftext|>')}")


def snapshot(model):
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def check_loss(model, steps, tok):
    print("loss on target positions only")
    model.eval()
    gaps = []
    for picks in (steps[:2], steps[2:4], steps[6:8], steps[1:3]):
        batch = M.collate([M.encode(step, tok, MAX_LENGTH) for step in picks], tok.pad_token_id, "cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ours = M.target_loss(model, batch).item()
            theirs = model(**batch).loss.item()
        gaps.append(abs(ours - theirs))
        print(f"  ours {ours:.5f}, the model's own {theirs:.5f}")
    require(max(gaps) < 1e-3, f"target_loss equals the model's own loss (largest gap {max(gaps):.2e})")


def check_update(model, steps, tok):
    print("one update")
    groups = M.trainable_groups(model)
    print("  trainable", groups)
    require(groups["other"] == 0 and groups["lora"] > 0 and groups["chat_rows"] == 25 * 1024, "only LoRA and the 25 chat rows train")
    before = snapshot(model)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=2e-4)
    model.train()
    batch = M.collate([M.encode(step, tok, MAX_LENGTH) for step in steps[:2]], tok.pad_token_id, "cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = M.target_loss(model, batch)
    loss.backward()
    require(torch.isfinite(loss).item(), f"loss is finite ({loss.item():.3f})")
    # lora_B starts at zero, so lora_A only gets a gradient from the second update on.
    first_round = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad and "lora_A" not in name]
    with_grad = sum(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0) for _, parameter in first_round)
    require(with_grad == len(first_round), f"lora_B and the chat rows got a gradient on the first update ({with_grad}/{len(first_round)})")
    optimizer.step()
    optimizer.zero_grad()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        M.target_loss(model, batch).backward()
    with_grad = sum(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0) for parameter in parameters)
    require(with_grad == len(parameters), f"all trainable tensors got a gradient on the second update ({with_grad}/{len(parameters)})")
    optimizer.step()
    changed_frozen = changed_lora = changed_rows = 0
    for name, parameter in model.named_parameters():
        moved = not torch.equal(before[name], parameter)
        if parameter.requires_grad:
            changed_rows += moved and "trainable_tokens_delta" in name
            changed_lora += moved and "lora_" in name
        else:
            changed_frozen += moved
    require(changed_frozen == 0, "no frozen weight moved")
    require(changed_lora > 0 and changed_rows == 1, f"{changed_lora} LoRA tensors and the chat rows moved")


def logits(model, batch):
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.float().cpu()


def check_save_and_merge(model, steps, tok, out):
    print("save, fresh load, official merge")
    path = M.ensure_dir(f"{out}/adapter")
    model.save_pretrained(path)
    with safe_open(f"{path}/adapter_model.safetensors", "pt") as f:
        names = list(f.keys())
    rows = [name for name in names if "trainable_tokens" in name]
    print(f"  {len(names)} tensors in the adapter, chat rows: {rows}")
    require(len(rows) >= 1, "the chat rows are in the adapter file")
    require(not any(name.endswith("embed_tokens.weight") or name.endswith("lm_head.weight") for name in names), "no full embedding table is saved")

    batch = M.collate([M.encode(step, tok, MAX_LENGTH) for step in steps[:2]], tok.pad_token_id, "cuda")
    trained = logits(model, batch)

    base = AutoModelForCausalLM.from_pretrained(M.BASE_MODEL, dtype=torch.bfloat16).cuda()
    reloaded = PeftModel.from_pretrained(base, path)
    fresh = logits(reloaded, batch)
    gap = float((trained - fresh).abs().max())
    require(gap < 1e-3, f"a fresh load gives the same logits (largest gap {gap:.2e})")
    del reloaded, base
    torch.cuda.empty_cache()

    from eval.generate import merge_adapter

    merged_dir = merge_adapter(M.BASE_MODEL, path, f"{out}/merged")
    merged = AutoModelForCausalLM.from_pretrained(merged_dir, dtype=torch.bfloat16).cuda()
    plain = AutoModelForCausalLM.from_pretrained(M.BASE_MODEL, dtype=torch.bfloat16)
    embed, head = merged.get_input_embeddings().weight, merged.get_output_embeddings().weight
    require(embed is head or torch.equal(embed, head), "after the merge, input and output rows are still the same")
    start, end = M.CHAT_TOKEN_IDS[0], M.CHAT_TOKEN_IDS[-1] + 1
    chat = embed.detach().cpu()[start:end]
    distinct = len({tuple(row.float().tolist()[:8]) for row in chat})
    require(distinct == 25, f"the merged model has {distinct} distinct chat rows, out of 25")
    original = plain.get_input_embeddings().weight.detach()
    outside = torch.cat([embed.detach().cpu()[:start], embed.detach().cpu()[end:]])
    require(torch.equal(outside, torch.cat([original[:start], original[end:]])), "every other row equals the Base model")
    merged_logits = logits(merged, batch)
    width = batch["attention_mask"].sum(dim=1)
    gap = max(float((trained[i, :n] - merged_logits[i, :n]).abs().max()) for i, n in enumerate(width.tolist()))
    print(f"  largest logit gap, merged vs trained: {gap:.3f}")
    agree = float((trained.argmax(-1) == merged_logits.argmax(-1))[batch["attention_mask"].cpu().bool()].float().mean())
    # A sanity floor only. After two updates the model is nearly untrained and this mostly
    # measures bf16 rounding over every prompt position (it ranged 98.9-99.5% between runs).
    # What counts is a trained adapter through the official path, which every score uses.
    require(agree > 0.95, f"merged and trained models pick the same next token at {agree:.1%} of positions")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data_work/processed/data_v0.jsonl")
    args = ap.parse_args()
    tok = M.F.tokenizer()
    steps = M.load_steps(args.data)
    check_examples(steps, tok)
    check_padding(steps, tok)
    rng = random.Random(5)
    picks = rng.sample([step for step in steps if step["task"] == "part1"], 6) + rng.sample([step for step in steps if step["task"] == "part2"], 6)
    for step in picks:
        show(step, tok)
    print()
    model = M.build_model(CONFIG).cuda()
    check_loss(model, picks, tok)
    check_update(model, picks, tok)
    check_save_and_merge(model, picks, tok, args.out)
    print("\nall checks passed")


if __name__ == "__main__":
    main()

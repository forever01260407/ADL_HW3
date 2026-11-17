"""Check an adapter directory the way the grader will use it.

    python code/check_adapter.py ADAPTER_DIR [MERGED_DIR]

Confirms the two required files, the rank limit and the chat-token rows, loads the adapter
on a fresh Base with peft, and, given a merged directory from the official merge, confirms
the rows reached the merged model. Nothing here depends on training state. Run from the
project root.
"""
import json
import os
import sys

import torch
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoModelForCausalLM

BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
FIRST, LAST = 151644, 151668
RANK_LIMIT = 512


def require(condition, what):
    print(f"  {'ok  ' if condition else 'FAIL'} {what}", flush=True)
    if not condition:
        sys.exit(1)


def main():
    adapter = sys.argv[1]
    merged_dir = sys.argv[2] if len(sys.argv) > 2 else None
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        require(os.path.isfile(f"{adapter}/{name}"), f"{name} present")
    size = os.path.getsize(f"{adapter}/adapter_model.safetensors")
    print(f"  adapter size {size / 2**20:.1f} MiB")
    with open(f"{adapter}/adapter_config.json", encoding="utf-8") as f:
        config = json.load(f)
    require(config["peft_type"] == "LORA", "the adapter is LoRA")
    require(config["r"] <= RANK_LIMIT, f"rank {config['r']} is within {RANK_LIMIT}")
    indices = config["trainable_token_indices"]
    require(sorted(indices["embed_tokens"]) == list(range(FIRST, LAST + 1)), "all 25 chat token ids are trainable")
    print(f"  base model recorded: {config.get('base_model_name_or_path')}")

    with safe_open(f"{adapter}/adapter_model.safetensors", "pt") as f:
        names = list(f.keys())
        row_names = [name for name in names if "trainable_tokens_delta" in name]
        require(len(row_names) == 1 and tuple(f.get_tensor(row_names[0]).shape) == (25, 1024), "the file holds 25 chat rows")
        rows = f.get_tensor(row_names[0]).float()
    require(not any(name.endswith(("embed_tokens.weight", "lm_head.weight")) for name in names), "no full embedding table")

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16)
    original = base.get_input_embeddings().weight.detach().clone()
    model = PeftModel.from_pretrained(base, adapter)
    print("  fresh Base + PeftModel.from_pretrained loaded")
    del model

    if merged_dir:
        merged = AutoModelForCausalLM.from_pretrained(merged_dir, dtype=torch.bfloat16)
        embed = merged.get_input_embeddings().weight.detach()
        head = merged.get_output_embeddings().weight.detach()
        require(torch.equal(embed, head), "input and output rows are equal after the merge")
        merged_rows = embed[FIRST:LAST + 1].float()
        require(float((merged_rows - rows).abs().max()) < 0.01, "the merged rows are the adapter's rows, up to bf16 rounding")
        require(not torch.equal(embed[FIRST:LAST + 1], original[FIRST:LAST + 1]), "the merged rows differ from the Base rows")
        keep = torch.ones(len(embed), dtype=torch.bool)
        keep[FIRST:LAST + 1] = False
        require(torch.equal(embed[keep], original[keep]), "every other row equals the Base model")


if __name__ == "__main__":
    main()

"""Build the trainable model and turn data steps into training examples.

Shared by train.py and check_train_setup.py so the check tests the code that trains.
Run from the project root.
"""
import json
import os
import sys

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

sys.path.insert(0, "hw3")

from eval import format as F  # noqa: E402

BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
CHAT_TOKEN_IDS = list(range(151644, 151669))
IM_END_ID = 151645
IGNORE = -100


def load_steps(path):
    """Every assistant message of every conversation, one example each."""
    sys.path.insert(0, "code/data")
    from common import expand_steps

    with open(path, encoding="utf-8") as f:
        return [step for line in f for step in expand_steps(json.loads(line))]


def encode(step, tok, max_length):
    """(input_ids, labels), or None when the example is too long to train on as a whole."""
    messages, tools = step["messages"], step["tools"]
    prompt = F.render_prompt(messages[:-1], tools, tok)
    full = tok.apply_chat_template(messages, tools=tools or None, tokenize=False, enable_thinking=False)
    if not full.startswith(prompt):
        raise ValueError(f"{step['id']}: the test prompt is not a prefix of the full text")
    target = full[len(prompt):].rstrip("\n")
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tok(target, add_special_tokens=False)["input_ids"]
    if len(target_ids) < 2 or target_ids[-1] != IM_END_ID:
        raise ValueError(f"{step['id']}: the target does not end with <|im_end|>")
    if len(prompt_ids) + len(target_ids) > max_length:
        return None
    return prompt_ids + target_ids, [IGNORE] * len(prompt_ids) + target_ids


def collate(pairs, pad_id, device=None):
    """Pad on the right. Padding is found by position, never by id, since a real token can share the pad id."""
    width = max(len(ids) for ids, _ in pairs)
    input_ids, labels, attention = [], [], []
    for ids, label in pairs:
        gap = width - len(ids)
        input_ids.append(ids + [pad_id] * gap)
        labels.append(label + [IGNORE] * gap)
        attention.append([1] * len(ids) + [0] * gap)
    batch = {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels), "attention_mask": torch.tensor(attention)}
    return {key: value.to(device) for key, value in batch.items()} if device else batch


def target_loss(model, batch):
    """Cross-entropy over the target tokens, the same value the model's own loss gives.

    The model's own forward scores all 151,936 words at every position of a prompt of
    several thousand tokens, then throws nearly all of it away. Scoring only the positions
    that have a label takes far less memory and time.
    """
    inner = model.get_base_model()
    hidden = inner.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
    labels = batch["labels"][:, 1:]
    scored = labels != IGNORE
    logits = inner.get_output_embeddings()(hidden[:, :-1][scored])
    return torch.nn.functional.cross_entropy(logits.float(), labels[scored])


def distinct_chat_rows(model, spread, seed):
    """All 25 chat tokens start as one identical row in the Base model. Give each its own
    start, at the spread of the rest of the vocabulary. The rows live in the adapter."""
    rows = next(parameter for name, parameter in model.named_parameters() if "trainable_tokens_delta" in name)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        rows.copy_((torch.randn(rows.shape, generator=generator) * spread).to(rows.device, rows.dtype))


def build_model(config):
    """Frozen Base with LoRA on all linear layers and the chat-token rows trainable.

    trainable_token_indices wraps both the embeddings and the tied lm_head around one set
    of rows (checked on peft 0.19.1). ensure_weight_tying only concerns LoRA on embed_tokens,
    which is not used here, and peft warns that it has nothing to do.
    """
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16)
    model.config.use_cache = False
    spread = model.get_input_embeddings().weight.data[:CHAT_TOKEN_IDS[0]].float().std().item()
    lora = LoraConfig(
        r=config["r"], lora_alpha=config["alpha"], lora_dropout=config["dropout"], bias="none", target_modules="all-linear",
        task_type="CAUSAL_LM", trainable_token_indices={"embed_tokens": CHAT_TOKEN_IDS},
    )
    model = get_peft_model(model, lora)
    distinct_chat_rows(model, spread, config["seed"])
    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def trainable_groups(model):
    """Trainable parameter counts by kind. `other` must stay empty."""
    groups = {"lora": 0, "chat_rows": 0, "other": 0}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            kind = "lora" if "lora_" in name else "chat_rows" if "trainable_tokens_delta" in name else "other"
            groups[kind] += parameter.numel()
    return groups


def file_hash(path):
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def peak_gpu_gb():
    return round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else 0.0


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

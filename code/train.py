"""Train one LoRA adapter on Qwen3-0.6B-Base for both homework parts.

    python code/train.py --config configs/e01.json --run runs/E01
    python code/train.py --config configs/e01.json --run runs/E01 --resume runs/E01/checkpoints/step_100
    python code/train.py --config B0.json --run runs/S/B0 --init-adapter runs/E23/checkpoints/step_1475

--init-adapter starts a NEW run from a trained adapter's weights only (LoRA and the 25 chat rows,
loaded as saved, not re-initialised); optimizer, LR schedule and sampling start fresh at update 0.
The run directory must not exist. --resume continues an interrupted run with its saved state.
A config may name a `schedule_file`: a JSON list of step ids, one per drawn example in order
(max_updates x micro_batch x accumulation), used instead of the random Mixer so two runs can see
exactly the same examples at chosen positions. Every id must exist; nothing is skipped silently.
Each checkpoint gets done.json (written last, atomically) listing its files and sizes.

Every assistant message is one example with the same prompt as the test. Only the
target is scored. A checkpoint holds the adapter plus the state needed to resume.
Run from the project root.
"""
import argparse
import importlib.metadata
import json
import os
import random
import shutil
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

import model_setup as M

UPDATE_LOG = "train_log.jsonl"


class ScheduleDrawer:
    """Draws the examples named by a fixed schedule, in order."""

    def __init__(self, examples, ids):
        index = {example["id"]: i for i, example in enumerate(examples)}
        missing = [step for step in ids if step not in index]
        if missing:
            raise SystemExit(f"{len(missing)} schedule ids are not training examples (too long or absent), e.g. {missing[:3]}")
        self.order = [index[step] for step in ids]
        self.pos = 0

    def draw(self):
        pick = self.order[self.pos]
        self.pos += 1
        return pick

    def state(self):
        return {"pos": self.pos}

    def load(self, state):
        self.pos = state["pos"]


class Mixer:
    """Draws examples so that Part 2 gets `part2_share` of them, whatever the data's own mix."""

    def __init__(self, examples, share, seed):
        self.rng = random.Random(seed)
        self.pools = {task: [i for i, example in enumerate(examples) if example["task"] == task] for task in ("part1", "part2")}
        self.queues = {task: [] for task in self.pools}
        self.share = share

    def draw(self):
        task = "part2" if self.rng.random() < self.share else "part1"
        if not self.queues[task]:
            self.queues[task] = self.rng.sample(self.pools[task], len(self.pools[task]))
        return self.queues[task].pop()

    def state(self):
        return {"rng": self.rng.getstate(), "queues": self.queues}

    def load(self, state):
        self.rng.setstate(state["rng"])
        self.queues = state["queues"]


def select_steps(config):
    steps = M.load_steps(config["train_file"])
    if config.get("max_examples"):
        random.Random(config["seed"]).shuffle(steps)
        steps = steps[:config["max_examples"]]
    return steps


def prepare(config, tok):
    examples, too_long = [], 0
    for step in select_steps(config):
        encoded = M.encode(step, tok, config["max_length"])
        if encoded is None:
            too_long += 1
            continue
        input_ids, labels = encoded
        examples.append({"input_ids": input_ids, "labels": labels, "task": step["task"], "id": step["id"]})
    return examples, too_long


def schedule(update, total, warmup):
    if update < warmup:
        return (update + 1) / warmup
    return max(0.0, (total - update) / max(1, total - warmup))


def versions():
    return {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "vllm", "jinja2")}


def save_checkpoint(model, optimizer, scheduler, mixer, update, run, provenance):
    path = M.ensure_dir(f"{run}/checkpoints/step_{update}")
    model.save_pretrained(path)
    torch.save({
        "update": update, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "mixer": mixer.state(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
    }, f"{path}/resume_state.pt")
    files = {name: os.path.getsize(f"{path}/{name}") for name in sorted(os.listdir(path)) if not name.startswith("done.json")}
    done = {"update": update, "files": files, "adapter_sha256": M.file_hash(f"{path}/adapter_model.safetensors"), **provenance}
    with open(f"{path}/done.json.tmp", "w", encoding="utf-8") as f:
        json.dump(done, f, indent=1)
    os.replace(f"{path}/done.json.tmp", f"{path}/done.json")
    return path


def adapter_architecture(path):
    with open(f"{path}/adapter_config.json", encoding="utf-8") as f:
        c = json.load(f)
    return {"r": c["r"], "alpha": c["lora_alpha"], "dropout": c["lora_dropout"]}


def load_trained(path, config):
    """Fresh Base plus a saved adapter, trainable. The chat rows come from the adapter as saved."""
    base = AutoModelForCausalLM.from_pretrained(M.BASE_MODEL, dtype=torch.bfloat16)
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, path, is_trainable=True)
    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--resume", help="a checkpoint directory of this run")
    group.add_argument("--init-adapter", help="start a new run from this adapter's weights only")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)
    if args.init_adapter:
        if os.path.exists(args.run):
            raise SystemExit(f"{args.run} exists; a new run needs a new directory")
        parent = adapter_architecture(args.init_adapter)
        wanted = {"r": config["r"], "alpha": config["alpha"], "dropout": config["dropout"]}
        if parent != wanted:
            raise SystemExit(f"config LoRA {wanted} differs from the parent adapter {parent}")

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    tok = M.F.tokenizer()
    examples, too_long = prepare(config, tok)
    print(f"{len(examples)} examples, {too_long} left out as longer than {config['max_length']} tokens", flush=True)

    if args.resume or args.init_adapter:
        model = load_trained(args.resume or args.init_adapter, config)
    else:
        model = M.build_model(config)
    groups = M.trainable_groups(model)
    print("trainable parameters", groups, flush=True)
    if groups["other"]:
        raise SystemExit("something other than LoRA and the chat rows is trainable")
    model.cuda().train()

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=config["weight_decay"])
    total, warmup = config["max_updates"], max(1, round(config["warmup_ratio"] * config["max_updates"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda update: schedule(update, total, warmup))
    schedule_ids = None
    if config.get("schedule_file"):
        with open(config["schedule_file"], encoding="utf-8") as f:
            schedule_ids = json.load(f)
        needed = total * config["micro_batch"] * config["accumulation"]
        if len(schedule_ids) != needed:
            raise SystemExit(f"schedule has {len(schedule_ids)} ids, the run draws {needed}")
        mixer = ScheduleDrawer(examples, schedule_ids)
    else:
        mixer = Mixer(examples, config["part2_share"], config["seed"])
    provenance = {"train_sha256": M.file_hash(config["train_file"]),
                  "schedule_sha256": M.file_hash(config["schedule_file"]) if schedule_ids else None,
                  "parent_adapter": args.init_adapter,
                  "parent_sha256": M.file_hash(f"{args.init_adapter}/adapter_model.safetensors") if args.init_adapter else None}
    start = 0
    if args.resume:
        if os.path.exists(f"{args.resume}/done.json"):  # checkpoints from before done.json have no record to compare
            with open(f"{args.resume}/done.json", encoding="utf-8") as f:
                saved = json.load(f)
            for key in ("train_sha256", "schedule_sha256"):
                if saved.get(key) != provenance[key]:
                    raise SystemExit(f"resume refused: {key} differs from the checkpoint's")
            provenance["parent_adapter"], provenance["parent_sha256"] = saved.get("parent_adapter"), saved.get("parent_sha256")
        state = torch.load(f"{args.resume}/resume_state.pt", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        mixer.load(state["mixer"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        start = state["update"]
    else:
        M.ensure_dir(args.run)
        shutil.copyfile(args.config, f"{args.run}/config.json")
        with open(f"{args.run}/manifest.json", "w", encoding="utf-8") as f:
            json.dump({
                "base_model": M.BASE_MODEL, "train_file": config["train_file"], "train_sha256": M.file_hash(config["train_file"]), "examples": len(examples),
                "too_long": too_long, "trainable": groups, "versions": versions(), "effective_batch": config["micro_batch"] * config["accumulation"],
                **provenance, "start_update": 0,
            }, f, indent=1)

    pad = tok.pad_token_id
    drawn = {"part1": 0, "part2": 0}
    target_tokens = {"part1": 0, "part2": 0}
    tokens = 0
    began = time.time()
    with open(f"{args.run}/{UPDATE_LOG}", "a", encoding="utf-8") as log:
        for update in range(start + 1, total + 1):
            running = 0.0
            for _ in range(config["accumulation"]):
                picks = [examples[mixer.draw()] for _ in range(config["micro_batch"])]
                for pick in picks:
                    drawn[pick["task"]] += 1
                    target_tokens[pick["task"]] += sum(label != M.IGNORE for label in pick["labels"])
                    tokens += len(pick["input_ids"])
                batch = M.collate([(pick["input_ids"], pick["labels"]) for pick in picks], pad, "cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = M.target_loss(model, batch)
                (loss / config["accumulation"]).backward()
                running += loss.item() / config["accumulation"]
            norm = torch.nn.utils.clip_grad_norm_(parameters, config["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            record = {
                "update": update, "loss": round(running, 5), "lr": scheduler.get_last_lr()[0], "grad_norm": round(float(norm), 3), "examples": dict(drawn),
                "target_tokens": dict(target_tokens), "tokens": tokens, "gpu_peak_gb": M.peak_gpu_gb(), "elapsed": round(time.time() - began, 1),
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            if update % config.get("print_every", 10) == 0 or update == start + 1:
                print(json.dumps(record), flush=True)
            if update in config["save_at"]:
                print("saved", save_checkpoint(model, optimizer, scheduler, mixer, update, args.run, provenance), flush=True)


if __name__ == "__main__":
    main()

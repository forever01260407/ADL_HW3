"""GPU checks for train.py --init-adapter, schedule_file, done.json and guarded resume (runbook 6.1).

    python code/tests/test_init_adapter.py --parent runs/E23/checkpoints/step_1475 --work <new dir> --out <json>

1. Loading the parent with load_trained keeps every LoRA tensor and the chat rows identical to
   the saved file (nothing re-initialised).
2. A 2-update run from the parent on a 4-id schedule: exit 0, finite losses, nothing but LoRA and
   chat rows trainable, done.json written, saved adapter loads on a fresh Base.
3. Resume from step_1 continues at schedule position 2 and gives the same update-2 loss (within
   1e-3); resume with a different schedule is refused.
4. A second --init-adapter into the same run directory is refused, files untouched.
The weights made here are test output only and never a starting point for real runs.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, "code")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def run(argv):
    p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout[-3000:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.work)
    results = {}
    parent_hash = sha(f"{a.parent}/adapter_model.safetensors")

    # 1. identical tensors after loading
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    import train as T
    model = T.load_trained(a.parent, {"gradient_checkpointing": False})
    saved = load_file(f"{a.parent}/adapter_model.safetensors")
    loaded = get_peft_model_state_dict(model)
    same = set(saved) == set(loaded) and all(torch.equal(saved[k].cpu(), loaded[k].detach().cpu().to(saved[k].dtype)) for k in saved)
    chat = [k for k in saved if "trainable_tokens" in k]
    groups = __import__("model_setup").trainable_groups(model)
    results["1_init_identical"] = {"passed": bool(same and chat and groups["other"] == 0), "tensors": len(saved), "chat_row_tensors": chat, "trainable": groups}
    del model
    torch.cuda.empty_cache()

    # 2. tiny run on a fixed schedule
    sys.path.insert(0, "code/data")
    from common import expand_steps
    records = []
    with open("data_work/frozen/data_v5/train.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if (r.get("task") == "part1" and sum(x.get("task") == "part1" for x in records) < 2) or (r.get("task") == "part2" and sum(x.get("task") == "part2" for x in records) < 2):
                records.append(r)
            if len(records) == 4:
                break
    train_file = f"{a.work}/tiny_train.jsonl"
    with open(train_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ids = [next(expand_steps(r))["id"] for r in records]
    sched = f"{a.work}/schedule.json"
    json.dump(ids, open(sched, "w"))
    config = {"train_file": train_file, "schedule_file": sched, "max_examples": None, "seed": 42, "r": 32, "alpha": 64, "dropout": 0.05,
              "learning_rate": 5e-5, "weight_decay": 0.0, "warmup_ratio": 0.03, "max_grad_norm": 1.0, "micro_batch": 1, "accumulation": 2,
              "max_length": 8192, "max_updates": 2, "save_at": [1, 2], "part2_share": 0.339, "gradient_checkpointing": True, "print_every": 1}
    cfg = f"{a.work}/config.json"
    json.dump(config, open(cfg, "w"))
    rundir = f"{a.work}/run"
    code, out = run([sys.executable, "code/train.py", "--config", cfg, "--run", rundir, "--init-adapter", a.parent])
    log = [json.loads(l) for l in open(f"{rundir}/train_log.jsonl")] if os.path.exists(f"{rundir}/train_log.jsonl") else []
    manifest = json.load(open(f"{rundir}/manifest.json")) if os.path.exists(f"{rundir}/manifest.json") else {}
    done = json.load(open(f"{rundir}/checkpoints/step_2/done.json")) if os.path.exists(f"{rundir}/checkpoints/step_2/done.json") else {}
    import math
    ok = (code == 0 and [r["update"] for r in log] == [1, 2] and all(math.isfinite(r["loss"]) for r in log)
          and manifest.get("trainable", {}).get("other") == 0 and manifest.get("parent_sha256") == parent_hash and manifest.get("start_update") == 0
          and done.get("update") == 2 and done.get("adapter_sha256") == sha(f"{rundir}/checkpoints/step_2/adapter_model.safetensors"))
    reload = run([sys.executable, "code/check_adapter.py", f"{rundir}/checkpoints/step_2"])
    results["2_tiny_run"] = {"passed": bool(ok and reload[0] == 0), "exit": code, "log": log, "check_adapter_exit": reload[0], "check_adapter_tail": reload[1][-600:],
                             "tail": out[-800:] if code else ""}

    # 3. resume continues at the right position; a different schedule is refused
    state = torch.load(f"{rundir}/checkpoints/step_1/resume_state.pt", weights_only=False)
    rerun = f"{a.work}/resumed"
    subprocess.run(["cp", "-r", rundir, rerun], check=True)
    os.remove(f"{rerun}/train_log.jsonl")
    code3, out3 = run([sys.executable, "code/train.py", "--config", cfg, "--run", rerun, "--resume", f"{rerun}/checkpoints/step_1"])
    log3 = [json.loads(l) for l in open(f"{rerun}/train_log.jsonl")] if os.path.exists(f"{rerun}/train_log.jsonl") else []
    same_loss = len(log3) == 1 and log3[0]["update"] == 2 and abs(log3[0]["loss"] - log[1]["loss"]) < 1e-3 if len(log) == 2 else False
    json.dump(list(reversed(ids)), open(f"{a.work}/other_schedule.json", "w"))
    cfg2 = f"{a.work}/config_other.json"
    json.dump({**config, "schedule_file": f"{a.work}/other_schedule.json"}, open(cfg2, "w"))
    code4, out4 = run([sys.executable, "code/train.py", "--config", cfg2, "--run", rerun, "--resume", f"{rerun}/checkpoints/step_1"])
    results["3_resume"] = {"passed": bool(code3 == 0 and state["mixer"] == {"pos": 2} and same_loss and code4 != 0 and "resume refused" in out4),
                           "mixer_state_at_1": state["mixer"], "resumed_log": log3, "original_update2_loss": log[1]["loss"] if len(log) == 2 else None,
                           "other_schedule_exit": code4, "other_schedule_msg": out4.strip().splitlines()[-1:]}

    # 4. no overwrite
    before = sha(f"{rundir}/checkpoints/step_2/adapter_model.safetensors")
    code5, out5 = run([sys.executable, "code/train.py", "--config", cfg, "--run", rundir, "--init-adapter", a.parent])
    results["4_no_overwrite"] = {"passed": code5 != 0 and "exists" in out5 and before == sha(f"{rundir}/checkpoints/step_2/adapter_model.safetensors"),
                                 "exit": code5, "msg": out5.strip().splitlines()[-1:]}
    results["parent_unchanged"] = {"passed": sha(f"{a.parent}/adapter_model.safetensors") == parent_hash}
    passed = all(v["passed"] for v in results.values())
    with open(a.out, "x") as f:
        json.dump({"passed": passed, **results}, f, indent=1, default=str)
    for k, v in results.items():
        print("PASS" if v["passed"] else "FAIL", k)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

"""Check the versions, CUDA and BF16 math of the HW3 environment.

    python code/env/check_env.py
"""
import importlib.metadata as metadata
import platform
import shutil
import sys

import torch

TARGET = {
    "python": "3.11.15",
    "torch": "2.11.0",
    "transformers": "5.13.1",
    "peft": "0.19.1",
    "vllm": "0.25.1",
    "jinja2": "3.1.6",
}


def version_of(name):
    return platform.python_version() if name == "python" else metadata.version(name)


def check_versions():
    ok = True
    for name, want in TARGET.items():
        have = version_of(name)
        same = have == want
        ok &= same
        print(f"{name:<13} {have:<10} target {want:<10} {'ok' if same else 'DIFFERENT'}")
    print(f"{'safetensors':<13} {metadata.version('safetensors')}")
    return ok


def check_bf16_training():
    """One forward, backward and optimizer step in BF16 on the GPU."""
    device = "cuda"
    layer = torch.nn.Linear(4096, 4096, dtype=torch.bfloat16, device=device)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3)
    x = torch.randn(64, 4096, dtype=torch.bfloat16, device=device)
    before = layer.weight.detach().clone()
    loss = layer(x).float().pow(2).mean()
    loss.backward()
    optimizer.step()
    moved = (layer.weight.detach() - before).float().abs().sum().item()
    print(f"bf16 loss {loss.item():.4f}, weight change {moved:.1f}")
    return moved > 0 and torch.isfinite(loss).item()


def main():
    print("platform", platform.platform())
    versions_ok = check_versions()
    print("torch cuda build", torch.version.cuda)
    if not torch.cuda.is_available():
        print("CUDA NOT AVAILABLE")
        sys.exit(1)
    props = torch.cuda.get_device_properties(0)
    print(f"gpu {props.name}, {props.total_memory / 2**30:.1f} GiB, bf16 supported {torch.cuda.is_bf16_supported()}")
    bf16_ok = check_bf16_training()
    print(f"peak gpu memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    free = shutil.disk_usage("/").free / 2**30
    print(f"free disk on / {free:.1f} GiB")
    import vllm  # noqa: F401

    print("vllm import ok")
    print("RESULT", "pass" if versions_ok and bf16_ok else "FAIL")
    sys.exit(0 if versions_ok and bf16_ok else 1)


if __name__ == "__main__":
    main()

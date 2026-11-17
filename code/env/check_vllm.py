"""Load Qwen3-0.6B-Base on vLLM with the official settings and generate once.

    python code/env/check_vllm.py [--gpu-util 0.85]

The output is not meaningful: the Base model is untrained on chat tokens. This
only checks that the engine loads the model and produces tokens.
"""
import argparse
import sys
import time

import torch

sys.path.insert(0, "hw3")
from eval import generate  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu-util", type=float, default=0.85)
    args = parser.parse_args()

    start = time.time()
    gen = generate.make_vllm(generate.BASE_MODEL, max_len=16384, gpu_util=args.gpu_util)
    print(f"engine ready in {time.time() - start:.0f}s", flush=True)

    start = time.time()
    results = gen(["The capital of France is"] * 4, [32] * 4)
    print(f"generated {len(results)} replies in {time.time() - start:.1f}s", flush=True)
    print("sample:", repr(results[0][0][:80]), "finish:", results[0][1])
    used = torch.cuda.mem_get_info()
    print(f"gpu used {(used[1] - used[0]) / 2**30:.1f} GiB of {used[1] / 2**30:.1f} GiB")
    print("RESULT pass")


if __name__ == "__main__":
    main()

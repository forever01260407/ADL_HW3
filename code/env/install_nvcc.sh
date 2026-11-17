#!/usr/bin/env bash
# Install NVIDIA's CUDA compiler. Run once as root.
#
# The judge uses an FP8 KV cache, which only FlashInfer's attention kernels support.
# FlashInfer has no prebuilt kernel for that combination and compiles it on first use.
#
#   wsl -d hw3-ubuntu24 -u root --exec bash code/env/install_nvcc.sh
set -euo pipefail

CUDA_VERSION="13-0"
REPO="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64"
KEYRING="cuda-keyring_1.1-1_all.deb"

cd /tmp
curl -fsSLO "$REPO/$KEYRING"
dpkg -i "$KEYRING"
apt-get update -qq
apt-cache search --names-only "^cuda-compiler-" | head -8
apt-get install -y -qq --no-install-recommends "cuda-compiler-$CUDA_VERSION" > /tmp/apt_cuda.log 2>&1
ls -d /usr/local/cuda*
/usr/local/cuda/bin/nvcc --version | head -2
du -sh /usr/local/cuda-*/

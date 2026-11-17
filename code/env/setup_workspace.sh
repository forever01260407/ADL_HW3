#!/usr/bin/env bash
# Create the Linux working tree at ~/HW3_ADL.
#
# Small, hand-edited files stay in the Windows project and are linked in, so they
# remain visible in the editor. Large outputs live on the distro's own disk.
#
#   bash code/env/setup_workspace.sh [windows_project_path_as_seen_from_linux]
set -euo pipefail

WIN_ROOT="${1:-/mnt/c/Users/yuting/Desktop/HW3_ADL}"
ROOT="$HOME/HW3_ADL"

mkdir -p "$ROOT" "$HOME/env"
mkdir -p "$WIN_ROOT/configs" "$WIN_ROOT/records"
for name in hw3 code configs records EXECUTION_PLAN.md CLAUDE.md; do
    ln -sfn "$WIN_ROOT/$name" "$ROOT/$name"
done
mkdir -p "$ROOT"/data_work/raw "$ROOT"/data_work/processed "$ROOT"/data_work/frozen
mkdir -p "$ROOT"/runs "$ROOT"/figures "$ROOT"/submission "$ROOT"/release

cat > "$HOME/env/hw3.env" <<'EOF'
export HF_HOME="$HOME/hf"
export PATH="$HOME/venv/bin:$HOME/.local/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
export PYTHONUTF8=1
export PYTHONPYCACHEPREFIX="$HOME/.cache/pycache"
# vLLM turns pinned memory off under WSL by default, and its V2 runner needs it.
# VLLM_WSL2_ENABLE_PIN_MEMORY=1 makes V2 work here too, but V1 repeated identically and
# V2 did not, so V1 stays the main path (records/e03_results.md).
if grep -qi microsoft /proc/version; then
    export VLLM_USE_V2_MODEL_RUNNER=0
fi
# FlashInfer's sampler is compiled on first use and needs nvcc. Generation and the
# judge are greedy, and greedy decoding returns the argmax before reaching it.
export VLLM_USE_FLASHINFER_SAMPLER=0
EOF

ls -la "$ROOT"

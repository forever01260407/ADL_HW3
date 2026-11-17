#!/usr/bin/env bash
# Build the pinned HW3 environment on Ubuntu (WSL or native).
# Versions follow the grading environment on slide 23.
#
# vLLM and FlashInfer compile GPU kernels and need C and C++ compilers, which must
# be installed once as root:  apt-get install -y gcc g++
# The judge also needs nvcc, see code/env/install_nvcc.sh.
#
#   bash code/env/setup_env.sh resolve   # lock dependencies, install nothing
#   bash code/env/setup_env.sh install   # install exactly what the lock file says
set -euo pipefail

MODE="${1:-resolve}"
PYTHON_VERSION="3.11.15"
VENV="$HOME/venv"
ENV_DIR="$HOME/env"
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv --version

uv python install "$PYTHON_VERSION"
[ -d "$VENV" ] || uv venv --python "$PYTHON_VERSION" "$VENV"
"$VENV/bin/python" --version

mkdir -p "$ENV_DIR"
cat > "$ENV_DIR/requirements.in" <<'EOF'
torch==2.11.0
transformers==5.13.1
peft==0.19.1
vllm==0.25.1
jinja2==3.1.6
safetensors>=0.4
EOF

case "$MODE" in
resolve)
    uv pip compile "$ENV_DIR/requirements.in" --python "$VENV/bin/python" \
        -o "$ENV_DIR/requirements.lock.txt" --quiet
    echo "--- resolved key packages"
    grep -E '^(torch|transformers|peft|vllm|jinja2|safetensors|accelerate|numpy|tokenizers|huggingface-hub)==' \
        "$ENV_DIR/requirements.lock.txt"
    echo "--- total packages: $(grep -cE '^[a-z0-9]' "$ENV_DIR/requirements.lock.txt")"
    ;;
install)
    uv pip sync "$ENV_DIR/requirements.lock.txt" --python "$VENV/bin/python"
    uv pip freeze --python "$VENV/bin/python" > "$ENV_DIR/requirements.freeze.txt"
    echo "--- installed: $(wc -l < "$ENV_DIR/requirements.freeze.txt") packages"
    du -sh "$VENV"
    ;;
*)
    echo "unknown mode: $MODE (use resolve or install)" >&2
    exit 2
    ;;
esac

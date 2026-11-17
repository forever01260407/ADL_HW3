#!/usr/bin/env bash
# Run a command in the HW3 environment from the project root.
#
#   wsl -d hw3-ubuntu24 --exec bash /home/hw3/HW3_ADL/code/env/run.sh python hw3/eval.py --help
set -euo pipefail
# shellcheck disable=SC1091
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${HW3_ENV_FILE:-$HOME/env/hw3.env}"
if [ -f "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
fi
if [ -d "$PROJECT_ROOT/.venv/bin" ]; then
    export PATH="$PROJECT_ROOT/.venv/bin:$PATH"
fi
cd "$PROJECT_ROOT"
exec "$@"

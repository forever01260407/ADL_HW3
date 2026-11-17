#!/usr/bin/env bash
# Record or verify the SHA256 of the official eval pack.
#
#   bash code/env/hash_official.sh write   # record the hashes once
#   bash code/env/hash_official.sh check   # confirm nothing has changed
set -euo pipefail

cd "$HOME/HW3_ADL"
HASH_FILE="records/official_hashes.sha256"

case "${1:-check}" in
write)
    find -L hw3 -type f -not -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum > "$HASH_FILE"
    echo "recorded $(wc -l < "$HASH_FILE") files in $HASH_FILE"
    ;;
check)
    sha256sum --check --quiet "$HASH_FILE"
    echo "all $(wc -l < "$HASH_FILE") official files unchanged"
    ;;
*)
    echo "unknown mode: $1 (use write or check)" >&2
    exit 2
    ;;
esac

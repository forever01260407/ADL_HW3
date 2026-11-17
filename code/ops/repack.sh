#!/usr/bin/env bash
# Swap the adapter of the submission folder, check it, zip, and copy the zip to the Windows Desktop.
#   bash code/env/run.sh bash code/ops/repack.sh <adapter_dir> <tag>
# The previous zip on the Desktop is kept as r15922a14_<previous tag>.zip (tag read from submission/ADAPTER_TAG).
set -euo pipefail
SRC="$1"
TAG="$2"
P=submission/r15922a14
PREV=$(cat submission/ADAPTER_TAG 2>/dev/null || echo E28)
DESK=/mnt/c/Users/yuting/Desktop
test -f "$SRC/adapter_model.safetensors" && test -f "$SRC/adapter_config.json"
if [ -f "$DESK/r15922a14.zip" ] && [ ! -f "$DESK/r15922a14_$PREV.zip" ]; then
    cp "$DESK/r15922a14.zip" "$DESK/r15922a14_$PREV.zip"
fi
cp "$SRC/adapter_config.json" "$SRC/adapter_model.safetensors" "$P/adapter_checkpoint/"
cp /mnt/c/Users/yuting/Desktop/HW3_ADL/submission_draft/README.md /mnt/c/Users/yuting/Desktop/HW3_ADL/submission_draft/report.pdf "$P/"
chmod -R u+rwX,go+rX "$P"
python code/check_adapter.py "$P/adapter_checkpoint" 2>&1 | grep -E "ok|FAIL|loaded|rank"
sha256sum "$SRC/adapter_model.safetensors" "$P/adapter_checkpoint/adapter_model.safetensors"
rm -f submission/r15922a14.zip
(cd submission && python - <<'PY'
import os, zipfile
with zipfile.ZipFile("r15922a14.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk("r15922a14"):
        dirs.sort()
        for f in sorted(files):
            z.write(os.path.join(root, f))
names = zipfile.ZipFile("r15922a14.zip").namelist()
print("entries", len(names), "top", sorted({n.split("/")[0] for n in names}))
PY
)
cp submission/r15922a14.zip "$DESK/r15922a14.zip"
echo "$TAG" > submission/ADAPTER_TAG
sha256sum submission/r15922a14.zip "$DESK/r15922a14.zip"
echo "repacked with $TAG"

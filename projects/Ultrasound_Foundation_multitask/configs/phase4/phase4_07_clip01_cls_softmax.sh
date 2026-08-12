#!/usr/bin/env bash
set -euo pipefail

ROOT='/scratch/dr/o.iraqy/UUSIVC-MMSeg'
PYTHON='/home/o.iraqy/.conda/envs/ultrasound/bin/python'
CONFIG="$ROOT/projects/Ultrasound_Foundation_multitask/configs/phase4/phase4_07_clip01_cls_softmax.py"
WORK_DIR="$ROOT/work_dirs/phase4_07_clip01_cls_softmax"

if [[ -e "$WORK_DIR" ]]; then
    printf 'Refusing to overwrite existing work directory: %s\n' "$WORK_DIR" >&2
    exit 1
fi

cd "$ROOT"
exec "$PYTHON" tools/train.py "$CONFIG" \
    --work-dir "$WORK_DIR"

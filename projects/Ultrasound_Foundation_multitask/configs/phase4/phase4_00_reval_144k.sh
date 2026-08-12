#!/usr/bin/env bash
set -euo pipefail

ROOT='/scratch/dr/o.iraqy/UUSIVC-MMSeg'
PYTHON='/home/o.iraqy/.conda/envs/ultrasound/bin/python'
CONFIG="$ROOT/projects/Ultrasound_Foundation_multitask/configs/phase4/phase4_00_reval_144k.py"
CHECKPOINT="$ROOT/work_dirs/phase2_r2_dinov2_vitb_mask2former_batch16/iter_144000.pth"
WORK_DIR="$ROOT/work_dirs/phase4_00_reval_144k"
VAL_ROOT='/scratch/dr/UUSIVC26/mmseg_format_full/val'

if [[ -e "$WORK_DIR" ]]; then
    printf 'Refusing to overwrite existing work directory: %s\n' "$WORK_DIR" >&2
    exit 1
fi

cd "$ROOT"
exec "$PYTHON" tools/test.py "$CONFIG" "$CHECKPOINT" \
    --work-dir "$WORK_DIR" \
    --cfg-options \
    test_dataloader.dataset.data_root="$VAL_ROOT" \
    test_evaluator.data_root="$VAL_ROOT"

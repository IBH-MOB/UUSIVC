#!/usr/bin/env bash
set -euo pipefail

ROOT='/scratch/dr/o.iraqy/UUSIVC-MMSeg'
PYTHON='/home/o.iraqy/.conda/envs/ultrasound/bin/python'
CONFIG_ROOT="$ROOT/projects/Ultrasound_Foundation_multitask/configs/phase4"

configs=(
    phase4_01_cls_softmax_finetune.py
    phase4_02_cls_balanced_sampling.py
    phase4_03_task_balanced_sampling.py
    phase4_04_aspect_preserving.py
    phase4_05_hard_organ_segmentation.py
    phase4_06_combined_candidate.py
)

cd "$ROOT"
for config_name in "${configs[@]}"; do
    config="$CONFIG_ROOT/$config_name"
    work_dir="$ROOT/work_dirs/${config_name%.py}"
    if [[ -e "$work_dir" ]]; then
        printf 'Refusing to overwrite existing work directory: %s\n' "$work_dir" >&2
        exit 1
    fi
    "$PYTHON" tools/train.py "$config" --work-dir "$work_dir"
done

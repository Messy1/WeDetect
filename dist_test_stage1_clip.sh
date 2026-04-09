#!/usr/bin/env bash

ANN=$1
CKPT=$2

if [ -z "$ANN" ] || [ -z "$CKPT" ]; then
    echo "Usage: bash dist_test_stage1_clip.sh <val_ann.json|jsonl> <stage1_epoch_ckpt> [extra args]"
    echo ""
    echo "Example:"
    echo "  bash dist_test_stage1_clip.sh ./data/stage1/clip_coco_val.jsonl ./work_dirs/stage1_llm2clip/stage1_epoch_30.pth --data-root ./data --amp --bf16"
    exit 1
fi

DATA_ROOT=${DATA_ROOT:-""}

CMD=(
    python
    eval_stage1_clip.py
    --ann "${ANN}"
    --checkpoint "${CKPT}"
)

if [ -n "${DATA_ROOT}" ]; then
    CMD+=(--data-root "${DATA_ROOT}")
fi

if [ $# -gt 2 ]; then
    CMD+=("${@:3}")
fi

PYTHONPATH="$(dirname "$0")/..":$PYTHONPATH "${CMD[@]}"

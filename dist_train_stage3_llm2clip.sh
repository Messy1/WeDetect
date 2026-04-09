#!/usr/bin/env bash

set -e

GPUS=$1
STAGE1_INIT_CKPT=$2

if [ -z "$GPUS" ] || [ -z "$STAGE1_INIT_CKPT" ]; then
    echo "Usage: bash dist_train_stage3_llm2clip.sh <gpus> <stage1_init_ckpt> [extra args]"
    echo ""
    echo "Required env:"
    echo "  BASE_CKPT      # default: ./checkpoints/wedetect_base.pth"
    echo ""
    echo "Optional env:"
    echo "  CONFIG         # default: config/wedetect_base_coco_full_tuning_llm2clip_8xbs4_2e-5.py"
    echo "  WORK_DIR       # default: ./work_dirs/stage3_llm2clip"
    echo "  MERGED_INIT    # default: \${WORK_DIR}/wedetect_stage3_init.pth"
    echo "  LLM2CLIP_MODEL_NAME   # override model.backbone.text_model.model_name"
    echo "  CLASS_TEXT_PATH       # override train/val/test class_text_path"
    exit 1
fi

BASE_CKPT=${BASE_CKPT:-"./checkpoints/wedetect_base.pth"}
CONFIG=${CONFIG:-"config/wedetect_base_coco_full_tuning_llm2clip_8xbs4_2e-5.py"}
WORK_DIR=${WORK_DIR:-"./work_dirs/stage3_llm2clip"}
MERGED_INIT=${MERGED_INIT:-"${WORK_DIR}/wedetect_stage3_init.pth"}

mkdir -p "${WORK_DIR}"

python prepare_stage3_init_ckpt.py \
    --base-ckpt "${BASE_CKPT}" \
    --stage1-ckpt "${STAGE1_INIT_CKPT}" \
    --out-ckpt "${MERGED_INIT}" \
    --replace-text

CFG_OPTS=("load_from=${MERGED_INIT}")
if [ -n "${LLM2CLIP_MODEL_NAME}" ]; then
    CFG_OPTS+=("model.backbone.text_model.model_name=${LLM2CLIP_MODEL_NAME}")
fi
if [ -n "${CLASS_TEXT_PATH}" ]; then
    CFG_OPTS+=("train_dataloader.dataset.class_text_path=${CLASS_TEXT_PATH}")
    CFG_OPTS+=("val_dataloader.dataset.class_text_path=${CLASS_TEXT_PATH}")
    CFG_OPTS+=("test_dataloader.dataset.class_text_path=${CLASS_TEXT_PATH}")
fi

EXTRA_ARGS=()
if [ $# -gt 2 ]; then
    EXTRA_ARGS=("${@:3}")
fi

bash dist_train.sh "${CONFIG}" "${GPUS}" \
    --work-dir "${WORK_DIR}" \
    --cfg-options "${CFG_OPTS[@]}" \
    "${EXTRA_ARGS[@]}"

#!/usr/bin/env bash

TRAIN_ANN=$1
GPUS=$2

if [ -z "$TRAIN_ANN" ] || [ -z "$GPUS" ]; then
    echo "Usage: bash dist_train_stage1_clip.sh <train_ann.json|jsonl> <gpus> [extra args]"
    echo ""
    echo "Example:"
    echo "  bash dist_train_stage1_clip.sh /data/laion/train.jsonl 8 --data-root /data/laion/images --trust-remote-code --amp --bf16"
    exit 1
fi

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

OUTPUT_DIR=${OUTPUT_DIR:-"./work_dirs/stage1_llm2clip"}
DATA_ROOT=${DATA_ROOT:-""}
VAL_ANN=${VAL_ANN:-""}

CMD=(
    torchrun
    --nnodes=${NNODES}
    --node_rank=${NODE_RANK}
    --master_addr=${MASTER_ADDR}
    --nproc_per_node=${GPUS}
    --master_port=${PORT}
    train_stage1_clip.py
    --train-ann "${TRAIN_ANN}"
    --output-dir "${OUTPUT_DIR}"
)

if [ -n "${DATA_ROOT}" ]; then
    CMD+=(--data-root "${DATA_ROOT}")
fi

if [ -n "${VAL_ANN}" ]; then
    CMD+=(--val-ann "${VAL_ANN}")
fi

# Optional override when you don't want to use the default path set in train_stage1_clip.py
if [ -n "${LLM2CLIP_MODEL_NAME}" ]; then
    CMD+=(--llm2clip-model-name "${LLM2CLIP_MODEL_NAME}")
fi

if [ $# -gt 2 ]; then
    CMD+=("${@:3}")
fi

PYTHONPATH="$(dirname "$0")/..":$PYTHONPATH "${CMD[@]}"

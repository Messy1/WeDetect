#!/usr/bin/env bash

GPUS=$1

if [ -z "$GPUS" ]; then
    echo "Usage: bash dist_train_stage3_openclip_fulltune.sh <gpus> [extra train args]"
    echo ""
    echo "Example:"
    echo "  OPENCLIP_STAGE3_INIT_CKPT=./work_dirs/stage2_openclip_bridge/epoch_20.pth \\"
    echo "  OPENCLIP_STAGE3_WORK_DIR=./work_dirs/stage3_openclip_fulltune \\"
    echo "  bash dist_train_stage3_openclip_fulltune.sh 4"
    exit 1
fi

CONFIG=${CONFIG:-config/wedetect_openclip_stage3_fulltune.py}

bash dist_train.sh "${CONFIG}" "${GPUS}" "${@:2}"


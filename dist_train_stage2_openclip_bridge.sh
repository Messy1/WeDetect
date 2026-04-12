#!/usr/bin/env bash

GPUS=$1

if [ -z "$GPUS" ]; then
    echo "Usage: bash dist_train_stage2_openclip_bridge.sh <gpus> [extra train args]"
    echo ""
    echo "Example:"
    echo "  OPENCLIP_STAGE2_INIT_CKPT=./work_dirs/openclip_init/wedetect_openclip_bridge_init.pth \\"
    echo "  OPENCLIP_STAGE2_WORK_DIR=./work_dirs/stage2_openclip_bridge \\"
    echo "  bash dist_train_stage2_openclip_bridge.sh 4"
    exit 1
fi

CONFIG=${CONFIG:-config/wedetect_openclip_stage2_bridge.py}

bash dist_train.sh "${CONFIG}" "${GPUS}" "${@:2}"


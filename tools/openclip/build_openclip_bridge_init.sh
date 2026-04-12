#!/usr/bin/env bash
set -euo pipefail

OPENCLIP_MODEL=${OPENCLIP_MODEL:-laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg}
OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-}
WEDETECT_BASE_CKPT=${WEDETECT_BASE_CKPT:-./checkpoints/wedetect_base.pth}
OUT_DIR=${OUT_DIR:-./work_dirs/openclip_init}

OPENCLIP_BB_CKPT=${OPENCLIP_BB_CKPT:-${OUT_DIR}/openclip_backbone_for_wedetect.pth}
DETECTOR_CKPT=${DETECTOR_CKPT:-${OUT_DIR}/wedetect_detector_neck_head_only.pth}
MERGED_CKPT=${MERGED_CKPT:-${OUT_DIR}/wedetect_openclip_bridge_init.pth}
REPORT_JSON=${REPORT_JSON:-${OUT_DIR}/openclip_convert_report.json}
INSPECT_JSON=${INSPECT_JSON:-${OUT_DIR}/openclip_model_info.json}

mkdir -p "${OUT_DIR}"

python tools/openclip/check_openclip_model.py \
  --model-name "${OPENCLIP_MODEL}" \
  --pretrained "${OPENCLIP_PRETRAINED}" \
  --output-json "${INSPECT_JSON}"

python tools/openclip/convert_openclip_to_wedetect_backbone.py \
  --openclip-model "${OPENCLIP_MODEL}" \
  --openclip-pretrained "${OPENCLIP_PRETRAINED}" \
  --vision-model-size base \
  --text-output-dim 768 \
  --output "${OPENCLIP_BB_CKPT}" \
  --report-json "${REPORT_JSON}"

python tools/openclip/extract_wedetect_detector_init.py \
  --input "${WEDETECT_BASE_CKPT}" \
  --output "${DETECTOR_CKPT}"

python tools/openclip/assemble_openclip_wedetect_init.py \
  --backbone-init "${OPENCLIP_BB_CKPT}" \
  --detector-init "${DETECTOR_CKPT}" \
  --output "${MERGED_CKPT}"

echo "[done] merged init checkpoint: ${MERGED_CKPT}"


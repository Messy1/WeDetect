#!/usr/bin/env bash

set -euo pipefail

# Convert O365/Flickr/GQA annotations to WeDetect finetune format:
# 1) COCO detection annotation json
# 2) class_text json (List[List[str]])
#
# All three datasets are treated as XYXY bbox format (verified by user).
#
# Usage:
#   bash convert_wedetect_datasets_xyxy.sh
#
# Optional env overrides:
#   O365_ANN
#   FLICKR_ANN
#   GQA_ANN
#   OUT_DIR
#   NORMALIZE_TEXT   # 1/0, default 1

O365_ANN=${O365_ANN:-"/ssd/wzh/data/objects365v1/o365v1_train_odvg.json"}
FLICKR_ANN=${FLICKR_ANN:-"/ssd/wzh/data/flickr30k_entities/final_flickr_separateGT_train_vg.json"}
GQA_ANN=${GQA_ANN:-"/ssd/wzh/data/gqa/final_mixed_train_no_coco_vg.json"}
OUT_DIR=${OUT_DIR:-"./data/converted_wedetect_xyxy"}
NORMALIZE_TEXT=${NORMALIZE_TEXT:-1}

mkdir -p "${OUT_DIR}"

EXTRA_TEXT_FLAGS=()
if [ "${NORMALIZE_TEXT}" = "1" ]; then
  EXTRA_TEXT_FLAGS+=(--lowercase --strip-punct)
fi

echo "[1/3] Converting O365 (odvg_detection_jsonl, xyxy)..."
python convert_to_wedetect_coco.py \
  --source-type odvg_detection_jsonl \
  --ann "${O365_ANN}" \
  --out-coco "${OUT_DIR}/o365_train_coco_xyxy.json" \
  --out-class-text "${OUT_DIR}/o365_class_texts_xyxy.json" \
  --out-classes "${OUT_DIR}/o365_classes_xyxy.json" \
  --bbox-format xyxy \
  "${EXTRA_TEXT_FLAGS[@]}"

echo "[2/3] Converting Flickr (grounding_jsonl, xyxy)..."
python convert_to_wedetect_coco.py \
  --source-type grounding_jsonl \
  --ann "${FLICKR_ANN}" \
  --out-coco "${OUT_DIR}/flickr_train_coco_xyxy.json" \
  --out-class-text "${OUT_DIR}/flickr_class_texts_xyxy.json" \
  --out-classes "${OUT_DIR}/flickr_classes_xyxy.json" \
  --bbox-format xyxy \
  "${EXTRA_TEXT_FLAGS[@]}"

echo "[3/3] Converting GQA (grounding_jsonl, xyxy)..."
python convert_to_wedetect_coco.py \
  --source-type grounding_jsonl \
  --ann "${GQA_ANN}" \
  --out-coco "${OUT_DIR}/gqa_train_coco_xyxy.json" \
  --out-class-text "${OUT_DIR}/gqa_class_texts_xyxy.json" \
  --out-classes "${OUT_DIR}/gqa_classes_xyxy.json" \
  --bbox-format xyxy \
  "${EXTRA_TEXT_FLAGS[@]}"

echo "[done] Converted files are under: ${OUT_DIR}"

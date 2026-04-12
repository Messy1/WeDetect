# OpenCLIP Bridge Workflow (WeDetect)

This directory provides a modular control-group pipeline using
`laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg`:

1. Convert OpenCLIP backbone to WeDetect-compatible init.
2. Keep WeDetect neck/head initialization from pretrained WeDetect.
3. Run stage2 bridge training (freeze backbone bodies).
4. Run stage3 full-tuning.

## 1) Inspect downloaded OpenCLIP model

```bash
python tools/openclip/check_openclip_model.py \
  --model-name laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg \
  --output-json work_dirs/openclip_init/openclip_model_info.json
```

## 2) Build OpenCLIP->WeDetect initialization checkpoint

One-shot helper:

```bash
OPENCLIP_MODEL=laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg \
WEDETECT_BASE_CKPT=./checkpoints/wedetect_base.pth \
OUT_DIR=./work_dirs/openclip_init \
bash tools/openclip/build_openclip_bridge_init.sh
```

Equivalent manual steps:

```bash
python tools/openclip/convert_openclip_to_wedetect_backbone.py \
  --openclip-model laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg \
  --vision-model-size base \
  --text-output-dim 768 \
  --output ./work_dirs/openclip_init/openclip_backbone_for_wedetect.pth

python tools/openclip/extract_wedetect_detector_init.py \
  --input ./checkpoints/wedetect_base.pth \
  --output ./work_dirs/openclip_init/wedetect_detector_neck_head_only.pth

python tools/openclip/assemble_openclip_wedetect_init.py \
  --backbone-init ./work_dirs/openclip_init/openclip_backbone_for_wedetect.pth \
  --detector-init ./work_dirs/openclip_init/wedetect_detector_neck_head_only.pth \
  --output ./work_dirs/openclip_init/wedetect_openclip_bridge_init.pth
```

## 3) Sanity checks before stage2

Backbone-only forward:

```bash
python tools/openclip/minimal_backbone_only_forward.py \
  --backbone-init ./work_dirs/openclip_init/openclip_backbone_for_wedetect.pth \
  --openclip-model laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg \
  --device cuda
```

Full stage2 sanity checks:

```bash
python tools/openclip/sanity_check_openclip_bridge.py \
  --config config/wedetect_openclip_stage2_bridge.py \
  --checkpoint ./work_dirs/openclip_init/wedetect_openclip_bridge_init.pth \
  --class-text-json data/texts/coco_class_texts_en.json \
  --image-path data/coco/val2017/000000000139.jpg \
  --device cuda
```

## 4) Stage2 bridge training

```bash
OPENCLIP_STAGE2_INIT_CKPT=./work_dirs/openclip_init/wedetect_openclip_bridge_init.pth \
OPENCLIP_STAGE2_WORK_DIR=./work_dirs/stage2_openclip_bridge \
COCO_TRAIN_CLASS_TEXT_EN=data/texts/coco_class_texts_en.json \
COCO_VAL_CLASS_TEXT_EN=data/texts/coco_class_texts_en.json \
bash dist_train_stage2_openclip_bridge.sh 4
```

## 5) Stage3 full-tuning

```bash
OPENCLIP_STAGE3_INIT_CKPT=./work_dirs/stage2_openclip_bridge/epoch_20.pth \
OPENCLIP_STAGE3_WORK_DIR=./work_dirs/stage3_openclip_fulltune \
bash dist_train_stage3_openclip_fulltune.sh 4
```

## 6) Evaluation

COCO:

```bash
bash dist_test.sh \
  config/wedetect_openclip_stage3_fulltune.py \
  ./work_dirs/stage3_openclip_fulltune/epoch_30.pth \
  4
```

LVIS mini-val:

```bash
LVIS_ANN_FILE=annotations/lvis_v1_minival_inserted_image_name.json \
LVIS_CLASS_TEXT_EN=data/texts/lvis_v1_class_texts_en.json \
bash dist_test.sh \
  config/wedetect_openclip_lvis_eval.py \
  ./work_dirs/stage3_openclip_fulltune/epoch_30.pth \
  4
```


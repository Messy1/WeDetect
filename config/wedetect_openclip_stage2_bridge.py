_base_ = ["wedetect_base.py"]

import os


# ---------------------------
# Experiment identity
# ---------------------------
work_dir = os.environ.get("OPENCLIP_STAGE2_WORK_DIR", "./work_dirs/stage2_openclip_bridge")
load_from = os.environ.get(
    "OPENCLIP_STAGE2_INIT_CKPT",
    "./work_dirs/openclip_init/wedetect_openclip_bridge_init.pth",
)


# ---------------------------
# Hyper-parameters (paper-like stage2)
# ---------------------------
num_training_classes = 80
num_classes = 80
base_lr = 5e-4
weight_decay = 0.05
train_batch_size_per_gpu = 4
max_epochs = 20
close_mosaic_epochs = 4
save_epoch_intervals = 1
persistent_workers = True

img_scale = (640, 640)
affine_scale = 0.5
mixup_prob = 0.15

find_unused_parameters = True


# ---------------------------
# OpenCLIP backbone setup
# ---------------------------
openclip_model_name = os.environ.get(
    "OPENCLIP_MODEL_NAME",
    "laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg",
)
openclip_pretrained = os.environ.get("OPENCLIP_PRETRAINED", "")

# IMPORTANT: use English class texts consistently.
coco_train_class_text_path = os.environ.get(
    "COCO_TRAIN_CLASS_TEXT_EN", "data/texts/coco_class_texts_en.json"
)
coco_val_class_text_path = os.environ.get(
    "COCO_VAL_CLASS_TEXT_EN", "data/texts/coco_class_texts_en.json"
)


model = dict(
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    backbone=dict(
        image_model=dict(
            type="ConvNextVisionBackbone",
            model_name="base",
            frozen_modules=["all"],  # stage2: freeze visual backbone
        ),
        text_model=dict(
            _delete_=True,
            type="OpenCLIPTextLanguageBackbone",
            model_name=openclip_model_name,
            pretrained=openclip_pretrained,
            output_dim=768,
            max_length=77,
            # stage2: freeze text tower body, keep text_proj trainable
            frozen_modules=["model"],
        ),
    ),
    bbox_head=dict(
        head_module=dict(
            num_classes=num_training_classes,
        ),
    ),
    train_cfg=dict(assigner=dict(num_classes=num_classes)),
)


# ---------------------------
# Dataset settings (keep WeDetect training flow)
# ---------------------------
pre_transform = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="LoadAnnotations", with_bbox=True),
]

text_transform = [
    dict(
        type="RandomLoadText",
        num_neg_samples=(num_classes, num_classes),
        max_num_samples=num_training_classes,
        padding_to_max=True,
        padding_value="",
    ),
    dict(
        type="mmdet.PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "flip", "flip_direction", "texts"),
    ),
]

mosaic_affine_transform = [
    dict(
        type="MultiModalMosaic",
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform,
    ),
    dict(
        type="WeDetectRandomAffine",
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_aspect_ratio=100.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114),
    ),
]

albu_train_transforms = [
    dict(type="Blur", p=0.01),
    dict(type="MedianBlur", p=0.01),
    dict(type="ToGray", p=0.01),
    dict(type="CLAHE", p=0.01),
]

train_pipeline = [
    *pre_transform,
    *mosaic_affine_transform,
    dict(
        type="YOLOv5MultiModalMixUp",
        prob=mixup_prob,
        pre_transform=[*pre_transform, *mosaic_affine_transform],
    ),
    dict(
        type="mmdet.Albu",
        transforms=albu_train_transforms,
        bbox_params=dict(
            type="BboxParams",
            format="pascal_voc",
            label_fields=["gt_bboxes_labels", "gt_ignore_flags"],
        ),
        keymap={"img": "image", "gt_bboxes": "bboxes"},
    ),
    dict(type="WeDetectHSVRandomAug"),
    dict(type="mmdet.RandomFlip", prob=0.5),
    *text_transform,
]

train_pipeline_stage2 = [
    *pre_transform,
    dict(type="WeDetectKeepRatioResize", scale=img_scale),
    dict(
        type="WeDetectLetterResize",
        scale=img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114.0),
    ),
    dict(
        type="WeDetectRandomAffine",
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        max_aspect_ratio=100,
        border_val=(114, 114, 114),
    ),
    dict(
        type="mmdet.Albu",
        transforms=albu_train_transforms,
        bbox_params=dict(
            type="BboxParams",
            format="pascal_voc",
            label_fields=["gt_bboxes_labels", "gt_ignore_flags"],
        ),
        keymap={"img": "image", "gt_bboxes": "bboxes"},
    ),
    dict(type="WeDetectHSVRandomAug"),
    dict(type="mmdet.RandomFlip", prob=0.5),
    *text_transform,
]

coco_train_dataset = dict(
    type="MultiModalDataset",
    dataset=dict(
        type="WeCocoDataset",
        data_root="data/coco/",
        ann_file="data/coco/annotations/instances_train2017.json",
        data_prefix=dict(img="train2017/"),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
    ),
    class_text_path=coco_train_class_text_path,
    pipeline=train_pipeline,
)

train_dataloader = dict(
    num_workers=2,
    persistent_workers=persistent_workers,
    batch_size=train_batch_size_per_gpu,
    collate_fn=dict(type="yolow_collate"),
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=coco_train_dataset,
)


# ---------------------------
# Validation / test (English class texts)
# ---------------------------
coco_val_data_root = os.environ.get("COCO_VAL_DATA_ROOT", "./data/coco/")
coco_val_ann = os.environ.get("COCO_VAL_ANN", "annotations/instances_val2017.json")

val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="MultiModalDataset",
        dataset=dict(
            type="WeCocoDataset",
            data_root=coco_val_data_root,
            test_mode=True,
            ann_file=coco_val_ann,
            data_prefix=dict(img="val2017"),
            batch_shapes_cfg=None,
        ),
        class_text_path=coco_val_class_text_path,
        pipeline=_base_.test_pipeline,
    ),
)
val_evaluator = dict(
    _delete_=True,
    type="CocoMetric",
    ann_file=os.path.join(coco_val_data_root, coco_val_ann),
    metric="bbox",
)
test_dataloader = val_dataloader
test_evaluator = val_evaluator


# ---------------------------
# Training schedule
# ---------------------------
param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, end_factor=1.0, begin=0, end=1000, by_epoch=False),
    dict(
        type="LinearLR",
        start_factor=1.0,
        end_factor=0.001,
        begin=0,
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

custom_hooks = [
    dict(
        type="mmdet.PipelineSwitchHook",
        switch_epoch=max_epochs - close_mosaic_epochs,
        switch_pipeline=train_pipeline_stage2,
    )
]

train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=max_epochs,
    val_interval=1,
    dynamic_intervals=[((max_epochs - close_mosaic_epochs), 1)],
)

optim_wrapper = dict(
    type="OptimWrapper",
    clip_grad=dict(max_norm=10.0),
    optimizer=dict(
        type="AdamW",
        lr=base_lr,
        weight_decay=weight_decay,
        batch_size_per_gpu=train_batch_size_per_gpu,
    ),
    paramwise_cfg=dict(
        custom_keys={
            "backbone.image_model": dict(lr_mult=0.0),
            "backbone.text_model.model": dict(lr_mult=0.0),
            "backbone.text_model.text_proj": dict(lr_mult=1.0),
            "logit_scale": dict(weight_decay=0.0),
        }
    ),
    constructor="YOLOWv5OptimizerConstructor",
)


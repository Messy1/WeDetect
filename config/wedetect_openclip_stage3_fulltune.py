_base_ = ["wedetect_openclip_stage2_bridge.py"]

import os


work_dir = os.environ.get("OPENCLIP_STAGE3_WORK_DIR", "./work_dirs/stage3_openclip_fulltune")

# Usually set to stage2 best/latest checkpoint when starting stage3.
_stage3_init = os.environ.get("OPENCLIP_STAGE3_INIT_CKPT", "")
load_from = _stage3_init if _stage3_init else None

base_lr = 1e-5
max_epochs = 30
close_mosaic_epochs = 4

find_unused_parameters = False


model = dict(
    backbone=dict(
        image_model=dict(
            frozen_modules=[],
        ),
        text_model=dict(
            frozen_modules=[],
        ),
    ),
)


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
        switch_pipeline=_base_.train_pipeline_stage2,
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
        weight_decay=_base_.weight_decay,
        batch_size_per_gpu=_base_.train_batch_size_per_gpu,
    ),
    paramwise_cfg=dict(
        custom_keys={
            # main optimization targets
            "neck.": dict(lr_mult=1.0),
            "bbox_head.": dict(lr_mult=1.0),
            "backbone.text_model.text_proj": dict(lr_mult=1.0),
            # smaller LR for backbone bodies
            "backbone.image_model": dict(lr_mult=0.2),
            "backbone.text_model.model": dict(lr_mult=0.1),
            # no decay on scale parameters if any
            "logit_scale": dict(weight_decay=0.0),
        }
    ),
    constructor="YOLOWv5OptimizerConstructor",
)

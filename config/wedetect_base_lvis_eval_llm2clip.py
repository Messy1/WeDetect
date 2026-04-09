_base_ = ["wedetect_base_coco_full_tuning_llm2clip_8xbs4_2e-5.py"]

import os


# LVIS evaluation settings (can be overridden by env vars)
lvis_data_root = os.environ.get("LVIS_DATA_ROOT", "data/coco/")
lvis_ann_file = os.environ.get(
    "LVIS_ANN_FILE", "annotations/lvis_v1_minival_inserted_image_name.json"
)
lvis_class_text_path = os.environ.get(
    "LVIS_CLASS_TEXT", "data/texts/lvis_v1_class_texts.json"
)
lvis_num_classes = int(os.environ.get("LVIS_NUM_CLASSES", 1203))


# Keep model architecture from training config, only switch test class count.
model = dict(num_test_classes=lvis_num_classes)


test_dataloader = dict(
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
            type="YOLOv5LVISV1Dataset",
            data_root=lvis_data_root,
            test_mode=True,
            ann_file=lvis_ann_file,
            data_prefix=dict(img=""),
            batch_shapes_cfg=None,
        ),
        class_text_path=lvis_class_text_path,
        pipeline=_base_.test_pipeline,
    ),
)

test_evaluator = dict(
    _delete_=True,
    type="LVISMetric",
    ann_file=lvis_data_root + lvis_ann_file,
    metric="bbox",
)

val_dataloader = test_dataloader
val_evaluator = test_evaluator


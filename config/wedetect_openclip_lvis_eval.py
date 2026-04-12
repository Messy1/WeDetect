_base_ = ["wedetect_openclip_stage3_fulltune.py"]

import os


model = dict(num_test_classes=int(os.environ.get("LVIS_NUM_CLASSES", 1203)))

lvis_data_root = os.environ.get("LVIS_DATA_ROOT", "data/coco/")
lvis_ann_file = os.environ.get(
    "LVIS_ANN_FILE", "annotations/lvis_v1_minival_inserted_image_name.json"
)
lvis_class_text_path = os.environ.get(
    "LVIS_CLASS_TEXT_EN", "data/texts/lvis_v1_class_texts_en.json"
)
lvis_ann_file_eval = (
    lvis_ann_file if os.path.isabs(lvis_ann_file) else os.path.join(lvis_data_root, lvis_ann_file)
)

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
    ann_file=lvis_ann_file_eval,
    metric="bbox",
)

val_dataloader = test_dataloader
val_evaluator = test_evaluator


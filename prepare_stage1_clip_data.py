import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Prepare COCO+Flickr30k data for stage1 CLIP-like training")
    parser.add_argument(
        "--coco-train-ann",
        type=str,
        default="./data/coco/annotations/captions_train2017.json",
        help="Path to COCO captions_train2017 json.",
    )
    parser.add_argument(
        "--coco-train-image-dir",
        type=str,
        default="./data/coco/train2017",
        help="Path to COCO train images directory.",
    )
    parser.add_argument(
        "--coco-val-ann",
        type=str,
        default="./data/coco/annotations/captions_val2017.json",
        help="Path to COCO captions_val2017 json.",
    )
    parser.add_argument(
        "--coco-val-image-dir",
        type=str,
        default="./data/coco/val2017",
        help="Path to COCO val images directory.",
    )
    parser.add_argument(
        "--flickr-token",
        type=str,
        default="./data/flickr30k_entities/results_20130124.token",
        help="Path to Flickr30k token file.",
    )
    parser.add_argument(
        "--flickr-image-dir",
        type=str,
        default="./data/flickr30k_entities/flickr30k_images",
        help="Path to Flickr30k image directory.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Base path used to write relative image paths.",
    )
    parser.add_argument(
        "--output-train",
        type=str,
        default="./data/stage1/clip_coco_flickr_train.jsonl",
        help="Output train jsonl path.",
    )
    parser.add_argument(
        "--output-val",
        type=str,
        default="./data/stage1/clip_coco_val.jsonl",
        help="Output val jsonl path (COCO val only).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed for shuffling/splitting.",
    )
    parser.add_argument(
        "--check-image-exists",
        action="store_true",
        help="If set, drop samples whose image files do not exist.",
    )
    return parser.parse_args()


def normalize_caption(text: str) -> str:
    return " ".join(str(text).strip().split())


def to_data_relative(path: Path, data_root: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(data_root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def load_coco_pairs(
    coco_ann_path: Path,
    coco_image_dir: Path,
    data_root: Path,
    check_exists: bool,
    source: str,
) -> List[Dict]:
    if not coco_ann_path.exists():
        raise FileNotFoundError(f"COCO annotation file not found: {coco_ann_path}")
    if not coco_image_dir.exists():
        raise FileNotFoundError(f"COCO image dir not found: {coco_image_dir}")

    with coco_ann_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    id_to_file = {img["id"]: img["file_name"] for img in coco.get("images", [])}
    pairs = []
    missing = 0
    bad_caption = 0

    for ann in coco.get("annotations", []):
        image_id = ann.get("image_id")
        file_name = id_to_file.get(image_id, "")
        caption = normalize_caption(ann.get("caption", ""))
        if not file_name:
            continue
        if not caption:
            bad_caption += 1
            continue

        image_path = coco_image_dir / file_name
        if check_exists and not image_path.exists():
            missing += 1
            continue

        pairs.append(
            {
                "image": to_data_relative(image_path, data_root),
                "text": caption,
                "source": source,
            }
        )

    print(
        f"[COCO:{source}] pairs={len(pairs)}, "
        f"missing_images={missing}, empty_caption={bad_caption}"
    )
    return pairs


def load_flickr_pairs(
    token_path: Path,
    flickr_image_dir: Path,
    data_root: Path,
    check_exists: bool,
) -> List[Dict]:
    if not token_path.exists():
        raise FileNotFoundError(f"Flickr token file not found: {token_path}")
    if not flickr_image_dir.exists():
        raise FileNotFoundError(f"Flickr image dir not found: {flickr_image_dir}")

    pairs = []
    missing = 0
    bad_line = 0
    bad_caption = 0

    with token_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "\t" not in line:
                bad_line += 1
                continue
            key, caption = line.split("\t", 1)
            caption = normalize_caption(caption)
            if not caption:
                bad_caption += 1
                continue

            # format: 1000092795.jpg#0
            image_file = key.split("#", 1)[0]
            image_path = flickr_image_dir / image_file
            if check_exists and not image_path.exists():
                missing += 1
                continue

            pairs.append(
                {
                    "image": to_data_relative(image_path, data_root),
                    "text": caption,
                    "source": "flickr30k",
                }
            )

    print(
        f"[Flickr30k] pairs={len(pairs)}, missing_images={missing}, "
        f"bad_line={bad_line}, empty_caption={bad_caption}"
    )
    return pairs


def write_jsonl(path: Path, items: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root)
    coco_train_ann_path = Path(args.coco_train_ann)
    coco_train_image_dir = Path(args.coco_train_image_dir)
    coco_val_ann_path = Path(args.coco_val_ann)
    coco_val_image_dir = Path(args.coco_val_image_dir)
    flickr_token_path = Path(args.flickr_token)
    flickr_image_dir = Path(args.flickr_image_dir)

    coco_train_pairs = load_coco_pairs(
        coco_ann_path=coco_train_ann_path,
        coco_image_dir=coco_train_image_dir,
        data_root=data_root,
        check_exists=args.check_image_exists,
        source="coco_train",
    )
    coco_val_pairs = load_coco_pairs(
        coco_ann_path=coco_val_ann_path,
        coco_image_dir=coco_val_image_dir,
        data_root=data_root,
        check_exists=args.check_image_exists,
        source="coco_val",
    )
    flickr_pairs = load_flickr_pairs(
        token_path=flickr_token_path,
        flickr_image_dir=flickr_image_dir,
        data_root=data_root,
        check_exists=args.check_image_exists,
    )

    train_items = coco_train_pairs + flickr_pairs
    random.Random(args.seed).shuffle(train_items)
    val_items = coco_val_pairs

    output_train = Path(args.output_train)
    write_jsonl(output_train, train_items)
    print(f"[Output] train={len(train_items)} -> {output_train}")

    output_val = Path(args.output_val)
    write_jsonl(output_val, val_items)
    print(f"[Output] val={len(val_items)} -> {output_val}")


if __name__ == "__main__":
    main()


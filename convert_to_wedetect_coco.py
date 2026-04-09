import argparse
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert ODVG / grounding annotations to WeDetect finetune format "
            "(COCO detection json + class_text json)."
        )
    )
    parser.add_argument(
        "--source-type",
        type=str,
        default="auto",
        choices=[
            "auto",
            "odvg_detection_jsonl",
            "grounding_jsonl",
            "grounding_coco_json",
        ],
    )
    parser.add_argument("--ann", type=str, required=True, help="Input annotation path.")
    parser.add_argument(
        "--out-coco",
        type=str,
        required=True,
        help="Output COCO detection annotation path.",
    )
    parser.add_argument(
        "--out-class-text",
        type=str,
        required=True,
        help="Output class text json path. Format: List[List[str]].",
    )
    parser.add_argument(
        "--out-classes",
        type=str,
        default="",
        help="Optional output plain class-name json path. Format: List[str].",
    )
    parser.add_argument(
        "--bbox-format",
        type=str,
        default="auto",
        choices=["auto", "xyxy", "xywh"],
        help="How to decode bbox in input annotation.",
    )
    parser.add_argument(
        "--file-name-prefix",
        type=str,
        default="",
        help="Optional prefix to prepend to image file_name in output COCO json.",
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="Lowercase category / phrase text.",
    )
    parser.add_argument(
        "--strip-punct",
        action="store_true",
        help="Strip punctuation at both ends of category / phrase text.",
    )
    return parser.parse_args()


def _is_box4(x: Any) -> bool:
    if not isinstance(x, (list, tuple)) or len(x) != 4:
        return False
    return all(isinstance(v, (int, float)) for v in x)


def _iter_boxes(box_obj: Any) -> Iterable[List[float]]:
    if _is_box4(box_obj):
        yield [float(v) for v in box_obj]
        return
    if isinstance(box_obj, (list, tuple)):
        for it in box_obj:
            if _is_box4(it):
                yield [float(v) for v in it]


def _flatten_spans(spans: Any) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []

    def _dfs(x: Any) -> None:
        if isinstance(x, (list, tuple)):
            if len(x) == 2 and all(isinstance(v, (int, float)) for v in x):
                s, e = int(x[0]), int(x[1])
                if e > s:
                    pairs.append((s, e))
            else:
                for y in x:
                    _dfs(y)

    _dfs(spans)
    return pairs


def _phrase_from_caption(caption: str, spans: Any) -> str:
    if not caption:
        return ""
    spans_flat = _flatten_spans(spans)
    if not spans_flat:
        return ""

    pieces: List[str] = []
    for s, e in spans_flat:
        s = max(0, min(len(caption), s))
        e = max(0, min(len(caption), e))
        if e <= s:
            continue
        piece = caption[s:e].strip()
        if piece:
            pieces.append(piece)

    uniq: List[str] = []
    seen = set()
    for p in pieces:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return " ".join(uniq)


def _normalize_text(text: str, lowercase: bool, strip_punct: bool) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    if lowercase:
        text = text.lower()
    if strip_punct:
        text = text.strip(".,;:!?()[]{}\"'`")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _bbox_to_xyxy(bbox: List[float], fmt: str) -> Tuple[float, float, float, float]:
    if fmt == "xywh":
        x, y, w, h = bbox
        return float(x), float(y), float(x + w), float(y + h)
    x1, y1, x2, y2 = bbox
    return float(x1), float(y1), float(x2), float(y2)


class StreamingCocoBuilder:
    def __init__(
        self,
        out_coco_path: Path,
        out_class_text_path: Path,
        lowercase: bool = False,
        strip_punct: bool = False,
    ) -> None:
        self.out_coco_path = out_coco_path
        self.out_class_text_path = out_class_text_path
        self.lowercase = lowercase
        self.strip_punct = strip_punct

        self._tmp_dir = tempfile.TemporaryDirectory(prefix="wedetect_convert_")
        self._images_tmp = Path(self._tmp_dir.name) / "images.jsonl"
        self._anns_tmp = Path(self._tmp_dir.name) / "annotations.jsonl"
        self._images_fp = self._images_tmp.open("w", encoding="utf-8")
        self._anns_fp = self._anns_tmp.open("w", encoding="utf-8")

        self.image_key_to_id: Dict[str, int] = {}
        self.image_size_by_id: Dict[int, Tuple[int, int]] = {}
        self.category_name_to_id: Dict[str, int] = {}
        self.category_id_to_name: Dict[int, str] = {}
        self.ann_id = 1

        self.total_boxes = 0
        self.valid_boxes = 0

    def close(self) -> None:
        if not self._images_fp.closed:
            self._images_fp.close()
        if not self._anns_fp.closed:
            self._anns_fp.close()
        self._tmp_dir.cleanup()

    def _category_id(self, raw_name: str) -> Optional[int]:
        name = _normalize_text(raw_name, self.lowercase, self.strip_punct)
        if not name:
            return None
        if name not in self.category_name_to_id:
            cid = len(self.category_name_to_id) + 1
            self.category_name_to_id[name] = cid
            self.category_id_to_name[cid] = name
        return self.category_name_to_id[name]

    def register_image(self, image_key: str, file_name: str, width: int, height: int) -> int:
        if image_key in self.image_key_to_id:
            return self.image_key_to_id[image_key]
        image_id = len(self.image_key_to_id) + 1
        self.image_key_to_id[image_key] = image_id
        self.image_size_by_id[image_id] = (width, height)
        rec = {
            "id": image_id,
            "file_name": file_name,
            "width": int(width),
            "height": int(height),
        }
        self._images_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return image_id

    def add_annotation(
        self,
        image_id: int,
        raw_bbox: List[float],
        bbox_format: str,
        raw_category: str,
    ) -> bool:
        self.total_boxes += 1
        if image_id not in self.image_size_by_id:
            return False
        cat_id = self._category_id(raw_category)
        if cat_id is None:
            return False

        width, height = self.image_size_by_id[image_id]
        x1, y1, x2, y2 = _bbox_to_xyxy(raw_bbox, bbox_format)

        x1 = max(0.0, min(float(width), x1))
        y1 = max(0.0, min(float(height), y1))
        x2 = max(0.0, min(float(width), x2))
        y2 = max(0.0, min(float(height), y2))

        bw = x2 - x1
        bh = y2 - y1
        if bw <= 1e-6 or bh <= 1e-6:
            return False

        ann = {
            "id": self.ann_id,
            "image_id": int(image_id),
            "category_id": int(cat_id),
            "bbox": [float(x1), float(y1), float(bw), float(bh)],
            "area": float(bw * bh),
            "iscrowd": 0,
        }
        self._anns_fp.write(json.dumps(ann, ensure_ascii=False) + "\n")
        self.ann_id += 1
        self.valid_boxes += 1
        return True

    def _write_json_array_from_jsonl(self, out_fp, jsonl_path: Path) -> int:
        cnt = 0
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if cnt > 0:
                    out_fp.write(",\n")
                out_fp.write(line)
                cnt += 1
        return cnt

    def finalize(self) -> None:
        self._images_fp.close()
        self._anns_fp.close()
        self.out_coco_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_class_text_path.parent.mkdir(parents=True, exist_ok=True)

        with self.out_coco_path.open("w", encoding="utf-8") as out_fp:
            out_fp.write("{\n")
            out_fp.write('  "images": [\n')
            self._write_json_array_from_jsonl(out_fp, self._images_tmp)
            out_fp.write("\n  ],\n")
            out_fp.write('  "annotations": [\n')
            self._write_json_array_from_jsonl(out_fp, self._anns_tmp)
            out_fp.write("\n  ],\n")
            out_fp.write('  "categories": [\n')
            first = True
            for cid in sorted(self.category_id_to_name.keys()):
                cat = {"id": int(cid), "name": self.category_id_to_name[cid]}
                if not first:
                    out_fp.write(",\n")
                out_fp.write(json.dumps(cat, ensure_ascii=False))
                first = False
            out_fp.write("\n  ]\n")
            out_fp.write("}\n")

        class_text = [[self.category_id_to_name[cid]] for cid in sorted(self.category_id_to_name.keys())]
        with self.out_class_text_path.open("w", encoding="utf-8") as f:
            json.dump(class_text, f, ensure_ascii=False, indent=2)

    def save_classes(self, out_classes_path: Path) -> None:
        out_classes_path.parent.mkdir(parents=True, exist_ok=True)
        class_names = [self.category_id_to_name[cid] for cid in sorted(self.category_id_to_name.keys())]
        with out_classes_path.open("w", encoding="utf-8") as f:
            json.dump(class_names, f, ensure_ascii=False, indent=2)


def detect_source_type(ann_path: Path) -> str:
    suffix = ann_path.suffix.lower()
    if suffix == ".jsonl":
        with ann_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict) and "detection" in obj:
                    return "odvg_detection_jsonl"
                if isinstance(obj, dict) and "grounding" in obj:
                    return "grounding_jsonl"
                break
    elif suffix == ".json":
        with ann_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and "images" in obj and "annotations" in obj:
            return "grounding_coco_json"
    raise ValueError("Cannot auto detect source type. Please set --source-type explicitly.")


def resolve_bbox_format(source_type: str, user_fmt: str) -> str:
    if user_fmt != "auto":
        return user_fmt
    if source_type == "odvg_detection_jsonl":
        return "xyxy"
    if source_type == "grounding_jsonl":
        return "xyxy"
    if source_type == "grounding_coco_json":
        return "xywh"
    raise ValueError(f"Unknown source type: {source_type}")


def convert_odvg_detection_jsonl(
    ann_path: Path,
    builder: StreamingCocoBuilder,
    bbox_format: str,
    file_name_prefix: str,
) -> Tuple[int, int]:
    image_cnt = 0
    ann_cnt = 0
    with ann_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            file_name_raw = obj.get("filename")
            if not file_name_raw:
                continue
            file_name = f"{file_name_prefix}{file_name_raw}"
            width = int(float(obj.get("width", 0)))
            height = int(float(obj.get("height", 0)))
            image_id = builder.register_image(
                image_key=file_name_raw,
                file_name=file_name,
                width=width,
                height=height,
            )
            image_cnt += 1
            instances = obj.get("detection", {}).get("instances", [])
            for ins in instances:
                cat = ins.get("category", "")
                if not cat:
                    cat = f"label_{ins.get('label', -1)}"
                bbox = ins.get("bbox")
                if not _is_box4(bbox):
                    continue
                ok = builder.add_annotation(
                    image_id=image_id,
                    raw_bbox=[float(v) for v in bbox],
                    bbox_format=bbox_format,
                    raw_category=str(cat),
                )
                if ok:
                    ann_cnt += 1
    return image_cnt, ann_cnt


def convert_grounding_jsonl(
    ann_path: Path,
    builder: StreamingCocoBuilder,
    bbox_format: str,
    file_name_prefix: str,
) -> Tuple[int, int]:
    image_cnt = 0
    ann_cnt = 0
    with ann_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            file_name_raw = obj.get("filename")
            if not file_name_raw:
                continue
            file_name = f"{file_name_prefix}{file_name_raw}"
            width = int(float(obj.get("width", 0)))
            height = int(float(obj.get("height", 0)))
            image_id = builder.register_image(
                image_key=file_name_raw,
                file_name=file_name,
                width=width,
                height=height,
            )
            image_cnt += 1

            regions = obj.get("grounding", {}).get("regions", [])
            for region in regions:
                phrase = region.get("phrase", "")
                if not phrase:
                    continue
                for box in _iter_boxes(region.get("bbox")):
                    ok = builder.add_annotation(
                        image_id=image_id,
                        raw_bbox=box,
                        bbox_format=bbox_format,
                        raw_category=phrase,
                    )
                    if ok:
                        ann_cnt += 1
    return image_cnt, ann_cnt


def _derive_category_from_ann(
    ann: Dict[str, Any],
    caption: str,
    category_id_to_name: Dict[int, str],
) -> str:
    for key in ("phrase", "category", "text", "name"):
        val = ann.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    cat_id = ann.get("category_id")
    if isinstance(cat_id, int) and cat_id in category_id_to_name:
        return category_id_to_name[cat_id]

    if "tokens_positive" in ann:
        phrase = _phrase_from_caption(caption, ann["tokens_positive"])
        if phrase:
            return phrase
    return ""


def convert_grounding_coco_json(
    ann_path: Path,
    builder: StreamingCocoBuilder,
    bbox_format: str,
    file_name_prefix: str,
) -> Tuple[int, int]:
    with ann_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    category_id_to_name: Dict[int, str] = {}
    for cat in obj.get("categories", []):
        if isinstance(cat, dict) and isinstance(cat.get("id"), int):
            name = str(cat.get("name", "")).strip()
            if name:
                category_id_to_name[int(cat["id"])] = name

    sent_image_to_real: Dict[int, Tuple[int, str]] = {}
    image_cnt = 0
    ann_cnt = 0

    for im in obj.get("images", []):
        sid = im.get("id")
        file_name_raw = im.get("file_name", im.get("filename", ""))
        if sid is None or not file_name_raw:
            continue
        width = int(float(im.get("width", 0)))
        height = int(float(im.get("height", 0)))
        original_img_id = im.get("original_img_id", sid)
        image_key = f"{original_img_id}::{file_name_raw}"
        image_id = builder.register_image(
            image_key=image_key,
            file_name=f"{file_name_prefix}{file_name_raw}",
            width=width,
            height=height,
        )
        sent_image_to_real[int(sid)] = (image_id, str(im.get("caption", "")))
        image_cnt += 1

    for ann in obj.get("annotations", []):
        sid = ann.get("image_id")
        if not isinstance(sid, int) or sid not in sent_image_to_real:
            continue
        image_id, caption = sent_image_to_real[sid]
        category = _derive_category_from_ann(ann, caption, category_id_to_name)
        if not category:
            continue
        for box in _iter_boxes(ann.get("bbox")):
            ok = builder.add_annotation(
                image_id=image_id,
                raw_bbox=box,
                bbox_format=bbox_format,
                raw_category=category,
            )
            if ok:
                ann_cnt += 1
    return image_cnt, ann_cnt


def main() -> None:
    args = parse_args()
    ann_path = Path(args.ann)
    source_type = args.source_type
    if source_type == "auto":
        source_type = detect_source_type(ann_path)
    bbox_format = resolve_bbox_format(source_type, args.bbox_format)

    out_coco = Path(args.out_coco)
    out_class_text = Path(args.out_class_text)
    out_classes = Path(args.out_classes) if args.out_classes else None
    builder = StreamingCocoBuilder(
        out_coco_path=out_coco,
        out_class_text_path=out_class_text,
        lowercase=args.lowercase,
        strip_punct=args.strip_punct,
    )

    try:
        if source_type == "odvg_detection_jsonl":
            image_cnt, ann_cnt = convert_odvg_detection_jsonl(
                ann_path=ann_path,
                builder=builder,
                bbox_format=bbox_format,
                file_name_prefix=args.file_name_prefix,
            )
        elif source_type == "grounding_jsonl":
            image_cnt, ann_cnt = convert_grounding_jsonl(
                ann_path=ann_path,
                builder=builder,
                bbox_format=bbox_format,
                file_name_prefix=args.file_name_prefix,
            )
        elif source_type == "grounding_coco_json":
            image_cnt, ann_cnt = convert_grounding_coco_json(
                ann_path=ann_path,
                builder=builder,
                bbox_format=bbox_format,
                file_name_prefix=args.file_name_prefix,
            )
        else:
            raise ValueError(f"Unsupported source type: {source_type}")

        builder.finalize()
        if out_classes is not None:
            builder.save_classes(out_classes)
    finally:
        builder.close()

    print(f"[source_type] {source_type}")
    print(f"[bbox_format] {bbox_format}")
    print(f"[images_seen] {image_cnt}")
    print(f"[annotations_written] {ann_cnt}")
    print(f"[valid_boxes] {builder.valid_boxes}/{builder.total_boxes}")
    print(f"[num_categories] {len(builder.category_name_to_id)}")
    print(f"[save_coco] {out_coco}")
    print(f"[save_class_text] {out_class_text}")
    if out_classes is not None:
        print(f"[save_classes] {out_classes}")


if __name__ == "__main__":
    main()

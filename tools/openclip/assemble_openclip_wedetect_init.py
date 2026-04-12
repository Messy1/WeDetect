import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Assemble OpenCLIP backbone init + WeDetect detector init into one checkpoint."
    )
    parser.add_argument(
        "--backbone-init",
        type=str,
        required=True,
        help="Checkpoint containing backbone.image_model.* and backbone.text_model.*",
    )
    parser.add_argument(
        "--detector-init",
        type=str,
        required=True,
        help="Checkpoint containing neck.* and bbox_head.*",
    )
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def extract_state_dict(ckpt_obj) -> Tuple[Dict[str, torch.Tensor], str]:
    if isinstance(ckpt_obj, dict):
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"], "state_dict"
        if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            return ckpt_obj["model"], "model"
        if all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return ckpt_obj, "raw_dict"
    raise ValueError("Cannot infer state_dict from checkpoint.")


def main() -> None:
    args = parse_args()

    b_obj = torch.load(args.backbone_init, map_location="cpu")
    d_obj = torch.load(args.detector_init, map_location="cpu")
    b_sd, b_field = extract_state_dict(b_obj)
    d_sd, d_field = extract_state_dict(d_obj)

    if not any(k.startswith("backbone.image_model.") for k in b_sd):
        raise ValueError("backbone-init has no backbone.image_model.* keys.")
    if not any(k.startswith("backbone.text_model.") for k in b_sd):
        raise ValueError("backbone-init has no backbone.text_model.* keys.")
    if not any(k.startswith("neck.") for k in d_sd):
        raise ValueError("detector-init has no neck.* keys.")
    if not any(k.startswith("bbox_head.") for k in d_sd):
        raise ValueError("detector-init has no bbox_head.* keys.")

    merged = {}
    merged.update(d_sd)
    overlap = set(merged.keys()) & set(b_sd.keys())
    merged.update(b_sd)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": merged}, str(out_path))

    print(f"[backbone-init] {args.backbone_init} ({b_field})")
    print(f"[detector-init] {args.detector_init} ({d_field})")
    print(f"[save] {out_path}")
    print(
        "[merge] "
        f"overlap={len(overlap)} "
        f"image={sum(k.startswith('backbone.image_model.') for k in merged)} "
        f"text={sum(k.startswith('backbone.text_model.') for k in merged)} "
        f"neck={sum(k.startswith('neck.') for k in merged)} "
        f"bbox_head={sum(k.startswith('bbox_head.') for k in merged)}"
    )


if __name__ == "__main__":
    main()


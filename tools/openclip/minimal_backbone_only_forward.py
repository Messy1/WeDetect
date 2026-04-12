import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch

from wedetect.models.backbones.mm_backbone import (
    ConvNextVisionBackbone,
    OpenCLIPTextLanguageBackbone,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Minimal forward check: load only new OpenCLIP-initialized backbone."
    )
    parser.add_argument(
        "--backbone-init",
        type=str,
        required=True,
        help="Checkpoint with backbone.image_model.* and backbone.text_model.*",
    )
    parser.add_argument(
        "--openclip-model",
        type=str,
        default="laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg",
    )
    parser.add_argument("--openclip-pretrained", type=str, default="")
    parser.add_argument(
        "--vision-model-size",
        type=str,
        default="base",
        choices=["tiny", "base", "large", "xlarge"],
    )
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", type=str, default="cuda")
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
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    vision = ConvNextVisionBackbone(model_name=args.vision_model_size, frozen_modules=[])
    text = OpenCLIPTextLanguageBackbone(
        model_name=args.openclip_model,
        pretrained=args.openclip_pretrained or None,
        output_dim=768,
        frozen_modules=[],
    )

    ckpt = torch.load(args.backbone_init, map_location="cpu")
    state_dict, field = extract_state_dict(ckpt)
    img_sd = {}
    txt_sd = {}
    for k, v in state_dict.items():
        if k.startswith("backbone.image_model."):
            img_sd[k[len("backbone.image_model.") :]] = v
        elif k.startswith("backbone.text_model."):
            txt_sd[k[len("backbone.text_model.") :]] = v

    if len(img_sd) == 0 or len(txt_sd) == 0:
        raise ValueError(
            "Invalid backbone-init checkpoint. "
            "Need both backbone.image_model.* and backbone.text_model.*"
        )

    msg_img = vision.load_state_dict(img_sd, strict=False)
    msg_txt = text.load_state_dict(txt_sd, strict=False)
    print(f"[load] checkpoint={args.backbone_init} ({field})")
    print(
        f"[load-image] keys={len(img_sd)} missing={len(msg_img.missing_keys)} "
        f"unexpected={len(msg_img.unexpected_keys)}"
    )
    print(
        f"[load-text] keys={len(txt_sd)} missing={len(msg_txt.missing_keys)} "
        f"unexpected={len(msg_txt.unexpected_keys)}"
    )

    vision.to(device).eval()
    text.to(device).eval()

    dummy = torch.randn(1, 3, args.image_size, args.image_size, device=device)
    texts = [["person", "car", "dog", "traffic light"]]

    with torch.no_grad():
        img_feats = vision(dummy)
        txt_feats = text(texts)

    print("[forward] image feature shapes:")
    for i, feat in enumerate(img_feats):
        print(f"  - level{i}: {tuple(feat.shape)} dtype={feat.dtype}")
    print(f"[forward] text feature shape: {tuple(txt_feats.shape)} dtype={txt_feats.dtype}")
    print(f"[nan-check] image_has_nan={any(torch.isnan(x).any().item() for x in img_feats)}")
    print(f"[nan-check] text_has_nan={torch.isnan(txt_feats).any().item()}")
    print("[ok] backbone-only forward succeeded.")


if __name__ == "__main__":
    main()


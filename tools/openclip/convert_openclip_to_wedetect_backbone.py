import argparse
import json
import re
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch

try:
    import open_clip
except ImportError as e:
    raise ImportError(
        "open_clip is required. Please install open-clip-torch first."
    ) from e

from wedetect.models.backbones.mm_backbone import (
    ConvNextVisionBackbone,
    OpenCLIPTextLanguageBackbone,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Convert OpenCLIP (ConvNeXt) backbone to WeDetect-compatible checkpoint."
    )
    parser.add_argument(
        "--openclip-model",
        type=str,
        default="laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg",
    )
    parser.add_argument(
        "--openclip-pretrained",
        type=str,
        default="",
        help="Optional pretrained tag for non-hf-hub OpenCLIP models.",
    )
    parser.add_argument(
        "--vision-model-size",
        type=str,
        default="base",
        choices=["tiny", "base", "large", "xlarge"],
    )
    parser.add_argument("--text-output-dim", type=int, default=768)
    parser.add_argument("--text-max-length", type=int, default=77)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--report-json", type=str, default="")
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def normalize_model_name(model_name: str, pretrained: str) -> str:
    if "/" in model_name and not model_name.startswith("hf-hub:") and not pretrained:
        return f"hf-hub:{model_name}"
    return model_name


def create_openclip_model(model_name: str, pretrained: str):
    kwargs = dict(model_name=model_name, device="cpu", precision="fp32")
    if pretrained:
        return open_clip.create_model_and_transforms(pretrained=pretrained, **kwargs)
    try:
        return open_clip.create_model_and_transforms(pretrained=None, **kwargs)
    except Exception:
        return open_clip.create_model_and_transforms(**kwargs)


def extract_visual_state_dict(openclip_state: Dict[str, torch.Tensor]) -> Tuple[str, Dict[str, torch.Tensor]]:
    candidate_prefixes = ["visual.trunk.", "visual."]
    for prefix in candidate_prefixes:
        sub = {k[len(prefix):]: v for k, v in openclip_state.items() if k.startswith(prefix)}
        if len(sub) > 0:
            return prefix, sub
    raise ValueError("Cannot find visual.* keys in OpenCLIP state_dict.")


def map_visual_key_to_wedetect(k: str) -> Optional[str]:
    # Already in official ConvNeXt style.
    if (
        k.startswith("downsample_layers.")
        or k.startswith("stages.")
        or k.startswith("norm.")
        or k.startswith("head.")
    ):
        return f"model.{k}"

    # timm ConvNeXt style (used by OpenCLIP timm visual trunk)
    if k.startswith("stem.0."):
        return "model.downsample_layers.0.0." + k[len("stem.0.") :]
    if k.startswith("stem.1."):
        return "model.downsample_layers.0.1." + k[len("stem.1.") :]

    m = re.match(r"^stages\.(\d+)\.downsample\.norm\.(weight|bias)$", k)
    if m:
        stage = int(m.group(1))
        param = m.group(2)
        return f"model.downsample_layers.{stage}.0.{param}"

    m = re.match(r"^stages\.(\d+)\.downsample\.conv\.(weight|bias)$", k)
    if m:
        stage = int(m.group(1))
        param = m.group(2)
        return f"model.downsample_layers.{stage}.1.{param}"

    m = re.match(r"^stages\.(\d+)\.blocks\.(\d+)\.conv_dw\.(weight|bias)$", k)
    if m:
        s, b, p = int(m.group(1)), int(m.group(2)), m.group(3)
        return f"model.stages.{s}.{b}.dwconv.{p}"

    m = re.match(r"^stages\.(\d+)\.blocks\.(\d+)\.norm\.(weight|bias)$", k)
    if m:
        s, b, p = int(m.group(1)), int(m.group(2)), m.group(3)
        return f"model.stages.{s}.{b}.norm.{p}"

    m = re.match(r"^stages\.(\d+)\.blocks\.(\d+)\.mlp\.fc1\.(weight|bias)$", k)
    if m:
        s, b, p = int(m.group(1)), int(m.group(2)), m.group(3)
        return f"model.stages.{s}.{b}.pwconv1.{p}"

    m = re.match(r"^stages\.(\d+)\.blocks\.(\d+)\.mlp\.fc2\.(weight|bias)$", k)
    if m:
        s, b, p = int(m.group(1)), int(m.group(2)), m.group(3)
        return f"model.stages.{s}.{b}.pwconv2.{p}"

    m = re.match(r"^stages\.(\d+)\.blocks\.(\d+)\.gamma$", k)
    if m:
        s, b = int(m.group(1)), int(m.group(2))
        return f"model.stages.{s}.{b}.gamma"

    if k.startswith("norm_pre."):
        return "model.norm." + k[len("norm_pre.") :]
    if k.startswith("norm."):
        return "model.norm." + k[len("norm.") :]

    # Classification head in OpenCLIP visual trunk is not needed for WeDetect.
    if k.startswith("head.") or k.startswith("global_pool"):
        return None

    return None


def build_wedetect_vision_state(
    visual_state: Dict[str, torch.Tensor],
    vision_model_size: str,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    vision_model = ConvNextVisionBackbone(model_name=vision_model_size, frozen_modules=[])
    target_state = vision_model.state_dict()

    mapped = {}
    unmapped = 0
    shape_mismatch = 0
    loaded = 0

    for src_key, value in visual_state.items():
        dst_key = map_visual_key_to_wedetect(src_key)
        if dst_key is None:
            unmapped += 1
            continue
        if dst_key not in target_state:
            unmapped += 1
            continue
        if tuple(target_state[dst_key].shape) != tuple(value.shape):
            shape_mismatch += 1
            continue
        mapped[dst_key] = value
        loaded += 1

    msg = vision_model.load_state_dict(mapped, strict=False)
    stats = {
        "mapped_loaded": loaded,
        "unmapped_or_ignored": unmapped,
        "shape_mismatch": shape_mismatch,
        "missing_after_load": len(msg.missing_keys),
        "unexpected_after_load": len(msg.unexpected_keys),
        "target_num_params": len(target_state),
    }
    return vision_model.state_dict(), stats


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    model_name = normalize_model_name(args.openclip_model, args.openclip_pretrained)
    openclip_model, _, _ = create_openclip_model(model_name, args.openclip_pretrained)
    openclip_sd = openclip_model.state_dict()
    visual_prefix, visual_sd = extract_visual_state_dict(openclip_sd)

    vision_state, vision_stats = build_wedetect_vision_state(
        visual_state=visual_sd,
        vision_model_size=args.vision_model_size,
    )

    text_model = OpenCLIPTextLanguageBackbone(
        model_name=args.openclip_model,
        pretrained=args.openclip_pretrained or None,
        output_dim=args.text_output_dim,
        max_length=args.text_max_length,
        frozen_modules=[],
    )
    text_state = text_model.state_dict()

    export_state = {}
    for key, value in vision_state.items():
        export_state[f"backbone.image_model.{key}"] = value.cpu()
    for key, value in text_state.items():
        export_state[f"backbone.text_model.{key}"] = value.cpu()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": export_state}, str(out_path))

    report = {
        "openclip_model": args.openclip_model,
        "resolved_openclip_model": model_name,
        "visual_prefix": visual_prefix,
        "vision_stats": vision_stats,
        "vision_export_keys": sum(k.startswith("backbone.image_model.") for k in export_state),
        "text_export_keys": sum(k.startswith("backbone.text_model.") for k in export_state),
        "text_clip_embed_dim": int(text_model.clip_embed_dim),
        "text_output_dim": int(args.text_output_dim),
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[save] {out_path}")

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[save] {report_path}")


if __name__ == "__main__":
    main()


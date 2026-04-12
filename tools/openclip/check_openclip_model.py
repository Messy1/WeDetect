import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

try:
    import open_clip
except ImportError as e:
    raise ImportError(
        "open_clip is required. Please install open-clip-torch first."
    ) from e


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Inspect OpenCLIP model for WeDetect integration.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg",
        help="OpenCLIP model id. For HF hub ids, plain `org/name` is accepted.",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="",
        help="OpenCLIP pretrained tag (optional, not used for hf-hub ids).",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--output-json",
        type=str,
        default="work_dirs/openclip_inspect/openclip_model_info.json",
    )
    parser.add_argument(
        "--texts",
        type=str,
        nargs="+",
        default=["a photo of a person", "a photo of a car", "a photo of a dog"],
    )
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def normalize_model_name(model_name: str, pretrained: str) -> str:
    if "/" in model_name and not model_name.startswith("hf-hub:") and not pretrained:
        return f"hf-hub:{model_name}"
    return model_name


def create_openclip_model(
    model_name: str,
    pretrained: str,
    device: str,
):
    kwargs = dict(model_name=model_name, device=device, precision="fp32")
    if pretrained:
        return open_clip.create_model_and_transforms(pretrained=pretrained, **kwargs)
    try:
        return open_clip.create_model_and_transforms(pretrained=None, **kwargs)
    except Exception:
        return open_clip.create_model_and_transforms(**kwargs)


def create_openclip_tokenizer(model_name: str):
    try:
        return open_clip.get_tokenizer(model_name)
    except Exception:
        if model_name.startswith("hf-hub:"):
            return open_clip.get_tokenizer(model_name[len("hf-hub:") :])
        raise


def infer_image_size(model: torch.nn.Module, preprocess) -> Any:
    image_size = getattr(getattr(model, "visual", None), "image_size", None)
    if image_size is not None:
        return image_size
    if hasattr(preprocess, "transforms"):
        size_candidates: List[Any] = []
        for t in preprocess.transforms:
            if hasattr(t, "size"):
                size_candidates.append(t.size)
        if size_candidates:
            return size_candidates[-1]
    return None


def encode_with_fallback(fn, x):
    try:
        return fn(x, normalize=False)
    except TypeError:
        return fn(x)


def to_python_dtype(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def extract_convnext_signature(visual_trunk) -> Dict[str, Any]:
    sig: Dict[str, Any] = {"stage_depths": None, "stage_dims": None}
    if visual_trunk is None:
        return sig

    stages = getattr(visual_trunk, "stages", None)
    if stages is not None:
        try:
            sig["stage_depths"] = [len(s.blocks) for s in stages]
            stage_dims = []
            for s in stages:
                if hasattr(s, "blocks") and len(s.blocks) > 0 and hasattr(s.blocks[0], "conv_dw"):
                    stage_dims.append(int(s.blocks[0].conv_dw.weight.shape[0]))
            if len(stage_dims) > 0:
                sig["stage_dims"] = stage_dims
        except Exception:
            pass
    return sig


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_name = normalize_model_name(args.model_name, args.pretrained)
    model, preprocess_train, preprocess_val = create_openclip_model(
        model_name=model_name,
        pretrained=args.pretrained,
        device=args.device,
    )
    tokenizer = create_openclip_tokenizer(model_name)

    image_size = infer_image_size(model, preprocess_val)
    if isinstance(image_size, int):
        h = w = image_size
    elif isinstance(image_size, (list, tuple)) and len(image_size) >= 2:
        h, w = int(image_size[0]), int(image_size[1])
    else:
        h = w = 224

    img = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))
    image_tensor = preprocess_val(img).unsqueeze(0).to(args.device)

    token_ids = tokenizer(args.texts)
    if isinstance(token_ids, dict):
        token_ids = token_ids["input_ids"]
    token_ids = token_ids.to(args.device)

    model.eval()
    with torch.no_grad():
        image_feat = encode_with_fallback(model.encode_image, image_tensor)
        text_feat = encode_with_fallback(model.encode_text, token_ids)

    visual = getattr(model, "visual", None)
    visual_trunk = getattr(visual, "trunk", None)
    convnext_sig = extract_convnext_signature(visual_trunk)
    text_projection = getattr(model, "text_projection", None)

    info: Dict[str, Any] = {
        "input_model_name": args.model_name,
        "resolved_model_name": model_name,
        "pretrained": args.pretrained,
        "visual_class": type(visual).__name__ if visual is not None else None,
        "visual_has_trunk": visual_trunk is not None,
        "visual_trunk_class": type(visual_trunk).__name__ if visual_trunk is not None else None,
        "visual_is_timm_like": visual_trunk is not None,
        "visual_contains_convnext": (
            "convnext" in str(type(visual_trunk)).lower()
            or "convnext" in str(type(visual)).lower()
        ),
        "visual_convnext_stage_depths": convnext_sig["stage_depths"],
        "visual_convnext_stage_dims": convnext_sig["stage_dims"],
        "wedetect_convnext_base_expected_depths": [3, 3, 27, 3],
        "wedetect_convnext_base_expected_dims": [128, 256, 512, 1024],
        "text_embed_dim": int(text_feat.shape[-1]),
        "text_projection_shape": list(text_projection.shape) if text_projection is not None else None,
        "tokenizer_type": type(tokenizer).__name__,
        "context_length": int(getattr(model, "context_length", token_ids.shape[-1])),
        "image_preprocess_size": [h, w],
        "encode_image_shape": list(image_feat.shape),
        "encode_image_dtype": to_python_dtype(image_feat.dtype),
        "encode_text_shape": list(text_feat.shape),
        "encode_text_dtype": to_python_dtype(text_feat.dtype),
        "preprocess_train_type": type(preprocess_train).__name__,
        "preprocess_val_type": type(preprocess_val).__name__,
        "state_dict_num_keys": int(len(model.state_dict())),
    }

    print(json.dumps(info, indent=2, ensure_ascii=False))

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()

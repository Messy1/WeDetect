import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint

try:
    import open_clip
except ImportError as e:
    raise ImportError(
        "open_clip is required. Please install open-clip-torch first."
    ) from e


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Sanity checks before OpenCLIP stage2 bridge training.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--openclip-model",
        type=str,
        default="",
        help="Optional override. If empty, read from config.model.backbone.text_model.model_name.",
    )
    parser.add_argument("--openclip-pretrained", type=str, default="")
    parser.add_argument("--class-text-json", type=str, default="")
    parser.add_argument("--image-path", type=str, default="")
    parser.add_argument("--num-texts", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def normalize_model_name(model_name: str, pretrained: str) -> str:
    if "/" in model_name and not model_name.startswith("hf-hub:") and not pretrained:
        return f"hf-hub:{model_name}"
    return model_name


def create_openclip_model(model_name: str, pretrained: str, device: str):
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


def pick_texts(class_text_json: str, num_texts: int) -> List[str]:
    default_texts = [
        "person",
        "car",
        "dog",
        "cat",
        "traffic light",
        "bicycle",
        "bus",
        "chair",
    ]
    if not class_text_json:
        return default_texts[:num_texts]

    p = Path(class_text_json)
    if not p.exists():
        print(f"[warn] class_text_json not found: {class_text_json}, fallback to defaults.")
        return default_texts[:num_texts]

    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    texts: List[str] = []
    if isinstance(obj, list):
        for x in obj:
            if isinstance(x, list) and len(x) > 0:
                texts.append(str(x[0]))
            elif isinstance(x, str):
                texts.append(x)
            if len(texts) >= num_texts:
                break
    if len(texts) == 0:
        texts = default_texts[:num_texts]
    return texts


def load_image_for_openclip(
    image_path: str,
    preprocess,
    fallback_size: int = 224,
) -> torch.Tensor:
    if image_path and Path(image_path).exists():
        image = Image.open(image_path).convert("RGB")
    else:
        image = Image.fromarray(np.zeros((fallback_size, fallback_size, 3), dtype=np.uint8))
    return preprocess(image).unsqueeze(0)


def summarize_grad_norm(model: torch.nn.Module, prefix: str) -> str:
    vals = []
    for name, p in model.named_parameters():
        if name.startswith(prefix) and p.grad is not None:
            vals.append(float(p.grad.detach().norm().item()))
    if len(vals) == 0:
        return "no_grad"
    return f"count={len(vals)} mean={sum(vals)/len(vals):.4e} max={max(vals):.4e}"


def reduce_loss_dict(losses: Dict[str, Any]) -> torch.Tensor:
    total = None
    for _, v in losses.items():
        if isinstance(v, torch.Tensor):
            cur = v.mean()
        elif isinstance(v, (list, tuple)):
            cur = sum(x.mean() for x in v if isinstance(x, torch.Tensor))
        else:
            continue
        total = cur if total is None else total + cur
    if total is None:
        raise ValueError("No tensor losses found.")
    return total


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = Config.fromfile(args.config)
    cfg.launcher = "none"
    cfg.load_from = args.checkpoint
    cfg.work_dir = cfg.get("work_dir", "./work_dirs/sanity_openclip_bridge")

    runner = Runner.from_cfg(cfg)
    model = runner.model.to(device)
    load_checkpoint(model, args.checkpoint, map_location="cpu", strict=False)

    # ------------------------
    # 1) text side sanity
    # ------------------------
    text_candidates = pick_texts(args.class_text_json, args.num_texts)
    with torch.no_grad():
        txt = model.backbone.forward_text([text_candidates])
    txt_has_nan = bool(torch.isnan(txt).any().item())
    print(
        "[sanity:text] "
        f"shape={tuple(txt.shape)} dtype={txt.dtype} has_nan={txt_has_nan}"
    )

    # ------------------------
    # 2) OpenCLIP image-text sanity
    # ------------------------
    cfg_openclip_name = cfg.model["backbone"]["text_model"].get("model_name", "")
    cfg_openclip_pretrained = cfg.model["backbone"]["text_model"].get("pretrained", "")
    openclip_model_name = args.openclip_model or cfg_openclip_name
    openclip_pretrained = args.openclip_pretrained or cfg_openclip_pretrained
    openclip_model_name = normalize_model_name(openclip_model_name, openclip_pretrained)

    openclip_model, _, preprocess_val = create_openclip_model(
        model_name=openclip_model_name,
        pretrained=openclip_pretrained,
        device=str(device),
    )
    tokenizer = create_openclip_tokenizer(openclip_model_name)
    image_tensor = load_image_for_openclip(args.image_path, preprocess_val).to(device)
    token_ids = tokenizer(text_candidates)
    if isinstance(token_ids, dict):
        token_ids = token_ids["input_ids"]
    token_ids = token_ids.to(device)

    openclip_model.eval()
    with torch.no_grad():
        try:
            img_feat = openclip_model.encode_image(image_tensor, normalize=False)
        except TypeError:
            img_feat = openclip_model.encode_image(image_tensor)
        try:
            txt_feat = openclip_model.encode_text(token_ids, normalize=False)
        except TypeError:
            txt_feat = openclip_model.encode_text(token_ids)
        img_feat = F.normalize(img_feat.float(), dim=-1)
        txt_feat = F.normalize(txt_feat.float(), dim=-1)
        sim = img_feat @ txt_feat.t()
    print(
        "[sanity:openclip] "
        f"sim_shape={tuple(sim.shape)} sim_min={sim.min().item():.4f} "
        f"sim_max={sim.max().item():.4f} sim_mean={sim.mean().item():.4f}"
    )

    # ------------------------
    # 3) detector-side dummy forward + backward
    # ------------------------
    train_loader = runner.build_dataloader(cfg.train_dataloader)
    batch = next(iter(train_loader))
    model.train()
    model.zero_grad(set_to_none=True)

    processed = model.data_preprocessor(batch, training=True)
    losses = model.loss(processed["inputs"], processed["data_samples"])
    total_loss = reduce_loss_dict(losses)
    if torch.isnan(total_loss):
        raise ValueError("Detector sanity failed: total loss is NaN.")
    total_loss.backward()

    has_nan_grad = False
    for _, p in model.named_parameters():
        if p.grad is not None and torch.isnan(p.grad).any():
            has_nan_grad = True
            break
    if has_nan_grad:
        raise ValueError("Detector sanity failed: gradient contains NaN.")

    print(f"[sanity:detector] total_loss={float(total_loss.item()):.6f} has_nan_grad={has_nan_grad}")
    print(
        "[sanity:grads] "
        f"text_proj({summarize_grad_norm(model, 'backbone.text_model.text_proj')}) "
        f"neck({summarize_grad_norm(model, 'neck.')}) "
        f"bbox_head({summarize_grad_norm(model, 'bbox_head.')})"
    )
    print("[ok] all sanity checks passed.")


if __name__ == "__main__":
    main()


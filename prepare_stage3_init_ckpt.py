import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge WeDetect detector checkpoint with Stage1 CLIP-aligned backbone weights."
    )
    parser.add_argument("--base-ckpt", type=str, required=True)
    parser.add_argument("--stage1-ckpt", type=str, required=True)
    parser.add_argument("--out-ckpt", type=str, required=True)
    parser.add_argument(
        "--replace-text",
        action="store_true",
        help="Replace backbone.text_model.* in base checkpoint by Stage1 weights.",
    )
    parser.add_argument(
        "--replace-vision",
        action="store_true",
        help="Replace backbone.image_model.* in base checkpoint by Stage1 weights.",
    )
    return parser.parse_args()


def extract_state_dict(ckpt_obj: Dict) -> Tuple[Dict[str, torch.Tensor], str]:
    if not isinstance(ckpt_obj, dict):
        raise ValueError("Checkpoint should be a dict-like object.")

    if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
        return ckpt_obj["state_dict"], "state_dict"
    if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
        return ckpt_obj["model"], "model"

    if all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
        return ckpt_obj, ""

    raise ValueError("Cannot infer state_dict from checkpoint.")


def normalize_stage1_keys(stage1_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    normalized: Dict[str, torch.Tensor] = {}
    for key, value in stage1_state.items():
        if key.startswith("backbone.image_model.") or key.startswith("backbone.text_model."):
            normalized[key] = value
            continue

        if key.startswith("vision_backbone."):
            mapped = "backbone.image_model." + key[len("vision_backbone.") :]
            normalized[mapped] = value
            continue

        if key.startswith("text_backbone."):
            mapped = "backbone.text_model." + key[len("text_backbone.") :]
            normalized[mapped] = value
            continue
    return normalized


def split_stage1_state(stage1_norm_state: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    vision_state: Dict[str, torch.Tensor] = {}
    text_state: Dict[str, torch.Tensor] = {}
    for key, value in stage1_norm_state.items():
        if key.startswith("backbone.image_model."):
            vision_state[key] = value
        elif key.startswith("backbone.text_model."):
            text_state[key] = value
    return vision_state, text_state


def remove_prefix_keys(state: Dict[str, torch.Tensor], prefix: str) -> int:
    to_remove = [k for k in state.keys() if k.startswith(prefix)]
    for k in to_remove:
        state.pop(k)
    return len(to_remove)


def main() -> None:
    args = parse_args()

    if not args.replace_text and not args.replace_vision:
        args.replace_text = True

    base_path = Path(args.base_ckpt)
    stage1_path = Path(args.stage1_ckpt)
    out_path = Path(args.out_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_obj = torch.load(str(base_path), map_location="cpu")
    stage1_obj = torch.load(str(stage1_path), map_location="cpu")

    base_state, base_field = extract_state_dict(base_obj)
    stage1_state_raw, _ = extract_state_dict(stage1_obj)
    stage1_state = normalize_stage1_keys(stage1_state_raw)
    stage1_vision, stage1_text = split_stage1_state(stage1_state)

    if args.replace_text and len(stage1_text) == 0:
        raise ValueError("No backbone.text_model.* keys found in Stage1 checkpoint.")
    if args.replace_vision and len(stage1_vision) == 0:
        raise ValueError("No backbone.image_model.* keys found in Stage1 checkpoint.")

    merged_state = dict(base_state)
    removed_text = 0
    removed_vision = 0
    injected_text = 0
    injected_vision = 0

    if args.replace_text:
        removed_text = remove_prefix_keys(merged_state, "backbone.text_model.")
        merged_state.update(stage1_text)
        injected_text = len(stage1_text)

    if args.replace_vision:
        removed_vision = remove_prefix_keys(merged_state, "backbone.image_model.")
        merged_state.update(stage1_vision)
        injected_vision = len(stage1_vision)

    if base_field == "":
        save_obj = merged_state
    else:
        save_obj = dict(base_obj)
        save_obj[base_field] = merged_state

    torch.save(save_obj, str(out_path))

    print(f"[base] {base_path}")
    print(f"[stage1] {stage1_path}")
    print(f"[save] {out_path}")
    print(
        f"[merge] text: removed={removed_text}, injected={injected_text} | "
        f"vision: removed={removed_vision}, injected={injected_vision}"
    )


if __name__ == "__main__":
    main()

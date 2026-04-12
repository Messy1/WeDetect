import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Extract WeDetect detector-side initialization (neck/head only)."
    )
    parser.add_argument("--input", type=str, required=True, help="Pretrained WeDetect checkpoint.")
    parser.add_argument("--output", type=str, required=True, help="Output checkpoint path.")
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
    ckpt = torch.load(args.input, map_location="cpu")
    state_dict, field = extract_state_dict(ckpt)

    detector_state = {}
    for k, v in state_dict.items():
        if k.startswith("neck.") or k.startswith("bbox_head."):
            detector_state[k] = v.cpu()

    if len(detector_state) == 0:
        raise ValueError(
            "No neck./bbox_head. keys found. "
            "Please verify the input checkpoint is a WeDetect detector checkpoint."
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": detector_state}, str(out_path))

    print(f"[input] {args.input} ({field})")
    print(f"[save] {out_path}")
    print(
        "[extract] "
        f"neck={sum(k.startswith('neck.') for k in detector_state)} "
        f"bbox_head={sum(k.startswith('bbox_head.') for k in detector_state)}"
    )


if __name__ == "__main__":
    main()


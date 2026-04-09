import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from train_stage1_clip import (
    ImageTextDataset,
    Stage1AlignModel,
    clip_loss,
    collate_fn,
    load_image_text_annotations,
    load_text_init,
    load_vision_init,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate stage1 CLIP-like checkpoints")
    parser.add_argument("--ann", type=str, required=True, help="Validation json/jsonl annotation.")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to stage1_epoch_*.pth.")
    parser.add_argument("--vision-init", type=str, default="", help="Load vision backbone from checkpoint.")
    parser.add_argument("--text-init", type=str, default="", help="Load text backbone from checkpoint.")
    parser.add_argument(
        "--proj-init",
        type=str,
        default="",
        help="Load image_proj (and optional logit_scale) from a stage1 checkpoint.",
    )
    parser.add_argument("--data-root", type=str, default="")
    parser.add_argument("--image-key", type=str, default="image")
    parser.add_argument("--text-key", type=str, default="text")

    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--query-chunk-size", type=int, default=512)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")

    # Fallback model args used only when checkpoint does not contain `args`.
    parser.add_argument("--vision-model-size", type=str, default="base", choices=["tiny", "base", "large", "xlarge"])
    parser.add_argument("--text-backbone", type=str, default="llm2clip", choices=["llm2clip", "xlmroberta"])
    parser.add_argument("--llm2clip-model-name", type=str, default="/ssd/wzh/models/LLM2CLIP-Llama-3.2-1B-Instruct-CC-Finetuned")
    parser.add_argument("--xlm-roberta-model-name", type=str, default="./xlm-roberta-base")
    parser.add_argument("--embed-dim", type=int, default=768)
    parser.add_argument("--adapter-dim", type=int, default=64)
    parser.add_argument("--adapter-dropout", type=float, default=0.0)
    parser.add_argument("--disable-adapter", action="store_true")
    parser.add_argument("--text-pooling", type=str, default="eos", choices=["eos", "cls", "mean"])
    parser.add_argument("--llm2clip-max-length", type=int, default=77)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def _build_model_from_ckpt_args(ckpt: Dict, cli_args: argparse.Namespace) -> Stage1AlignModel:
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    if not isinstance(ckpt_args, dict):
        ckpt_args = {}

    model = Stage1AlignModel(
        vision_model_size=ckpt_args.get("vision_model_size", cli_args.vision_model_size),
        text_backbone=ckpt_args.get("text_backbone", cli_args.text_backbone),
        llm2clip_model_name=ckpt_args.get("llm2clip_model_name", cli_args.llm2clip_model_name),
        xlm_roberta_model_name=ckpt_args.get("xlm_roberta_model_name", cli_args.xlm_roberta_model_name),
        embed_dim=int(ckpt_args.get("embed_dim", cli_args.embed_dim)),
        adapter_dim=int(ckpt_args.get("adapter_dim", cli_args.adapter_dim)),
        adapter_dropout=float(ckpt_args.get("adapter_dropout", cli_args.adapter_dropout)),
        use_adapter=not bool(ckpt_args.get("disable_adapter", cli_args.disable_adapter)),
        pooling=ckpt_args.get("text_pooling", cli_args.text_pooling),
        max_length=int(ckpt_args.get("llm2clip_max_length", cli_args.llm2clip_max_length)),
        trust_remote_code=bool(ckpt_args.get("trust_remote_code", cli_args.trust_remote_code)),
    )
    return model


def _build_model_from_cli_args(cli_args: argparse.Namespace) -> Stage1AlignModel:
    model = Stage1AlignModel(
        vision_model_size=cli_args.vision_model_size,
        text_backbone=cli_args.text_backbone,
        llm2clip_model_name=cli_args.llm2clip_model_name,
        xlm_roberta_model_name=cli_args.xlm_roberta_model_name,
        embed_dim=cli_args.embed_dim,
        adapter_dim=cli_args.adapter_dim,
        adapter_dropout=cli_args.adapter_dropout,
        use_adapter=not cli_args.disable_adapter,
        pooling=cli_args.text_pooling,
        max_length=cli_args.llm2clip_max_length,
        trust_remote_code=cli_args.trust_remote_code,
    )
    return model


def load_proj_init(model: Stage1AlignModel, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if not isinstance(ckpt, dict):
        raise ValueError("Unsupported checkpoint format for --proj-init")

    proj_state = {}
    for key, value in ckpt.items():
        if key in ("image_proj.weight", "image_proj.bias", "logit_scale"):
            proj_state[key] = value
        elif key.startswith("module.") and key.replace("module.", "") in (
            "image_proj.weight",
            "image_proj.bias",
            "logit_scale",
        ):
            proj_state[key.replace("module.", "")] = value

    if "image_proj.weight" not in proj_state or "image_proj.bias" not in proj_state:
        raise ValueError(
            "No image_proj weights found in --proj-init checkpoint. "
            "Please provide stage1_epoch_*.pth."
        )

    target_state = model.state_dict()
    loadable = {}
    skipped = 0
    for key, value in proj_state.items():
        if key in target_state and target_state[key].shape == value.shape:
            loadable[key] = value
        else:
            skipped += 1

    msg = model.load_state_dict(loadable, strict=False)
    print(f"[proj-init] loaded from {ckpt_path}")
    print(
        f"[proj-init] loaded keys: {len(loadable)} skipped keys: {skipped} "
        f"missing keys: {len(msg.missing_keys)} unexpected keys: {len(msg.unexpected_keys)}"
    )


def _topk_recall_i2t(
    image_emb: torch.Tensor,
    text_emb: torch.Tensor,
    text_to_img_idx: torch.Tensor,
    ks: List[int],
    chunk_size: int,
) -> Dict[str, float]:
    max_k = max(ks)
    hits = {k: 0 for k in ks}
    num_images = image_emb.size(0)

    for start in range(0, num_images, chunk_size):
        end = min(start + chunk_size, num_images)
        sim = image_emb[start:end] @ text_emb.t()
        topk_idx = sim.topk(max_k, dim=1).indices

        gt_img_idx = torch.arange(start, end, device=image_emb.device)
        topk_img_idx = text_to_img_idx[topk_idx]
        match = topk_img_idx.eq(gt_img_idx[:, None])
        for k in ks:
            hits[k] += int(match[:, :k].any(dim=1).sum().item())

    return {f"i2t_R@{k}": hits[k] / max(1, num_images) for k in ks}


def _topk_recall_t2i(
    text_emb: torch.Tensor,
    image_emb: torch.Tensor,
    text_to_img_idx: torch.Tensor,
    ks: List[int],
    chunk_size: int,
) -> Dict[str, float]:
    max_k = max(ks)
    hits = {k: 0 for k in ks}
    num_texts = text_emb.size(0)

    for start in range(0, num_texts, chunk_size):
        end = min(start + chunk_size, num_texts)
        sim = text_emb[start:end] @ image_emb.t()
        topk_idx = sim.topk(max_k, dim=1).indices
        gt = text_to_img_idx[start:end]
        match = topk_idx.eq(gt[:, None])
        for k in ks:
            hits[k] += int(match[:, :k].any(dim=1).sum().item())

    return {f"t2i_R@{k}": hits[k] / max(1, num_texts) for k in ks}


@torch.no_grad()
def main() -> None:
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

    ckpt_obj: Dict = {}
    mode = "init_only"
    load_msg = None
    ckpt_path_str = args.checkpoint

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt_loaded = torch.load(str(ckpt_path), map_location="cpu")
        ckpt_obj = ckpt_loaded if isinstance(ckpt_loaded, dict) else {}
        state_dict = ckpt_obj["model"] if isinstance(ckpt_obj, dict) and "model" in ckpt_obj else ckpt_loaded
        if not isinstance(state_dict, dict):
            raise ValueError("Unsupported checkpoint format.")

        model = _build_model_from_ckpt_args(ckpt_obj if isinstance(ckpt_obj, dict) else {}, args)
        if any(k.startswith("image_proj.") for k in state_dict.keys()):
            mode = "stage1_ckpt"
            load_msg = model.load_state_dict(state_dict, strict=False)
        else:
            # likely wedetect init checkpoint (or full detector checkpoint) without image_proj.
            mode = "backbone_init_from_checkpoint"
            load_vision_init(model, args.checkpoint)
            load_text_init(model, args.checkpoint)
    else:
        model = _build_model_from_cli_args(args)

    if args.vision_init:
        load_vision_init(model, args.vision_init)
        mode = "backbone_init_from_flags"
    if args.text_init:
        load_text_init(model, args.text_init)
        mode = "backbone_init_from_flags"
    if args.proj_init:
        load_proj_init(model, args.proj_init)
        mode = "backbone_plus_proj_init"

    if not args.checkpoint and not args.vision_init and not args.text_init:
        raise ValueError("Please provide --checkpoint or at least one of --vision-init/--text-init.")

    model.to(device)
    model.eval()

    samples = load_image_text_annotations(args.ann, args.image_key, args.text_key)
    dataset = ImageTextDataset(samples, args.data_root, args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    total_loss = 0.0
    total_acc = 0.0
    total_count = 0

    text_feats_all: List[torch.Tensor] = []
    text_uids: List[int] = []
    image_feat_sum: Dict[int, torch.Tensor] = {}
    image_feat_count: Dict[int, int] = {}

    for images, texts, image_ids in loader:
        images = images.to(device, non_blocking=True)
        image_ids = image_ids.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            image_features, text_features, logit_scale = model(images, texts)
            loss, acc = clip_loss(image_features, text_features, logit_scale, image_ids)

        batch = images.size(0)
        total_loss += float(loss.item()) * batch
        total_acc += float(acc.item()) * batch
        total_count += batch

        image_features_cpu = image_features.float().cpu()
        text_features_cpu = text_features.float().cpu()
        image_ids_cpu = image_ids.cpu().tolist()

        text_feats_all.append(text_features_cpu)
        text_uids.extend(image_ids_cpu)

        for i, uid in enumerate(image_ids_cpu):
            feat = image_features_cpu[i]
            if uid in image_feat_sum:
                image_feat_sum[uid] += feat
                image_feat_count[uid] += 1
            else:
                image_feat_sum[uid] = feat.clone()
                image_feat_count[uid] = 1

    if len(text_feats_all) == 0 or len(image_feat_sum) == 0:
        raise ValueError("No valid features collected from dataset.")

    unique_uids = sorted(image_feat_sum.keys())
    image_emb = torch.stack(
        [image_feat_sum[uid] / float(image_feat_count[uid]) for uid in unique_uids],
        dim=0,
    )
    text_emb = torch.cat(text_feats_all, dim=0)

    uid_to_img_idx = {uid: idx for idx, uid in enumerate(unique_uids)}
    text_to_img_idx = torch.tensor([uid_to_img_idx[uid] for uid in text_uids], dtype=torch.long)

    image_emb = image_emb.to(device)
    text_emb = text_emb.to(device)
    text_to_img_idx = text_to_img_idx.to(device)

    ks = [1, 5, 10]
    i2t = _topk_recall_i2t(
        image_emb=image_emb,
        text_emb=text_emb,
        text_to_img_idx=text_to_img_idx,
        ks=ks,
        chunk_size=args.query_chunk_size,
    )
    t2i = _topk_recall_t2i(
        text_emb=text_emb,
        image_emb=image_emb,
        text_to_img_idx=text_to_img_idx,
        ks=ks,
        chunk_size=args.query_chunk_size,
    )

    val_loss = total_loss / max(1, total_count)
    val_acc = total_acc / max(1, total_count)

    if ckpt_path_str:
        print(f"[ckpt] {ckpt_path_str}")
    if args.vision_init:
        print(f"[vision-init] {args.vision_init}")
    if args.text_init:
        print(f"[text-init] {args.text_init}")
    if args.proj_init:
        print(f"[proj-init] {args.proj_init}")
    if load_msg is not None:
        print(
            f"[load] mode={mode} missing={len(load_msg.missing_keys)} "
            f"unexpected={len(load_msg.unexpected_keys)} "
            f"samples={len(samples)} unique_images={len(unique_uids)}"
        )
    else:
        print(f"[load] mode={mode} samples={len(samples)} unique_images={len(unique_uids)}")
    if mode != "stage1_ckpt":
        print(
            "[warn] image_proj is randomly initialized in this mode. "
            "Results are a rough initialization baseline, not directly comparable "
            "to fully trained stage1 checkpoints."
        )
    print(f"[val] loss={val_loss:.4f} acc={val_acc:.4f}")
    print(
        "[retrieval] "
        + " ".join([f"{k}={v:.4f}" for k, v in {**i2t, **t2i}.items()])
    )


if __name__ == "__main__":
    main()

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from wedetect.models.backbones.mm_backbone import (
    ConvNextVisionBackbone,
    LLM2CLIPLanguageBackbone,
    XLMRobertaLanguageBackbone,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage1 CLIP-like pretraining for WeDetect")
    parser.add_argument("--train-ann", type=str, required=True)
    parser.add_argument("--val-ann", type=str, default="")
    parser.add_argument("--data-root", type=str, default="")
    parser.add_argument("--image-key", type=str, default="image")
    parser.add_argument("--text-key", type=str, default="text")
    parser.add_argument("--output-dir", type=str, default="./work_dirs/stage1_llm2clip")

    parser.add_argument("--vision-model-size", type=str, default="base",
                        choices=["tiny", "base", "large", "xlarge"])
    parser.add_argument("--vision-init", type=str, default="")
    parser.add_argument(
        "--text-backbone",
        type=str,
        default="llm2clip",
        choices=["llm2clip", "xlmroberta"],
    )
    parser.add_argument("--text-init", type=str, default="")
    parser.add_argument("--llm2clip-model-name", type=str, default="/ssd/wzh/models/LLM2CLIP-Llama-3.2-1B-Instruct-CC-Finetuned")
    parser.add_argument("--xlm-roberta-model-name", type=str, default="./xlm-roberta-base")
    parser.add_argument("--llm2clip-max-length", type=int, default=77)
    parser.add_argument("--text-pooling", type=str, default="eos",
                        choices=["eos", "cls", "mean"])
    parser.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument("--embed-dim", type=int, default=768)
    parser.add_argument("--adapter-dim", type=int, default=64)
    parser.add_argument("--adapter-dropout", type=float, default=0.0)
    parser.add_argument("--disable-adapter", action="store_true")
    parser.add_argument("--train-text-base", action="store_true")
    parser.add_argument("--unfreeze-vision", action="store_true")

    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--vision-lr", type=float, default=1e-4)
    parser.add_argument("--text-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=1)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    return parser.parse_args()


def setup_distributed(args: argparse.Namespace) -> Tuple[bool, int, int, int]:
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def seed_everything(seed: int, rank: int = 0) -> None:
    final_seed = seed + rank
    random.seed(final_seed)
    torch.manual_seed(final_seed)
    torch.cuda.manual_seed_all(final_seed)


def _extract_text_from_sample(sample: Dict, text_key: str) -> str:
    if text_key in sample and sample[text_key] is not None:
        return str(sample[text_key])
    if "caption" in sample and sample["caption"] is not None:
        return str(sample["caption"])
    if "conversations" in sample and isinstance(sample["conversations"], list):
        for turn in sample["conversations"]:
            if turn.get("from", "").lower() in ("gpt", "assistant"):
                value = turn.get("value", "")
                if value:
                    return str(value)
    return ""


def load_image_text_annotations(
    ann_path: str,
    image_key: str,
    text_key: str,
) -> List[Tuple[str, str, str]]:
    path = Path(ann_path)
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    if path.suffix == ".jsonl":
        raw_samples = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_samples.append(json.loads(line))
    elif path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            if "data" in obj and isinstance(obj["data"], list):
                raw_samples = obj["data"]
            elif "annotations" in obj and isinstance(obj["annotations"], list):
                raw_samples = obj["annotations"]
            else:
                raise ValueError(f"Unsupported JSON schema in {ann_path}")
        elif isinstance(obj, list):
            raw_samples = obj
        else:
            raise ValueError(f"Unsupported JSON schema in {ann_path}")
    else:
        raise ValueError("Only .json and .jsonl annotations are supported.")

    samples: List[Tuple[str, str, str]] = []
    for sample in raw_samples:
        image_path = sample.get(image_key, sample.get("image", ""))
        text = _extract_text_from_sample(sample, text_key)
        if image_path and text:
            image_uid = sample.get("image_id", image_path)
            samples.append((str(image_path), text, str(image_uid)))
    if len(samples) == 0:
        raise ValueError("No valid image-text pairs found in annotation file.")
    return samples


class ImageTextDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Tuple[str, str, str]],
        data_root: str,
        image_size: int,
    ) -> None:
        self.samples = list(samples)
        self.data_root = Path(data_root) if data_root else None
        self.uid_to_int: Dict[str, int] = {}
        for _, _, uid in self.samples:
            if uid not in self.uid_to_int:
                self.uid_to_int[uid] = len(self.uid_to_int)
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, int]:
        image_rel, text, uid = self.samples[idx]
        image_path = Path(image_rel)
        if not image_path.is_absolute() and self.data_root is not None:
            image_path = self.data_root / image_path
        with Image.open(image_path).convert("RGB") as image:
            image = self.transform(image)
        return image, text, self.uid_to_int[uid]


def collate_fn(
    batch: Sequence[Tuple[torch.Tensor, str, int]]
) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
    images, texts, image_ids = zip(*batch)
    return (
        torch.stack(images, dim=0),
        list(texts),
        torch.tensor(image_ids, dtype=torch.long),
    )


class Stage1AlignModel(nn.Module):
    VISION_LAST_CHANNEL = {
        "tiny": 768,
        "base": 1024,
        "large": 1536,
        "xlarge": 1024,
    }

    def __init__(
        self,
        vision_model_size: str,
        text_backbone: str,
        llm2clip_model_name: str,
        xlm_roberta_model_name: str,
        embed_dim: int,
        adapter_dim: int,
        adapter_dropout: float,
        use_adapter: bool,
        pooling: str,
        max_length: int,
        trust_remote_code: bool,
    ) -> None:
        super().__init__()
        self.vision_backbone = ConvNextVisionBackbone(
            model_name=vision_model_size,
            frozen_modules=[],
        )
        self.text_backbone_name = text_backbone
        if text_backbone == "llm2clip":
            self.text_backbone = LLM2CLIPLanguageBackbone(
                model_name=llm2clip_model_name,
                output_dim=embed_dim,
                adapter_dim=adapter_dim,
                adapter_dropout=adapter_dropout,
                use_adapter=use_adapter,
                pooling=pooling,
                max_length=max_length,
                trust_remote_code=trust_remote_code,
                frozen_modules=[],
            )
        elif text_backbone == "xlmroberta":
            self.text_backbone = XLMRobertaLanguageBackbone(
                model_name=xlm_roberta_model_name,
                model_size=vision_model_size,
                frozen_modules=[],
            )
        else:
            raise ValueError(f"Unsupported text backbone: {text_backbone}")
        self.image_proj = nn.Linear(self.VISION_LAST_CHANNEL[vision_model_size], embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        feats = self.vision_backbone(image)[-1]
        feats = F.adaptive_avg_pool2d(feats, output_size=(1, 1)).flatten(1)
        feats = self.image_proj(feats)
        feats = F.normalize(feats, dim=-1)
        return feats

    def encode_text(self, text: List[str]) -> torch.Tensor:
        nested = [[t] for t in text]
        feats = self.text_backbone(nested).squeeze(1)
        feats = F.normalize(feats, dim=-1)
        return feats

    def forward(self, image: torch.Tensor, text: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_feats = self.encode_image(image)
        txt_feats = self.encode_text(text)
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        return img_feats, txt_feats, logit_scale


def load_vision_init(model: Stage1AlignModel, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    if not isinstance(ckpt, dict):
        raise ValueError("Unsupported checkpoint format for --vision-init")

    vision_state = {}
    for key, value in ckpt.items():
        if key.startswith("backbone.image_model."):
            vision_state[key.replace("backbone.image_model.", "")] = value
        elif key.startswith("backbone.") and "text_model." not in key:
            vision_state[key.replace("backbone.", "")] = value
        elif key.startswith("vision_backbone."):
            vision_state[key.replace("vision_backbone.", "")] = value
    if len(vision_state) == 0:
        raise ValueError("No ConvNeXt-compatible keys found in --vision-init checkpoint.")

    msg = model.vision_backbone.load_state_dict(vision_state, strict=False)
    print(f"[vision-init] loaded from {ckpt_path}")
    print(f"[vision-init] missing keys: {len(msg.missing_keys)} unexpected keys: {len(msg.unexpected_keys)}")


def load_text_init(model: Stage1AlignModel, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    if not isinstance(ckpt, dict):
        raise ValueError("Unsupported checkpoint format for --text-init")

    text_state = {}
    for key, value in ckpt.items():
        if key.startswith("backbone.text_model."):
            text_state[key.replace("backbone.text_model.", "")] = value
        elif key.startswith("text_backbone."):
            text_state[key.replace("text_backbone.", "")] = value

    if len(text_state) == 0:
        raise ValueError("No text backbone keys found in --text-init checkpoint.")

    model_state = model.text_backbone.state_dict()
    loadable_state = {}
    skipped = 0
    for key, value in text_state.items():
        if key in model_state and model_state[key].shape == value.shape:
            loadable_state[key] = value
        else:
            skipped += 1

    if len(loadable_state) == 0:
        raise ValueError(
            "No compatible text keys can be loaded. "
            "Please check --text-backbone and --text-init checkpoint."
        )

    msg = model.text_backbone.load_state_dict(loadable_state, strict=False)
    print(f"[text-init] loaded from {ckpt_path}")
    print(
        f"[text-init] loaded keys: {len(loadable_state)} skipped keys: {skipped} "
        f"missing keys: {len(msg.missing_keys)} unexpected keys: {len(msg.unexpected_keys)}"
    )


def freeze_modules(model: Stage1AlignModel, freeze_vision: bool, train_text_base: bool) -> None:
    # ConvNeXt classification tail (`norm` + `head`) is never used in this
    # stage1 CLIP forward path, so keep it frozen to avoid DDP unused-param errors
    # when vision backbone is unfrozen.
    vision_model = model.vision_backbone.model
    for maybe_unused_name in ("norm", "head"):
        if hasattr(vision_model, maybe_unused_name):
            maybe_unused_module = getattr(vision_model, maybe_unused_name)
            maybe_unused_module.eval()
            for p in maybe_unused_module.parameters():
                p.requires_grad = False

    if freeze_vision:
        for p in model.vision_backbone.parameters():
            p.requires_grad = False
        model.vision_backbone.eval()

    if not train_text_base:
        for p in model.text_backbone.model.parameters():
            p.requires_grad = False
        model.text_backbone.model.eval()


def gather_features(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    image_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not dist.is_available() or not dist.is_initialized():
        return image_features, text_features, image_ids

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    all_image = [torch.zeros_like(image_features) for _ in range(world_size)]
    all_text = [torch.zeros_like(text_features) for _ in range(world_size)]
    all_ids = [torch.zeros_like(image_ids) for _ in range(world_size)]

    dist.all_gather(all_image, image_features.detach())
    dist.all_gather(all_text, text_features.detach())
    dist.all_gather(all_ids, image_ids)
    all_image[rank] = image_features
    all_text[rank] = text_features

    return (
        torch.cat(all_image, dim=0),
        torch.cat(all_text, dim=0),
        torch.cat(all_ids, dim=0),
    )


def _multi_positive_nce_loss(logits: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    # -log( sum(exp(pos_logits)) / sum(exp(all_logits)) )
    # positive_mask shape: [N, M], at least one positive per row.
    neg_inf = torch.finfo(logits.dtype).min
    pos_logits = logits.masked_fill(~positive_mask, neg_inf)
    pos_logsumexp = torch.logsumexp(pos_logits, dim=1)
    all_logsumexp = torch.logsumexp(logits, dim=1)
    return -(pos_logsumexp - all_logsumexp).mean()


def clip_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
    image_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    all_image, all_text, all_ids = gather_features(image_features, text_features, image_ids)

    logits_per_image = logit_scale * image_features @ all_text.t()
    logits_per_text = logit_scale * text_features @ all_image.t()

    # Multi-positive: all captions/images from the same image id are positives.
    positive_mask_i2t = image_ids[:, None].eq(all_ids[None, :])
    positive_mask_t2i = image_ids[:, None].eq(all_ids[None, :])

    loss_i2t = _multi_positive_nce_loss(logits_per_image, positive_mask_i2t)
    loss_t2i = _multi_positive_nce_loss(logits_per_text, positive_mask_t2i)
    loss = 0.5 * (loss_i2t + loss_t2i)

    # Top1 retrieval is correct if predicted sample belongs to any positive pair.
    pred_txt_top1 = logits_per_image.argmax(dim=-1)
    pred_img_top1 = logits_per_text.argmax(dim=-1)
    acc_i2t = all_ids[pred_txt_top1].eq(image_ids).float().mean()
    acc_t2i = all_ids[pred_img_top1].eq(image_ids).float().mean()
    acc = 0.5 * (acc_i2t + acc_t2i)
    return loss, acc


def build_optimizer(model: Stage1AlignModel, args: argparse.Namespace) -> torch.optim.Optimizer:
    vision_params = []
    text_base_params = []
    text_adapter_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("vision_backbone."):
            vision_params.append(param)
        elif name.startswith("text_backbone.model."):
            text_base_params.append(param)
        elif name.startswith("text_backbone."):
            text_adapter_params.append(param)
        else:
            other_params.append(param)

    param_groups = []
    if len(vision_params) > 0:
        param_groups.append(
            {"params": vision_params, "lr": args.vision_lr, "weight_decay": args.weight_decay}
        )
    if len(text_base_params) > 0:
        param_groups.append(
            {"params": text_base_params, "lr": args.text_lr, "weight_decay": args.weight_decay}
        )
    if len(text_adapter_params) > 0:
        param_groups.append(
            {"params": text_adapter_params, "lr": args.text_lr, "weight_decay": args.weight_decay}
        )
    if len(other_params) > 0:
        param_groups.append(
            {"params": other_params, "lr": args.lr, "weight_decay": args.weight_decay}
        )

    if len(param_groups) == 0:
        raise ValueError("No trainable parameters found. Check freeze options.")

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.98), eps=1e-6)
    return optimizer


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return value
    value = value.clone()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    value /= dist.get_world_size()
    return value


def save_checkpoint(
    path: Path,
    model: Stage1AlignModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
) -> None:
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "args": vars(args),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    torch.save(ckpt, str(path))


def export_wedetect_init(path: Path, model: Stage1AlignModel) -> None:
    export_state = {}
    for key, value in model.vision_backbone.state_dict().items():
        export_state[f"backbone.image_model.{key}"] = value.cpu()
    for key, value in model.text_backbone.state_dict().items():
        export_state[f"backbone.text_model.{key}"] = value.cpu()
    torch.save(export_state, str(path))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_count = 0

    for images, texts, image_ids in loader:
        images = images.to(device, non_blocking=True)
        image_ids = image_ids.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            image_features, text_features, logit_scale = model(images, texts)
            loss, acc = clip_loss(image_features, text_features, logit_scale, image_ids)

        batch = images.size(0)
        total_loss += loss.item() * batch
        total_acc += acc.item() * batch
        total_count += batch

    loss_t = torch.tensor(total_loss / max(1, total_count), device=device)
    acc_t = torch.tensor(total_acc / max(1, total_count), device=device)
    loss_t = reduce_mean(loss_t)
    acc_t = reduce_mean(acc_t)
    return float(loss_t.item()), float(acc_t.item())


def main() -> None:
    args = parse_args()
    distributed, rank, world_size, local_rank = setup_distributed(args)
    seed_everything(args.seed, rank)

    is_main = rank == 0
    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda", local_rank if distributed else 0)
    torch.backends.cudnn.benchmark = True

    train_samples = load_image_text_annotations(args.train_ann, args.image_key, args.text_key)
    val_samples = []
    if args.val_ann:
        val_samples = load_image_text_annotations(args.val_ann, args.image_key, args.text_key)

    train_set = ImageTextDataset(train_samples, args.data_root, args.image_size)
    val_set = ImageTextDataset(val_samples, args.data_root, args.image_size) if len(val_samples) > 0 else None

    train_sampler = DistributedSampler(train_set, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if distributed and val_set is not None else None

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

    raw_model = Stage1AlignModel(
        vision_model_size=args.vision_model_size,
        text_backbone=args.text_backbone,
        llm2clip_model_name=args.llm2clip_model_name,
        xlm_roberta_model_name=args.xlm_roberta_model_name,
        embed_dim=args.embed_dim,
        adapter_dim=args.adapter_dim,
        adapter_dropout=args.adapter_dropout,
        use_adapter=not args.disable_adapter,
        pooling=args.text_pooling,
        max_length=args.llm2clip_max_length,
        trust_remote_code=args.trust_remote_code,
    )

    if args.vision_init:
        load_vision_init(raw_model, args.vision_init)
    if args.text_init:
        load_text_init(raw_model, args.text_init)

    freeze_modules(raw_model, freeze_vision=not args.unfreeze_vision, train_text_base=args.train_text_base)
    raw_model.to(device)

    optimizer = build_optimizer(raw_model, args)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = build_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    start_epoch = 0
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        raw_model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt["epoch"]) + 1
        global_step = int(ckpt.get("global_step", 0))
        if is_main:
            print(f"[resume] loaded from {args.resume}, start_epoch={start_epoch}")

    model = raw_model
    if distributed:
        model = DDP(
            raw_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    use_amp = args.amp
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and not args.bf16)

    for epoch in range(start_epoch, args.epochs):
        if distributed:
            assert train_sampler is not None
            train_sampler.set_epoch(epoch)
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)

        model.train()
        if not args.unfreeze_vision:
            raw_model.vision_backbone.eval()
        if not args.train_text_base:
            raw_model.text_backbone.model.eval()

        epoch_start = time.time()
        running_loss = 0.0
        running_acc = 0.0
        running_steps = 0

        for step, (images, texts, image_ids) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            image_ids = image_ids.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                image_features, text_features, logit_scale = model(images, texts)
                loss, acc = clip_loss(image_features, text_features, logit_scale, image_ids)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.max_grad_norm)
                optimizer.step()

            scheduler.step()
            global_step += 1

            running_loss += loss.item()
            running_acc += acc.item()
            running_steps += 1

            if is_main and (step + 1) % args.log_interval == 0:
                avg_loss = running_loss / max(1, running_steps)
                avg_acc = running_acc / max(1, running_steps)
                lr = scheduler.get_last_lr()[0]
                print(
                    f"[epoch {epoch + 1}/{args.epochs}] "
                    f"step {step + 1}/{len(train_loader)} "
                    f"loss={avg_loss:.4f} acc={avg_acc:.4f} lr={lr:.6e} "
                    f"logit_scale={raw_model.logit_scale.exp().item():.4f}"
                )

        train_loss = torch.tensor(running_loss / max(1, running_steps), device=device)
        train_acc = torch.tensor(running_acc / max(1, running_steps), device=device)
        train_loss = reduce_mean(train_loss)
        train_acc = reduce_mean(train_acc)
        epoch_time = time.time() - epoch_start

        val_msg = ""
        if val_loader is not None:
            val_loss, val_acc = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
            val_msg = f" | val_loss={val_loss:.4f} val_acc={val_acc:.4f}"

        if is_main:
            print(
                f"[epoch {epoch + 1}/{args.epochs}] "
                f"train_loss={train_loss.item():.4f} train_acc={train_acc.item():.4f} "
                f"time={epoch_time:.1f}s{val_msg}"
            )

            if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
                ckpt_path = output_dir / f"stage1_epoch_{epoch + 1}.pth"
                save_checkpoint(
                    path=ckpt_path,
                    model=raw_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    args=args,
                )
                init_path = output_dir / f"wedetect_stage1_init_epoch_{epoch + 1}.pth"
                export_wedetect_init(init_path, raw_model)
                print(f"[save] {ckpt_path}")
                print(f"[save] {init_path}")

    cleanup_distributed()


if __name__ == "__main__":
    main()

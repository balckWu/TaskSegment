from __future__ import annotations

"""Fine-tune official CLIPSeg in the unified TaskSegment comparison protocol.

Protocol:
* one model across all six ultrasound domains;
* fixed 512x512 input, target and metric grid;
* one predefined text prompt per domain;
* domain-balanced sampling by default;
* validation macro Dice selects the best checkpoint;
* no test masks, support masks, prototypes, threshold tuning or post-processing.

The default ``decoder`` mode follows the original CLIPSeg design by keeping the
CLIP backbone frozen and fine-tuning only the segmentation decoder. An optional
``vision_decoder`` mode is provided as a stronger medical-domain adaptation
variant and must be reported as a separate experiment.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paperlab.progress import paper_tqdm as tqdm

from tasksegment.configs import (  # noqa: E402
    DOMAIN_NAMES,
    DataConfig,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_LOSS_CONFIG,
    DEFAULT_METRIC_SIZE,
)
from tasksegment.data import get_datasets, set_seed  # noqa: E402
from tasksegment.training.augmentations import augment_episode_medical  # noqa: E402
from tasksegment.training.losses import segmentation_loss  # noqa: E402
from tasksegment.training.memory import make_domain_schedule  # noqa: E402
from tasksegment.training.metrics import binary_dice, binary_hd95, binary_iou  # noqa: E402

try:
    from transformers import AutoTokenizer, CLIPSegForImageSegmentation, CLIPSegProcessor
except ImportError as exc:  # pragma: no cover - gives a useful server-side error
    raise ImportError(
        "CLIPSeg requires Hugging Face Transformers. Install with: "
        "python -m pip install -r requirements-clipseg.txt"
    ) from exc


def parse_image_size(values: Sequence[int]) -> Tuple[int, int]:
    if len(values) == 1:
        size = (int(values[0]), int(values[0]))
    elif len(values) == 2:
        size = (int(values[0]), int(values[1]))
    else:
        raise ValueError("--image-size accepts either one value or H W")
    if size != tuple(DEFAULT_IMAGE_SIZE):
        raise ValueError(
            "The comparison protocol requires a fixed "
            f"{DEFAULT_IMAGE_SIZE[0]}x{DEFAULT_IMAGE_SIZE[1]} grid; got {size}."
        )
    return size


def resolve_device(value: str) -> torch.device:
    if str(value).startswith("cuda") and not torch.cuda.is_available():
        print("[warning] CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(value)


def amp_context(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return nullcontext()
    if hasattr(torch, "autocast"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.cuda.amp.autocast(enabled=True)


def make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def safe_torch_load(path: str | Path, map_location: str | torch.device = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, target)


def write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, ensure_ascii=False)


def append_history(path: str | Path, row: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_header = not target.exists()
    with target.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(dict(row))


def set_cosine_lr(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    num_epochs: int,
    warmup_epochs: int,
) -> List[float]:
    if warmup_epochs > 0 and epoch < warmup_epochs:
        factor = float(epoch + 1) / float(warmup_epochs)
    else:
        remaining = max(1, num_epochs - warmup_epochs - 1)
        progress = min(max((epoch - warmup_epochs) / remaining, 0.0), 1.0)
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))

    values: List[float] = []
    for group in optimizer.param_groups:
        base_lr = float(group["initial_lr"])
        min_lr = float(group.get("min_lr", 0.0))
        current = min_lr + (base_lr - min_lr) * factor
        group["lr"] = current
        values.append(current)
    return values


def load_prompts(path: str | Path) -> Dict[str, str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    prompts = {str(key): str(value).strip() for key, value in payload.items()}
    missing = [domain for domain in DOMAIN_NAMES if not prompts.get(domain)]
    if missing:
        raise ValueError(f"Prompt file is missing domains: {missing}")
    return {domain: prompts[domain] for domain in DOMAIN_NAMES}


def sample_domain_batch(dataset, domain: str, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    indices = dataset.get_organ_samples(domain, int(batch_size), positive_only=False)
    if len(indices) != int(batch_size):
        pool = dataset.get_organ_indices(domain, positive_only=False)
        extra = np.random.choice(
            pool, size=int(batch_size) - len(indices), replace=True
        ).tolist()
        indices = list(indices) + extra
    images = torch.stack([dataset[index][0] for index in indices], dim=0)
    masks = torch.stack([dataset[index][1] for index in indices], dim=0)
    return images, masks


def domain_to_modality(domain: str) -> str:
    if domain in {"thyroid", "TN3K"}:
        return "thyroid"
    if domain in {"BUSI_WHU", "BUS-BRA"}:
        return "BUSI"
    return "generic"


def configure_trainable_parameters(model: nn.Module, train_mode: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.decoder.parameters():
        parameter.requires_grad = True

    if train_mode == "vision_decoder":
        for parameter in model.clip.vision_model.parameters():
            parameter.requires_grad = True
    elif train_mode != "decoder":
        raise ValueError(f"Unsupported train mode: {train_mode}")


def restore_frozen_eval_state(model: nn.Module, train_mode: str) -> None:
    """Keep frozen CLIP modules deterministic after calling model.train()."""
    model.clip.text_model.eval()
    if train_mode == "decoder":
        model.clip.eval()
        model.decoder.train()
    else:
        model.clip.text_model.eval()
        model.clip.vision_model.train()
        model.decoder.train()


def build_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    groups: List[Dict[str, Any]] = []
    decoder_params = [p for p in model.decoder.parameters() if p.requires_grad]
    if decoder_params:
        groups.append(
            {
                "params": decoder_params,
                "lr": float(args.decoder_lr),
                "initial_lr": float(args.decoder_lr),
                "min_lr": float(args.min_lr),
                "name": "decoder",
            }
        )
    if args.train_mode == "vision_decoder":
        vision_params = [p for p in model.clip.vision_model.parameters() if p.requires_grad]
        groups.append(
            {
                "params": vision_params,
                "lr": float(args.vision_lr),
                "initial_lr": float(args.vision_lr),
                "min_lr": min(float(args.min_lr), float(args.vision_lr)),
                "name": "vision",
            }
        )
    return torch.optim.AdamW(groups, weight_decay=float(args.weight_decay))


def get_processor_stats(base_model: str) -> Tuple[List[float], List[float]]:
    processor = CLIPSegProcessor.from_pretrained(base_model)
    image_processor = processor.image_processor
    mean = [float(value) for value in image_processor.image_mean]
    std = [float(value) for value in image_processor.image_std]
    if len(mean) != 3 or len(std) != 3:
        raise ValueError(f"Expected three-channel processor statistics; got mean={mean}, std={std}")
    return mean, std


def normalize_clip_images(
    images: torch.Tensor,
    image_mean: torch.Tensor,
    image_std: torch.Tensor,
) -> torch.Tensor:
    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError(f"Expected Bx1xHxW grayscale images; got {tuple(images.shape)}")
    rgb = images.repeat(1, 3, 1, 1).clamp(0.0, 1.0)
    return (rgb - image_mean) / image_std


@torch.inference_mode()
def build_prompt_bank(
    model: nn.Module,
    tokenizer,
    prompts: Mapping[str, str],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    texts = [prompts[domain] for domain in DOMAIN_NAMES]
    tokenized = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    tokenized = {key: value.to(device) for key, value in tokenized.items()}
    embeddings = model.get_conditional_embeddings(
        batch_size=len(texts),
        input_ids=tokenized["input_ids"],
        attention_mask=tokenized.get("attention_mask"),
    )
    return {
        domain: embeddings[index : index + 1].detach()
        for index, domain in enumerate(DOMAIN_NAMES)
    }


def condition_for_domains(
    prompt_bank: Mapping[str, torch.Tensor],
    domains: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    return torch.cat([prompt_bank[str(domain)] for domain in domains], dim=0).to(device)


def binary_to_two_class_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError(f"CLIPSeg logits must be BxHxW; got {tuple(logits.shape)}")
    return torch.stack([torch.zeros_like(logits), logits], dim=1)


def flatten_validation_metrics(result: Mapping[str, Any]) -> Dict[str, float]:
    row: Dict[str, float] = {
        "val_macro_dice": float(result["dice"]),
        "val_macro_iou": float(result["iou"]),
        "val_macro_hd95": float(result["hd95"]),
        "val_weighted_dice": float(result["weighted_dice"]),
        "val_weighted_iou": float(result["weighted_iou"]),
        "val_weighted_hd95": float(result["weighted_hd95"]),
    }
    for domain in DOMAIN_NAMES:
        values = result["per_domain"].get(domain)
        for metric in ("dice", "iou", "hd95"):
            row[f"val_{domain}_{metric}"] = (
                float(values[metric]) if values is not None else float("nan")
            )
    return row


@torch.inference_mode()
def evaluate_clipseg(
    model: nn.Module,
    dataset,
    prompt_bank: Mapping[str, torch.Tensor],
    image_mean: torch.Tensor,
    image_std: torch.Tensor,
    device: torch.device,
    *,
    batch_size: int,
    amp_enabled: bool,
    threshold: float,
) -> Dict[str, Any]:
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    values: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"dice": [], "iou": [], "hd95": []}
    )

    for images, masks, domains in tqdm(loader, desc="Validating CLIPSeg", leave=False):
        images = images.to(device, non_blocking=True)
        conditional = condition_for_domains(prompt_bank, list(domains), device)
        pixel_values = normalize_clip_images(images, image_mean, image_std)
        with amp_context(device, amp_enabled):
            output = model(
                pixel_values=pixel_values,
                conditional_embeddings=conditional,
                interpolate_pos_encoding=True,
            )
            logits = output.logits
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(
                logits.unsqueeze(1).float(),
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        predictions = torch.sigmoid(logits.float()).ge(float(threshold)).cpu().numpy().astype(np.uint8)
        targets = masks.cpu().numpy().astype(np.uint8)
        for pred, target, domain in zip(predictions, targets, domains):
            key = str(domain)
            values[key]["dice"].append(binary_dice(pred, target))
            values[key]["iou"].append(binary_iou(pred, target))
            values[key]["hd95"].append(binary_hd95(pred, target))

    per_domain: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, int] = {}
    for domain in DOMAIN_NAMES:
        if not values[domain]["dice"]:
            continue
        per_domain[domain] = {
            metric: float(np.mean(metric_values))
            for metric, metric_values in values[domain].items()
        }
        counts[domain] = len(values[domain]["dice"])
    if not per_domain:
        raise RuntimeError("No validation predictions were produced.")
    total = sum(counts.values())
    return {
        "dice": float(np.mean([entry["dice"] for entry in per_domain.values()])),
        "iou": float(np.mean([entry["iou"] for entry in per_domain.values()])),
        "hd95": float(np.mean([entry["hd95"] for entry in per_domain.values()])),
        "weighted_dice": float(sum(per_domain[d]["dice"] * counts[d] for d in per_domain) / total),
        "weighted_iou": float(sum(per_domain[d]["iou"] * counts[d] for d in per_domain) / total),
        "weighted_hd95": float(sum(per_domain[d]["hd95"] * counts[d] for d in per_domain) / total),
        "per_domain": per_domain,
        "counts": counts,
        "threshold": float(threshold),
    }


def build_checkpoint(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    epoch: int,
    best_val_dice: float,
    patience_counter: int,
    args: argparse.Namespace,
    prompts: Mapping[str, str],
    image_mean: Sequence[float],
    image_std: Sequence[float],
    validation: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "method": "clipseg_official",
        "model_class": "CLIPSegForImageSegmentation",
        "base_model": str(args.base_model),
        "train_mode": str(args.train_mode),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": int(epoch),
        "best_val_dice": float(best_val_dice),
        "patience_counter": int(patience_counter),
        "validation": dict(validation),
        "training_args": vars(args),
        "prompts": dict(prompts),
        "image_mean": list(image_mean),
        "image_std": list(image_std),
        "image_size": tuple(DEFAULT_IMAGE_SIZE),
        "metric_size": tuple(DEFAULT_METRIC_SIZE),
        "threshold": float(args.threshold),
        "no_support_prompt": True,
        "no_test_mask_prompt": True,
        "no_postprocess": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune official CLIPSeg in the unified six-domain 512x512 protocol"
    )
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--save-path", default="comparisons/runs/clipseg_official_seed42/model.pth")
    parser.add_argument("--prompt-file", default="./clipseg_prompts.json")
    parser.add_argument("--base-model", default="CIDAS/clipseg-rd64-refined")
    parser.add_argument("--image-size", type=int, nargs="+", default=list(DEFAULT_IMAGE_SIZE))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--overwrite-history", action="store_true")

    cache = parser.add_mutually_exclusive_group()
    cache.add_argument("--cache-dataset", dest="cache_dataset", action="store_true")
    cache.add_argument("--no-cache-dataset", dest="cache_dataset", action="store_false")
    parser.set_defaults(cache_dataset=True)
    parser.add_argument("--preload-train-cache", action="store_true")
    parser.add_argument("--preload-val-cache", action="store_true")

    parser.add_argument("--train-mode", choices=("decoder", "vision_decoder"), default="decoder")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--val-batch-size", type=int, default=2)
    parser.add_argument("--episodes-per-epoch", type=int, default=1500)
    parser.add_argument("--grad-accum-steps", type=int, default=16)
    parser.add_argument("--decoder-lr", type=float, default=1e-4)
    parser.add_argument("--vision-lr", type=float, default=1e-5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--domain-sampling-alpha", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.5)

    amp = parser.add_mutually_exclusive_group()
    amp.add_argument("--amp", dest="amp", action="store_true")
    amp.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)

    augmentation = parser.add_mutually_exclusive_group()
    augmentation.add_argument("--augmentation", dest="augmentation", action="store_true")
    augmentation.add_argument("--no-augmentation", dest="augmentation", action="store_false")
    parser.set_defaults(augmentation=True)

    parser.add_argument("--ce-weight", type=float, default=DEFAULT_LOSS_CONFIG["ce"])
    parser.add_argument("--dice-weight", type=float, default=DEFAULT_LOSS_CONFIG["dice"])
    parser.add_argument("--focal-tversky-weight", type=float, default=DEFAULT_LOSS_CONFIG["focal_tversky"])
    parser.add_argument("--tversky-alpha", type=float, default=DEFAULT_LOSS_CONFIG["tversky_alpha"])
    parser.add_argument("--tversky-beta", type=float, default=DEFAULT_LOSS_CONFIG["tversky_beta"])
    parser.add_argument("--tversky-gamma", type=float, default=DEFAULT_LOSS_CONFIG["tversky_gamma"])
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> Tuple[int, int]:
    image_size = parse_image_size(args.image_size)
    if not Path(args.data_root).exists():
        raise FileNotFoundError(f"Data root does not exist: {args.data_root}")
    if not Path(args.prompt_file).exists():
        raise FileNotFoundError(f"Prompt file does not exist: {args.prompt_file}")
    if args.resume is not None and not Path(args.resume).exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {args.resume}")
    for name in ("epochs", "batch_size", "val_batch_size", "episodes_per_epoch", "grad_accum_steps"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not 0.0 < float(args.threshold) < 1.0:
        raise ValueError("--threshold must be in (0, 1)")
    return image_size


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    image_size = validate_args(args)
    set_seed(args.seed, args.deterministic)
    device = resolve_device(args.device)

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    history_path = Path(str(save_path) + ".history.csv")
    last_path = Path(str(save_path) + ".last.pth")
    config_path = Path(str(save_path) + ".config.json")
    if args.resume is None and history_path.exists():
        if args.overwrite_history:
            history_path.unlink()
        else:
            raise FileExistsError(
                f"History already exists: {history_path}. Use --resume, a new --save-path, "
                "or --overwrite-history."
            )

    prompts = load_prompts(args.prompt_file)
    data_cfg = DataConfig(data_root=args.data_root, image_size=image_size)
    train_dataset, val_dataset, _ = get_datasets(
        data_cfg.roots(),
        image_size=image_size,
        seed=args.seed,
        train_query_positive_ratio=1.0,
        prefer_empty_train_queries=False,
        cache_in_memory=args.cache_dataset,
        use_cpu_clahe=False,
    )
    # CLIP preprocessing expects raw [0,1] intensities, not per-image z-score tensors.
    train_dataset.image_transform = None
    val_dataset.image_transform = None
    if args.cache_dataset and args.preload_train_cache:
        train_dataset.warm_cache(show_progress=True)
    if args.cache_dataset and args.preload_val_cache:
        val_dataset.warm_cache(show_progress=True)

    image_mean_values, image_std_values = get_processor_stats(args.base_model)
    image_mean = torch.tensor(image_mean_values, device=device).view(1, 3, 1, 1)
    image_std = torch.tensor(image_std_values, device=device).view(1, 3, 1, 1)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = CLIPSegForImageSegmentation.from_pretrained(args.base_model).to(device)
    configure_trainable_parameters(model, args.train_mode)
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    optimizer = build_optimizer(model, args)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = make_grad_scaler(amp_enabled)
    prompt_bank = build_prompt_bank(model, tokenizer, prompts, device)

    loss_cfg = {
        "ce": float(args.ce_weight),
        "dice": float(args.dice_weight),
        "focal_tversky": float(args.focal_tversky_weight),
        "tversky_alpha": float(args.tversky_alpha),
        "tversky_beta": float(args.tversky_beta),
        "tversky_gamma": float(args.tversky_gamma),
    }
    domains = [d for d in DOMAIN_NAMES if d in train_dataset.organ_to_indices]
    if set(domains) != set(DOMAIN_NAMES):
        raise RuntimeError(f"Missing training domains: {sorted(set(DOMAIN_NAMES) - set(domains))}")
    domain_sizes = {d: len(train_dataset.get_organ_indices(d)) for d in domains}

    start_epoch = 0
    best_val_dice = -1.0
    patience_counter = 0
    if args.resume is not None:
        checkpoint = safe_torch_load(args.resume, map_location="cpu")
        if checkpoint.get("method") != "clipseg_official":
            raise ValueError(f"Not a CLIPSeg checkpoint: {args.resume}")
        if checkpoint.get("base_model") != args.base_model:
            raise ValueError(
                f"Resume base model mismatch: {checkpoint.get('base_model')} != {args.base_model}"
            )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_val_dice = float(checkpoint.get("best_val_dice", -1.0))
        patience_counter = int(checkpoint.get("patience_counter", 0))
        print(f"[resume] start_epoch={start_epoch + 1} best_val_dice={best_val_dice:.6f}")

    parameter_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    effective_batch = int(args.batch_size) * int(args.grad_accum_steps)
    run_config = {
        "method": "clipseg_official",
        "protocol": "unified_six_domain_512",
        "base_model": args.base_model,
        "train_mode": args.train_mode,
        "prompts": prompts,
        "image_mean": image_mean_values,
        "image_std": image_std_values,
        "parameters": parameter_count,
        "trainable_parameters": trainable_count,
        "domain_sizes": domain_sizes,
        "loss_config": loss_cfg,
        "training_args": vars(args),
        "effective_batch_size": effective_batch,
        "selection_metric": "validation macro Dice",
        "test_mask_used_for_prompting": False,
        "postprocess_during_validation": False,
    }
    write_json(run_config, config_path)

    print("=" * 96)
    print("Unified six-domain official CLIPSeg training")
    print(f"device={device} base_model={args.base_model} train_mode={args.train_mode}")
    print(f"train={len(train_dataset)} val={len(val_dataset)} input={image_size} metric={DEFAULT_METRIC_SIZE}")
    print(f"parameters={parameter_count / 1e6:.2f}M trainable={trainable_count / 1e6:.2f}M")
    print(
        f"batch={args.batch_size} grad_accum={args.grad_accum_steps} "
        f"effective_batch={effective_batch} episodes/epoch={args.episodes_per_epoch}"
    )
    print(f"decoder_lr={args.decoder_lr:.2e} vision_lr={args.vision_lr:.2e} amp={amp_enabled}")
    print(f"prompts={prompts}")
    print("No support mask, prototype, test-mask prompt, threshold tuning, or post-processing is used.")
    print("=" * 96)

    training_start = time.time()
    for epoch in range(start_epoch, int(args.epochs)):
        epoch_start = time.time()
        current_lrs = set_cosine_lr(optimizer, epoch, int(args.epochs), int(args.warmup_epochs))
        model.train()
        restore_frozen_eval_state(model, args.train_mode)
        schedule = make_domain_schedule(
            domains,
            int(args.episodes_per_epoch),
            domain_sizes=domain_sizes,
            alpha=float(args.domain_sampling_alpha),
        )
        optimizer.zero_grad(set_to_none=True)
        train_loss_sum = 0.0
        train_dice_sum = 0.0
        sample_count = 0

        progress = tqdm(
            enumerate(schedule),
            total=len(schedule),
            desc=f"Epoch {epoch + 1}/{args.epochs} [CLIPSeg]",
            dynamic_ncols=True,
        )
        for step, domain in progress:
            images, masks = sample_domain_batch(train_dataset, domain, args.batch_size)
            images = images.to(device, non_blocking=True)
            masks = masks.long().to(device, non_blocking=True)
            if args.augmentation:
                aug_images = images.mul(2.0).sub(1.0)
                aug_images, aug_masks = augment_episode_medical(
                    aug_images,
                    masks.unsqueeze(1).float(),
                    modality=domain_to_modality(domain),
                )
                images = aug_images.add(1.0).mul(0.5).clamp(0.0, 1.0)
                masks = aug_masks.squeeze(1).long()

            pixel_values = normalize_clip_images(images, image_mean, image_std)
            conditional = prompt_bank[domain].expand(images.shape[0], -1)
            with amp_context(device, amp_enabled):
                output = model(
                    pixel_values=pixel_values,
                    conditional_embeddings=conditional,
                    interpolate_pos_encoding=True,
                )
                logits = output.logits
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = F.interpolate(
                        logits.unsqueeze(1),
                        size=masks.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1)
                loss = segmentation_loss(binary_to_two_class_logits(logits), masks, loss_cfg)

            scaler.scale(loss / int(args.grad_accum_steps)).backward()
            should_step = (
                (step + 1) % int(args.grad_accum_steps) == 0
                or step + 1 == len(schedule)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=float(args.max_grad_norm),
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                predictions = torch.sigmoid(logits.float()).ge(float(args.threshold))
                targets = masks.eq(1)
                intersection = (predictions & targets).sum(dim=(1, 2)).float()
                denominator = predictions.sum(dim=(1, 2)).float() + targets.sum(dim=(1, 2)).float()
                batch_dice = torch.where(
                    denominator > 0,
                    2.0 * intersection / denominator.clamp_min(1e-8),
                    torch.ones_like(denominator),
                ).mean()

            batch_n = int(images.shape[0])
            train_loss_sum += float(loss.detach().item()) * batch_n
            train_dice_sum += float(batch_dice.detach().item()) * batch_n
            sample_count += batch_n
            lr_text = "/".join(f"{lr:.1e}" for lr in current_lrs)
            progress.set_postfix(
                loss=f"{train_loss_sum / max(sample_count, 1):.4f}",
                dice=f"{train_dice_sum / max(sample_count, 1):.4f}",
                domain=domain,
                lr=lr_text,
            )

        train_loss = train_loss_sum / max(sample_count, 1)
        train_dice = train_dice_sum / max(sample_count, 1)
        validation = evaluate_clipseg(
            model,
            val_dataset,
            prompt_bank,
            image_mean,
            image_std,
            device,
            batch_size=args.val_batch_size,
            amp_enabled=amp_enabled,
            threshold=args.threshold,
        )
        validation_flat = flatten_validation_metrics(validation)
        improved = float(validation["dice"]) > best_val_dice + 1e-8
        if improved:
            best_val_dice = float(validation["dice"])
            patience_counter = 0
        else:
            patience_counter += 1

        checkpoint = build_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            best_val_dice=best_val_dice,
            patience_counter=patience_counter,
            args=args,
            prompts=prompts,
            image_mean=image_mean_values,
            image_std=image_std_values,
            validation=validation,
        )
        atomic_torch_save(checkpoint, last_path)
        if improved:
            atomic_torch_save(checkpoint, save_path)

        epoch_seconds = time.time() - epoch_start
        append_history(
            history_path,
            {
                "epoch": epoch + 1,
                "decoder_lr": current_lrs[0],
                "vision_lr": current_lrs[1] if len(current_lrs) > 1 else 0.0,
                "train_loss": train_loss,
                "train_dice": train_dice,
                **validation_flat,
                "best_val_dice": best_val_dice,
                "improved": int(improved),
                "patience_counter": patience_counter,
                "epoch_seconds": epoch_seconds,
                "samples_seen": sample_count,
            },
        )
        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | train loss={train_loss:.4f} "
            f"dice={train_dice:.4f} | val macro Dice={validation['dice']:.4f} "
            f"IoU={validation['iou']:.4f} HD95={validation['hd95']:.2f} | "
            f"weighted Dice={validation['weighted_dice']:.4f} | best={best_val_dice:.4f} "
            f"patience={patience_counter}/{args.patience} | time={epoch_seconds / 60.0:.1f} min"
        )
        for domain in DOMAIN_NAMES:
            metric = validation["per_domain"].get(domain)
            if metric is not None:
                print(
                    f"  {domain:10s}: Dice={metric['dice']:.4f} IoU={metric['iou']:.4f} "
                    f"HD95={metric['hd95']:.2f} N={validation['counts'][domain]}"
                )
        if improved:
            print(f"  [best] saved: {save_path}")
        if patience_counter >= int(args.patience):
            print(f"[early stop] no validation macro Dice improvement for {args.patience} epochs")
            break

    elapsed_hours = (time.time() - training_start) / 3600.0
    print("=" * 96)
    print(f"Training complete. best_val_dice={best_val_dice:.6f}")
    print(f"Best checkpoint: {save_path}")
    print(f"Last checkpoint: {last_path}")
    print(f"Elapsed: {elapsed_hours:.2f} h")


if __name__ == "__main__":
    main()

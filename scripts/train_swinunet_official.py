from __future__ import annotations

"""Train the official Swin-Unet model in the TaskSegment comparison protocol.

The script is intentionally aligned with the TaskSegment comparison protocol:

* one unified model is trained on all six domains;
* images, masks, validation metrics, and HD95 use a fixed 512x512 grid;
* domain-balanced sampling is used by default;
* the same GPU augmentations and Dice-oriented loss are reused;
* the best checkpoint is selected by validation macro Dice;
* no text, support mask, prototype, post-processing, or test-time tuning is used.

Place this file at ``scripts/train_swinunet_official.py`` in the TSV2 project.
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
from torch.utils.data import DataLoader

# Make ``tasksegment`` importable when this script is launched from scripts/.
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


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

from tasksegment.models.swinunet_official import OfficialSwinUNet2D  # noqa: E402


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


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
    requested = str(value)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[warning] CUDA was requested but is unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def amp_context(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return nullcontext()
    # torch.autocast is available in modern PyTorch and avoids deprecated APIs.
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


def set_cosine_lr(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    num_epochs: int,
    base_lr: float,
    min_lr: float,
    warmup_epochs: int,
) -> float:
    if warmup_epochs > 0 and epoch < warmup_epochs:
        current = base_lr * float(epoch + 1) / float(warmup_epochs)
    else:
        remaining = max(1, num_epochs - warmup_epochs - 1)
        progress = min(max((epoch - warmup_epochs) / remaining, 0.0), 1.0)
        current = min_lr + 0.5 * (base_lr - min_lr) * (
            1.0 + math.cos(math.pi * progress)
        )
    for group in optimizer.param_groups:
        group["lr"] = float(current)
    return float(current)


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


def sample_domain_batch(dataset, domain: str, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    indices = dataset.get_organ_samples(
        domain,
        int(batch_size),
        positive_only=False,
    )
    if len(indices) != int(batch_size):
        # This branch is relevant only if a domain has fewer cases than the
        # requested batch size and get_organ_samples returns the complete pool.
        pool = dataset.get_organ_indices(domain, positive_only=False)
        extra = np.random.choice(
            pool,
            size=int(batch_size) - len(indices),
            replace=True,
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
        if values is None:
            row[f"val_{domain}_dice"] = float("nan")
            row[f"val_{domain}_iou"] = float("nan")
            row[f"val_{domain}_hd95"] = float("nan")
        else:
            row[f"val_{domain}_dice"] = float(values["dice"])
            row[f"val_{domain}_iou"] = float(values["iou"])
            row[f"val_{domain}_hd95"] = float(values["hd95"])
    return row


@torch.inference_mode()
def evaluate_swinunet(
    model: nn.Module,
    dataset,
    device: torch.device,
    *,
    batch_size: int,
    amp_enabled: bool,
    threshold: float = 0.5,
    show_progress: bool = True,
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

    iterator: Iterable = loader
    if show_progress:
        iterator = tqdm(loader, desc="Validating official Swin-Unet", leave=False)

    for images, masks, domains in iterator:
        images = images.to(device, non_blocking=True)
        with amp_context(device, amp_enabled):
            logits = model(images)
        probabilities = torch.softmax(logits.float(), dim=1)[:, 1]
        predictions = probabilities.ge(float(threshold)).cpu().numpy().astype(np.uint8)
        targets = masks.cpu().numpy().astype(np.uint8)

        for pred, target, domain in zip(predictions, targets, domains):
            domain = str(domain)
            values[domain]["dice"].append(binary_dice(pred, target))
            values[domain]["iou"].append(binary_iou(pred, target))
            values[domain]["hd95"].append(binary_hd95(pred, target))

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
        "weighted_dice": float(
            sum(per_domain[d]["dice"] * counts[d] for d in per_domain) / max(total, 1)
        ),
        "weighted_iou": float(
            sum(per_domain[d]["iou"] * counts[d] for d in per_domain) / max(total, 1)
        ),
        "weighted_hd95": float(
            sum(per_domain[d]["hd95"] * counts[d] for d in per_domain) / max(total, 1)
        ),
        "per_domain": per_domain,
        "counts": counts,
        "threshold": float(threshold),
        "metric_size": tuple(DEFAULT_METRIC_SIZE),
    }


def build_checkpoint(
    *,
    model: OfficialSwinUNet2D,
    optimizer: torch.optim.Optimizer,
    scaler,
    epoch: int,
    best_val_dice: float,
    patience_counter: int,
    args: argparse.Namespace,
    validation: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "method": "Swin-Unet-official",
        "model_class": "OfficialSwinUNet2D",
        "model_config": model.export_config(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": int(epoch),
        "best_val_dice": float(best_val_dice),
        "patience_counter": int(patience_counter),
        "validation": dict(validation),
        "training_args": vars(args),
        "image_size": tuple(DEFAULT_IMAGE_SIZE),
        "metric_size": tuple(DEFAULT_METRIC_SIZE),
        "threshold": float(args.threshold),
        "no_text": True,
        "no_support_prompt": True,
        "no_postprocess": True,
    }


# -----------------------------------------------------------------------------
# Command line and training
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train official Swin-Unet in the unified six-domain 512x512 protocol"
    )
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--save-path", default="comparisons/runs/swinunet_official_seed42/model.pth")
    parser.add_argument("--image-size", type=int, nargs="+", default=list(DEFAULT_IMAGE_SIZE))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--resume", default=None, help="Resume from a train_swinunet_official.py checkpoint")
    parser.add_argument(
        "--overwrite-history",
        action="store_true",
        help="Delete an existing history CSV before a fresh run",
    )

    cache = parser.add_mutually_exclusive_group()
    cache.add_argument("--cache-dataset", dest="cache_dataset", action="store_true")
    cache.add_argument("--no-cache-dataset", dest="cache_dataset", action="store_false")
    parser.set_defaults(cache_dataset=True)
    parser.add_argument("--preload-train-cache", action="store_true")
    val_cache = parser.add_mutually_exclusive_group()
    val_cache.add_argument("--preload-val-cache", dest="preload_val_cache", action="store_true")
    val_cache.add_argument("--no-preload-val-cache", dest="preload_val_cache", action="store_false")
    parser.set_defaults(preload_val_cache=True)
    parser.add_argument("--cpu-clahe", action="store_true")

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument("--episodes-per-epoch", type=int, default=1500)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument(
        "--domain-sampling-alpha",
        type=float,
        default=0.0,
        help="0=domain-balanced; 1=proportional to dataset size",
    )
    parser.add_argument("--threshold", type=float, default=0.5)

    amp = parser.add_mutually_exclusive_group()
    amp.add_argument("--amp", dest="amp", action="store_true")
    amp.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)

    augmentation = parser.add_mutually_exclusive_group()
    augmentation.add_argument("--augmentation", dest="augmentation", action="store_true")
    augmentation.add_argument("--no-augmentation", dest="augmentation", action="store_false")
    parser.set_defaults(augmentation=True)

    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=96)
    parser.add_argument(
        "--depths",
        type=int,
        nargs=4,
        default=[2, 2, 6, 2],
        metavar=("D1", "D2", "D3", "D4"),
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        nargs=4,
        default=[3, 6, 12, 24],
        metavar=("H1", "H2", "H3", "H4"),
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=8,
        help="Use 8 for the fixed 512x512 protocol. Official 224 config uses 7.",
    )
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--drop-path-rate", type=float, default=0.1)
    parser.add_argument("--ape", action="store_true")
    parser.add_argument("--no-patch-norm", dest="patch_norm", action="store_false")
    parser.set_defaults(patch_norm=True)
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument(
        "--pretrained-checkpoint",
        default=None,
        help=(
            "Optional official Swin-T checkpoint. For the controlled no-pretraining "
            "comparison, leave this unset."
        ),
    )

    # Use the same Dice-oriented main loss as TaskSegment for a controlled
    # architecture comparison. Official Swin-Unet has no auxiliary or dense-guidance losses.
    parser.add_argument("--ce-weight", type=float, default=DEFAULT_LOSS_CONFIG["ce"])
    parser.add_argument("--dice-weight", type=float, default=DEFAULT_LOSS_CONFIG["dice"])
    parser.add_argument(
        "--focal-tversky-weight",
        type=float,
        default=DEFAULT_LOSS_CONFIG["focal_tversky"],
    )
    parser.add_argument("--tversky-alpha", type=float, default=DEFAULT_LOSS_CONFIG["tversky_alpha"])
    parser.add_argument("--tversky-beta", type=float, default=DEFAULT_LOSS_CONFIG["tversky_beta"])
    parser.add_argument("--tversky-gamma", type=float, default=DEFAULT_LOSS_CONFIG["tversky_gamma"])
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> Tuple[int, int]:
    image_size = parse_image_size(args.image_size)
    if not Path(args.data_root).exists():
        raise FileNotFoundError(f"Data root does not exist: {args.data_root}")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0 or args.val_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")
    if args.episodes_per_epoch <= 0:
        raise ValueError("--episodes-per-epoch must be positive")
    if args.grad_accum_steps <= 0:
        raise ValueError("--grad-accum-steps must be positive")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be in (0, 1)")
    if args.patch_size <= 0 or image_size[0] % args.patch_size != 0:
        raise ValueError("--patch-size must be positive and divide 512")
    if args.embed_dim <= 0:
        raise ValueError("--embed-dim must be positive")
    if len(args.depths) != 4 or any(value <= 0 for value in args.depths):
        raise ValueError("--depths must contain four positive integers")
    if len(args.num_heads) != 4 or any(value <= 0 for value in args.num_heads):
        raise ValueError("--num-heads must contain four positive integers")
    for stage, heads in enumerate(args.num_heads):
        channels = args.embed_dim * (2 ** stage)
        if channels % heads != 0:
            raise ValueError(
                f"Stage {stage + 1}: channel dimension {channels} is not "
                f"divisible by {heads} attention heads"
            )
    patch_grid = image_size[0] // args.patch_size
    for stage in range(4):
        resolution = patch_grid // (2 ** stage)
        effective_window = min(args.window_size, resolution)
        if resolution % effective_window != 0:
            raise ValueError(
                f"Stage {stage + 1} resolution {resolution} is not divisible "
                f"by --window-size {args.window_size}"
            )
    if args.pretrained_checkpoint is not None and not Path(
        args.pretrained_checkpoint
    ).exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {args.pretrained_checkpoint}"
        )
    if args.resume is not None and not Path(args.resume).exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
    return image_size


def main() -> None:
    args = parse_args()
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
                f"History already exists: {history_path}. Use a new --save-path, "
                "--resume, or --overwrite-history."
            )

    data_cfg = DataConfig(data_root=args.data_root, image_size=image_size)
    train_dataset, val_dataset, _ = get_datasets(
        data_cfg.roots(),
        image_size=image_size,
        seed=args.seed,
        train_query_positive_ratio=1.0,
        prefer_empty_train_queries=False,
        cache_in_memory=args.cache_dataset,
        use_cpu_clahe=args.cpu_clahe,
    )

    if args.cache_dataset and args.preload_train_cache:
        train_dataset.warm_cache(show_progress=True)
    if args.cache_dataset and args.preload_val_cache:
        val_dataset.warm_cache(show_progress=True)

    model = OfficialSwinUNet2D(
        image_size=image_size,
        in_channels=1,
        num_classes=2,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depths=tuple(args.depths),
        num_heads=tuple(args.num_heads),
        window_size=args.window_size,
        mlp_ratio=args.mlp_ratio,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
        ape=args.ape,
        patch_norm=args.patch_norm,
        use_checkpoint=args.use_checkpoint,
        pretrained_checkpoint=args.pretrained_checkpoint,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = make_grad_scaler(amp_enabled)

    loss_cfg = {
        "ce": float(args.ce_weight),
        "dice": float(args.dice_weight),
        "focal_tversky": float(args.focal_tversky_weight),
        "tversky_alpha": float(args.tversky_alpha),
        "tversky_beta": float(args.tversky_beta),
        "tversky_gamma": float(args.tversky_gamma),
    }

    domains = [domain for domain in DOMAIN_NAMES if domain in train_dataset.organ_to_indices]
    domain_sizes = {
        domain: len(train_dataset.get_organ_indices(domain, positive_only=False))
        for domain in domains
    }
    if set(domains) != set(DOMAIN_NAMES):
        missing = sorted(set(DOMAIN_NAMES) - set(domains))
        raise RuntimeError(f"Missing training domains: {missing}")

    start_epoch = 0
    best_val_dice = -1.0
    patience_counter = 0

    if args.resume is not None:
        checkpoint = safe_torch_load(args.resume, map_location="cpu")
        if checkpoint.get("method") != "Swin-Unet-official":
            raise ValueError(f"Not a train_swinunet_official.py checkpoint: {args.resume}")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_val_dice = float(checkpoint.get("best_val_dice", -1.0))
        patience_counter = int(checkpoint.get("patience_counter", 0))
        print(
            f"[resume] checkpoint={args.resume} start_epoch={start_epoch + 1} "
            f"best_val_dice={best_val_dice:.6f}"
        )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    effective_batch_size = int(args.batch_size) * int(args.grad_accum_steps)

    run_config = {
        "method": "Swin-Unet-official",
        "protocol": "unified_six_domain_512",
        "data_root": str(Path(args.data_root).resolve()),
        "image_size": image_size,
        "metric_size": tuple(DEFAULT_METRIC_SIZE),
        "domains": domains,
        "domain_sizes": domain_sizes,
        "model_config": model.export_config(),
        "parameters": parameter_count,
        "trainable_parameters": trainable_count,
        "loss_config": loss_cfg,
        "training_args": vars(args),
        "effective_batch_size": effective_batch_size,
        "selection_metric": "validation macro Dice",
        "test_mask_used_for_prompting": False,
        "postprocess_during_validation": False,
    }
    write_json(run_config, config_path)

    print("=" * 96)
    print("Unified six-domain official Swin-Unet baseline training")
    print(
        f"device={device} train={len(train_dataset)} val={len(val_dataset)} "
        f"input={image_size} metric={tuple(DEFAULT_METRIC_SIZE)}"
    )
    print(
        f"model=OfficialSwinUNet2D embed_dim={args.embed_dim} "
        f"depths={tuple(args.depths)} heads={tuple(args.num_heads)} "
        f"window={args.window_size} pretrained={args.pretrained_checkpoint is not None} "
        f"parameters={parameter_count / 1e6:.2f}M"
    )
    print(
        f"batch={args.batch_size} grad_accum={args.grad_accum_steps} "
        f"effective_batch={effective_batch_size} steps/epoch={args.episodes_per_epoch}"
    )
    print(
        f"cache={args.cache_dataset} preload_train={args.preload_train_cache} "
        f"preload_val={args.preload_val_cache} amp={amp_enabled} "
        f"augmentation={args.augmentation} deterministic={args.deterministic}"
    )
    print(f"domain_sampling_alpha={args.domain_sampling_alpha} loss={loss_cfg}")
    print(f"best checkpoint={save_path}")
    print("No text, support mask, prototype, test prompt, or post-processing is used.")
    print("=" * 96)

    training_start = time.time()

    for epoch in range(start_epoch, int(args.epochs)):
        epoch_start = time.time()
        current_lr = set_cosine_lr(
            optimizer,
            epoch,
            int(args.epochs),
            float(args.lr),
            float(args.min_lr),
            int(args.warmup_epochs),
        )

        model.train()
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
            desc=f"Epoch {epoch + 1}/{args.epochs} [official Swin-Unet]",
            dynamic_ncols=True,
        )

        for step, domain in progress:
            images, masks = sample_domain_batch(
                train_dataset,
                domain,
                args.batch_size,
            )
            images = images.to(device, non_blocking=True)
            masks = masks.long().to(device, non_blocking=True)

            if args.augmentation:
                images, augmented_masks = augment_episode_medical(
                    images,
                    masks.unsqueeze(1).float(),
                    modality=domain_to_modality(domain),
                )
                masks = augmented_masks.squeeze(1).long()

            with amp_context(device, amp_enabled):
                logits = model(images)
                loss = segmentation_loss(logits, masks, loss_cfg)

            scaler.scale(loss / int(args.grad_accum_steps)).backward()

            should_step = (
                (step + 1) % int(args.grad_accum_steps) == 0
                or step + 1 == len(schedule)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(args.max_grad_norm),
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                probabilities = torch.softmax(logits.float(), dim=1)[:, 1]
                predictions = probabilities.ge(float(args.threshold))
                targets = masks.eq(1)
                intersection = (predictions & targets).sum(dim=(1, 2)).float()
                denominator = (
                    predictions.sum(dim=(1, 2)).float()
                    + targets.sum(dim=(1, 2)).float()
                )
                batch_dice = torch.where(
                    denominator > 0,
                    2.0 * intersection / denominator.clamp_min(1e-8),
                    torch.ones_like(denominator),
                ).mean()

            batch_n = int(images.shape[0])
            train_loss_sum += float(loss.detach().item()) * batch_n
            train_dice_sum += float(batch_dice.detach().item()) * batch_n
            sample_count += batch_n

            progress.set_postfix(
                loss=f"{train_loss_sum / max(sample_count, 1):.4f}",
                dice=f"{train_dice_sum / max(sample_count, 1):.4f}",
                domain=domain,
                lr=f"{current_lr:.2e}",
            )

        train_loss = train_loss_sum / max(sample_count, 1)
        train_dice = train_dice_sum / max(sample_count, 1)

        validation = evaluate_swinunet(
            model,
            val_dataset,
            device,
            batch_size=args.val_batch_size,
            amp_enabled=amp_enabled,
            threshold=args.threshold,
            show_progress=True,
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
            validation=validation,
        )
        atomic_torch_save(checkpoint, last_path)
        if improved:
            atomic_torch_save(checkpoint, save_path)

        epoch_seconds = time.time() - epoch_start
        history_row: Dict[str, Any] = {
            "epoch": epoch + 1,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_dice": train_dice,
            **validation_flat,
            "best_val_dice": best_val_dice,
            "improved": int(improved),
            "patience_counter": patience_counter,
            "epoch_seconds": epoch_seconds,
            "samples_seen": sample_count,
        }
        append_history(history_path, history_row)

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"train loss={train_loss:.4f} dice={train_dice:.4f} | "
            f"val macro Dice={validation['dice']:.4f} "
            f"IoU={validation['iou']:.4f} HD95={validation['hd95']:.2f} | "
            f"weighted Dice={validation['weighted_dice']:.4f} | "
            f"best={best_val_dice:.4f} patience={patience_counter}/{args.patience} | "
            f"time={epoch_seconds / 60.0:.1f} min"
        )
        for domain in DOMAIN_NAMES:
            if domain not in validation["per_domain"]:
                continue
            metric = validation["per_domain"][domain]
            print(
                f"  {domain:10s}: Dice={metric['dice']:.4f} "
                f"IoU={metric['iou']:.4f} HD95={metric['hd95']:.2f} "
                f"N={validation['counts'][domain]}"
            )
        if improved:
            print(f"  [best] saved: {save_path}")

        if patience_counter >= int(args.patience):
            print(
                f"[early stop] validation macro Dice did not improve for "
                f"{args.patience} epochs."
            )
            break

    elapsed_hours = (time.time() - training_start) / 3600.0
    print("=" * 96)
    print(f"Training complete. best_val_dice={best_val_dice:.6f}")
    print(f"Best checkpoint: {save_path}")
    print(f"Last checkpoint: {last_path}")
    print(f"History: {history_path}")
    print(f"Config: {config_path}")
    print(f"Elapsed: {elapsed_hours:.2f} hours")
    print("=" * 96)


if __name__ == "__main__":
    main()

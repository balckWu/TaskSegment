#!/usr/bin/env python3
from __future__ import annotations

"""Train the official LViT architecture under the unified six-domain protocol.

Primary paper configuration:
- official LViT architecture at its native 224x224 model input;
- the exact same six domain-level prompts used by the CLIPSeg comparison;
- frozen local BERT-base text embeddings, cached once per domain;
- logits resized to the common 512x512 metric grid before thresholding;
- domain-balanced sampling, controlled loss, fixed 0.5 threshold and no
  post-processing.

The paper name is ``LViT (native 224; domain-level prompts)``. This is an
adapted multi-domain comparison, not a claim of exact original-paper
reproduction.
"""

import argparse
import csv
import hashlib
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

from tasksegment.configs import (
    DOMAIN_NAMES,
    DataConfig,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_LOSS_CONFIG,
    DEFAULT_METRIC_SIZE,
)
from tasksegment.data import get_datasets, set_seed
from tasksegment.models.lvit_official import OfficialLViTAdapter
from tasksegment.training.augmentations import augment_episode_medical
from tasksegment.training.losses import segmentation_loss
from tasksegment.training.memory import make_domain_schedule
from tasksegment.training.metrics import binary_dice, binary_hd95, binary_iou

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError as exc:
    raise ImportError("Install requirements-lvit.txt before training LViT.") from exc


def resolve_device(value: str) -> torch.device:
    if value.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(value)


def amp_context(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def make_scaler(
    enabled: bool,
    init_scale: float = 4096.0,
    growth_interval: int = 4000,
):
    kwargs = {
        "enabled": enabled,
        "init_scale": float(init_scale),
        "growth_interval": int(growth_interval),
        "growth_factor": 2.0,
        "backoff_factor": 0.5,
    }
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", **kwargs)
        except TypeError:
            return torch.amp.GradScaler(**kwargs)
    return torch.cuda.amp.GradScaler(**kwargs)


def atomic_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temp)
    os.replace(temp, path)


def safe_load(path: str | Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def append_csv(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(dict(row))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_prompts(path: Path) -> Dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    prompts = {str(k): str(v).strip() for k, v in payload.items()}
    missing = [d for d in DOMAIN_NAMES if not prompts.get(d)]
    if missing:
        raise ValueError(f"Missing prompts for domains: {missing}")
    return {d: prompts[d] for d in DOMAIN_NAMES}


@torch.inference_mode()
def build_text_bank(
    encoder_path: str,
    prompts: Mapping[str, str],
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(encoder_path, local_files_only=True)
    encoder = AutoModel.from_pretrained(encoder_path, local_files_only=True).to(device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False

    texts = [prompts[d] for d in DOMAIN_NAMES]
    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=10,
        return_tensors="pt",
    )
    tokenized_device = {k: v.to(device) for k, v in tokenized.items()}
    output = encoder(**tokenized_device).last_hidden_state
    if output.shape != (len(DOMAIN_NAMES), 10, 768):
        raise ValueError(f"Expected six Bx10x768 BERT embeddings, got {tuple(output.shape)}")
    attention = tokenized_device.get("attention_mask")
    if attention is not None:
        output = output * attention.unsqueeze(-1)
    bank = {d: output[i : i + 1].detach().float().cpu() for i, d in enumerate(DOMAIN_NAMES)}
    metadata = {
        "text_encoder": str(encoder_path),
        "text_encoder_class": encoder.__class__.__name__,
        "text_encoder_hidden_size": int(encoder.config.hidden_size),
        "text_encoder_parameters": int(sum(p.numel() for p in encoder.parameters())),
        "text_encoder_frozen": True,
        "token_length": 10,
        "padding_embeddings_zeroed": True,
    }
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return bank, metadata


def text_for_domain(bank: Mapping[str, torch.Tensor], domain: str, batch: int, device: torch.device) -> torch.Tensor:
    return bank[str(domain)].to(device, non_blocking=True).expand(int(batch), -1, -1)


def text_for_domains(bank: Mapping[str, torch.Tensor], domains: Sequence[str], device: torch.device) -> torch.Tensor:
    return torch.cat([bank[str(d)] for d in domains], dim=0).to(device, non_blocking=True)


def sample_domain_batch(dataset, domain: str, batch_size: int):
    pool = dataset.get_organ_indices(domain, positive_only=False)
    indices = np.random.choice(pool, size=int(batch_size), replace=len(pool) < int(batch_size)).tolist()
    images = torch.stack([dataset[i][0] for i in indices])
    masks = torch.stack([dataset[i][1] for i in indices])
    return images, masks


def domain_to_modality(domain: str) -> str:
    if domain in {"thyroid", "TN3K"}:
        return "thyroid"
    if domain in {"BUSI_WHU", "BUS-BRA"}:
        return "BUSI"
    return "generic"


def official_weighted_dice_bce(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_f = target.float()
    bce = F.binary_cross_entropy(probability, target_f)
    dims = (1, 2)
    intersection = (probability * target_f).sum(dims)
    denominator = probability.sum(dims) + target_f.sum(dims)
    dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return 0.5 * bce + 0.5 * dice_loss


def cosine_lr(optimizer, epoch: int, epochs: int, warmup: int, base_lr: float, min_lr: float) -> float:
    if warmup > 0 and epoch < warmup:
        factor = (epoch + 1) / warmup
        value = min_lr + (base_lr - min_lr) * factor
    else:
        denominator = max(1, epochs - warmup - 1)
        progress = min(max((epoch - warmup) / denominator, 0.0), 1.0)
        value = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = value
    return value


def summarize_validation(values):
    per_domain, counts = {}, {}
    for domain in DOMAIN_NAMES:
        if not values[domain]["dice"]:
            continue
        per_domain[domain] = {k: float(np.mean(v)) for k, v in values[domain].items()}
        counts[domain] = len(values[domain]["dice"])
    total = sum(counts.values())
    return {
        "dice": float(np.mean([v["dice"] for v in per_domain.values()])),
        "iou": float(np.mean([v["iou"] for v in per_domain.values()])),
        "hd95": float(np.mean([v["hd95"] for v in per_domain.values()])),
        "weighted_dice": float(sum(per_domain[d]["dice"] * counts[d] for d in counts) / total),
        "weighted_iou": float(sum(per_domain[d]["iou"] * counts[d] for d in counts) / total),
        "weighted_hd95": float(sum(per_domain[d]["hd95"] * counts[d] for d in counts) / total),
        "per_domain": per_domain,
        "counts": counts,
    }


@torch.inference_mode()
def evaluate(model, dataset, text_bank, device, batch_size: int, amp: bool, threshold: float):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    values = defaultdict(lambda: {"dice": [], "iou": [], "hd95": []})
    for images, masks, domains in tqdm(loader, desc="Validating LViT", leave=False):
        images = images.to(device, non_blocking=True)
        text = text_for_domains(text_bank, domains, device)
        with amp_context(device, amp):
            logits = model(images, text)
        predictions = torch.softmax(logits.float(), dim=1)[:, 1].ge(threshold).cpu().numpy().astype(np.uint8)
        targets = masks.numpy().astype(np.uint8)
        for pred, target, domain in zip(predictions, targets, domains):
            domain = str(domain)
            values[domain]["dice"].append(binary_dice(pred, target))
            values[domain]["iou"].append(binary_iou(pred, target))
            values[domain]["hd95"].append(binary_hd95(pred, target))
    return summarize_validation(values)


def flatten_validation(result):
    row = {
        "val_macro_dice": result["dice"],
        "val_macro_iou": result["iou"],
        "val_macro_hd95": result["hd95"],
        "val_weighted_dice": result["weighted_dice"],
        "val_weighted_iou": result["weighted_iou"],
        "val_weighted_hd95": result["weighted_hd95"],
    }
    for domain in DOMAIN_NAMES:
        metric = result["per_domain"].get(domain, {})
        for name in ("dice", "iou", "hd95"):
            row[f"val_{domain}_{name}"] = metric.get(name, float("nan"))
    return row


def parse_args():
    p = argparse.ArgumentParser(description="Train LViT under the unified six-domain paper protocol")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--official-repo", default="./third_party/LViT")
    p.add_argument("--text-encoder", default="./pretrained/bert-base-uncased")
    p.add_argument("--prompt-file", default="./lvit_prompts.json")
    p.add_argument("--save-path", default="paper_artifacts/manual/lvit_native224_seed42/checkpoint.pth")
    p.add_argument("--model-resolution", type=int, choices=(224, 512), default=512)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--resume", default=None)
    p.add_argument("--overwrite-history", action="store_true")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--val-batch-size", type=int, default=1)
    p.add_argument("--episodes-per-epoch", type=int, default=1500)
    p.add_argument("--grad-accum-steps", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--max-grad-norm", type=float, default=5.0)
    p.add_argument("--domain-sampling-alpha", type=float, default=0.0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--loss-profile", choices=("controlled", "official"), default="controlled")
    p.add_argument("--cache-dataset", action="store_true", default=True)
    p.add_argument("--no-cache-dataset", action="store_false", dest="cache_dataset")
    p.add_argument("--preload-val-cache", action="store_true")
    p.add_argument("--preload-train-cache", action="store_true")
    p.add_argument("--augmentation", action="store_true", default=True)
    p.add_argument("--no-augmentation", action="store_false", dest="augmentation")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", action="store_false", dest="amp")
    p.add_argument("--amp-init-scale", type=float, default=4096.0)
    p.add_argument("--amp-growth-interval", type=int, default=4000)
    p.add_argument("--max-consecutive-amp-skips", type=int, default=8)
    p.add_argument("--ce-weight", type=float, default=DEFAULT_LOSS_CONFIG["ce"])
    p.add_argument("--dice-weight", type=float, default=DEFAULT_LOSS_CONFIG["dice"])
    p.add_argument("--focal-tversky-weight", type=float, default=DEFAULT_LOSS_CONFIG["focal_tversky"])
    p.add_argument("--tversky-alpha", type=float, default=DEFAULT_LOSS_CONFIG["tversky_alpha"])
    p.add_argument("--tversky-beta", type=float, default=DEFAULT_LOSS_CONFIG["tversky_beta"])
    p.add_argument("--tversky-gamma", type=float, default=DEFAULT_LOSS_CONFIG["tversky_gamma"])
    return p.parse_args()


def main():
    args = parse_args()
    os.chdir(ROOT)
    for path in (args.data_root, args.official_repo, args.text_encoder, args.prompt_file):
        if not Path(path).exists():
            raise FileNotFoundError(path)

    set_seed(args.seed, args.deterministic)
    device = resolve_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")

    save_path = Path(args.save_path)
    history_path = Path(str(save_path) + ".history.csv")
    last_path = Path(str(save_path) + ".last.pth")
    config_path = Path(str(save_path) + ".config.json")
    if args.resume is None and history_path.exists():
        if args.overwrite_history:
            history_path.unlink()
        else:
            raise FileExistsError(f"History exists: {history_path}")

    prompt_path = Path(args.prompt_file)
    prompts = load_prompts(prompt_path)
    data_cfg = DataConfig(data_root=args.data_root, image_size=DEFAULT_IMAGE_SIZE)
    train_dataset, val_dataset, _ = get_datasets(
        data_cfg.roots(),
        image_size=DEFAULT_IMAGE_SIZE,
        seed=args.seed,
        train_query_positive_ratio=1.0,
        prefer_empty_train_queries=False,
        cache_in_memory=args.cache_dataset,
        use_cpu_clahe=False,
    )
    # Official LViT data loading uses raw ToTensor intensities.
    train_dataset.image_transform = None
    val_dataset.image_transform = None
    if args.preload_train_cache:
        train_dataset.warm_cache()
    if args.preload_val_cache:
        val_dataset.warm_cache()

    text_bank, text_metadata = build_text_bank(args.text_encoder, prompts, device)
    model = OfficialLViTAdapter(
        official_repository_path=args.official_repo,
        model_resolution=args.model_resolution,
        output_resolution=DEFAULT_METRIC_SIZE,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = make_scaler(
        amp_enabled,
        init_scale=args.amp_init_scale,
        growth_interval=args.amp_growth_interval,
    )
    loss_cfg = {
        "ce": args.ce_weight,
        "dice": args.dice_weight,
        "focal_tversky": args.focal_tversky_weight,
        "tversky_alpha": args.tversky_alpha,
        "tversky_beta": args.tversky_beta,
        "tversky_gamma": args.tversky_gamma,
    }
    domains = list(DOMAIN_NAMES)
    domain_sizes = {d: len(train_dataset.get_organ_indices(d)) for d in domains}

    start_epoch, best, patience_counter = 0, -1.0, 0
    amp_skipped_updates = 0
    consecutive_amp_skips = 0
    if args.resume:
        checkpoint = safe_load(args.resume)
        if checkpoint.get("method") != "lvit_protocol":
            raise ValueError("Not an LViT protocol checkpoint")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint.get("scaler_state_dict", {}))
        text_bank = checkpoint["text_embeddings"]
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint["best_val_dice"])
        patience_counter = int(checkpoint.get("patience_counter", 0))
        amp_skipped_updates = int(checkpoint.get("amp_skipped_updates", 0))
        consecutive_amp_skips = int(checkpoint.get("consecutive_amp_skips", 0))

    parameter_count = sum(p.numel() for p in model.parameters())
    official_commit_file = Path(args.official_repo) / "OFFICIAL_COMMIT_USED.txt"
    official_commit = official_commit_file.read_text(encoding="utf-8").strip() if official_commit_file.exists() else "unknown"
    method_name = (
        "lvit_adapted512"
        if args.model_resolution == 512
        else "lvit_native224_controlled"
    )
    run_config = {
        "method": method_name,
        "architecture": "official HUANGLIZI/LViT with minimal dynamic-resolution patch",
        "numerical_output_policy": "terminal sigmoid replaced by identity; losses use native FP32 logits",
        "nonfinite_policy": (
            "finite logits/loss required; isolated FP16 gradient overflow causes "
            "GradScaler optimizer-step skip and scale backoff; repeated overflows fail"
        ),
        "amp_init_scale": args.amp_init_scale,
        "amp_growth_interval": args.amp_growth_interval,
        "max_consecutive_amp_skips": args.max_consecutive_amp_skips,
        "official_repository": "HUANGLIZI/LViT",
        "official_commit": official_commit,
        "model_resolution": args.model_resolution,
        "target_and_metric_resolution": DEFAULT_METRIC_SIZE,
        "resolution_patch_marker": str((Path(args.official_repo) / "TASKSEGMENT_DYNAMIC_RESOLUTION_PATCH.txt").resolve()),
        "prompts": prompts,
        "prompt_file_sha256": file_sha256(prompt_path),
        "prompt_protocol": "identical strings to CLIPSeg comparison",
        **text_metadata,
        "parameters": parameter_count,
        "trainable_parameters": parameter_count,
        "external_visual_pretraining": False,
        "domain_sizes": domain_sizes,
        "effective_batch_size": args.batch_size * args.grad_accum_steps,
        "loss_profile": args.loss_profile,
        "controlled_loss_config": loss_cfg if args.loss_profile == "controlled" else None,
        "selection_metric": "validation domain-macro Dice",
        "threshold": args.threshold,
        "test_mask_used_for_text": False,
        "text_derived_from_masks": False,
        "postprocess": False,
        "training_args": vars(args),
    }
    write_json(run_config, config_path)

    print("=" * 100)
    print("LViT unified paper comparison")
    print(f"method={method_name} physical GPU is controlled by CUDA_VISIBLE_DEVICES; logical device={device}")
    print(f"model_resolution={args.model_resolution} target/metric={DEFAULT_METRIC_SIZE}")
    print(f"parameters={parameter_count / 1e6:.2f}M external_visual_pretraining=False")
    print(f"text_encoder={args.text_encoder} frozen=True token_shape=10x768")
    print(f"prompts identical to CLIPSeg: {prompts}")
    print(f"loss_profile={args.loss_profile} lr={args.lr:.2e} effective_batch={args.batch_size * args.grad_accum_steps}")
    print("No per-case text, mask-derived text, test tuning, support prompt, prototype or post-processing.")
    print("=" * 100)

    for epoch in range(start_epoch, args.epochs):
        started = time.time()
        lr = cosine_lr(optimizer, epoch, args.epochs, args.warmup_epochs, args.lr, args.min_lr)
        model.train()
        schedule = make_domain_schedule(
            domains,
            args.episodes_per_epoch,
            domain_sizes=domain_sizes,
            alpha=args.domain_sampling_alpha,
        )
        optimizer.zero_grad(set_to_none=True)
        loss_sum = dice_sum = 0.0
        seen = 0

        progress = tqdm(enumerate(schedule), total=len(schedule), desc=f"Epoch {epoch+1}/{args.epochs} [LViT]")
        for step, domain in progress:
            images, masks = sample_domain_batch(train_dataset, domain, args.batch_size)
            images = images.to(device, non_blocking=True)
            masks = masks.long().to(device, non_blocking=True)
            if args.augmentation:
                temp = images.mul(2).sub(1)
                temp, aug_masks = augment_episode_medical(
                    temp,
                    masks[:, None].float(),
                    modality=domain_to_modality(domain),
                )
                images = temp.add(1).mul(0.5).clamp(0, 1)
                masks = aug_masks[:, 0].long()
            text = text_for_domain(text_bank, domain, images.shape[0], device)

            with amp_context(device, amp_enabled):
                logits = model(images, text)

            # Always compute the loss in FP32. The official architecture's old
            # Sigmoid->logit inversion is intentionally removed by the adapter.
            logits = logits.float()
            if args.loss_profile == "official":
                probability = torch.softmax(logits, dim=1)[:, 1]
                loss = official_weighted_dice_bce(probability, masks)
            else:
                loss = segmentation_loss(logits, masks, loss_cfg)

            if not torch.isfinite(logits).all():
                raise FloatingPointError(
                    f"Non-finite LViT logits at epoch={epoch+1}, step={step+1}, domain={domain}; "
                    f"min={float(torch.nan_to_num(logits).min())}, "
                    f"max={float(torch.nan_to_num(logits).max())}"
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite LViT loss at epoch={epoch+1}, step={step+1}, domain={domain}. "
                    "The run is stopped before corrupting the optimizer state."
                )

            scaler.scale(loss / args.grad_accum_steps).backward()
            if (step + 1) % args.grad_accum_steps == 0 or step + 1 == len(schedule):
                scaler.unscale_(optimizer)

                nonfinite_gradient = False
                for parameter in model.parameters():
                    gradient = parameter.grad
                    if gradient is not None and not torch.isfinite(gradient).all():
                        nonfinite_gradient = True
                        break

                scale_before = float(scaler.get_scale())
                if nonfinite_gradient:
                    if not amp_enabled:
                        optimizer.zero_grad(set_to_none=True)
                        raise FloatingPointError(
                            f"Non-finite FP32 LViT gradient at epoch={epoch+1}, "
                            f"step={step+1}, domain={domain}."
                        )

                    # GradScaler.unscale_ has already recorded found_inf.
                    # scaler.step() therefore skips optimizer.step(), and
                    # scaler.update() backs off the scale as designed.
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                    amp_skipped_updates += 1
                    consecutive_amp_skips += 1
                    scale_after = float(scaler.get_scale())
                    progress.write(
                        "[amp overflow] "
                        f"epoch={epoch+1} step={step+1} domain={domain} "
                        f"scale={scale_before:g}->{scale_after:g} "
                        f"total_skips={amp_skipped_updates} "
                        f"consecutive={consecutive_amp_skips}"
                    )
                    if consecutive_amp_skips > args.max_consecutive_amp_skips:
                        raise FloatingPointError(
                            "Too many consecutive AMP-overflow updates: "
                            f"{consecutive_amp_skips} > "
                            f"{args.max_consecutive_amp_skips}."
                        )
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        args.max_grad_norm,
                        error_if_nonfinite=True,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    consecutive_amp_skips = 0

            with torch.no_grad():
                pred = torch.softmax(logits.float(), dim=1)[:, 1].ge(args.threshold)
                target = masks.eq(1)
                inter = (pred & target).sum((1, 2)).float()
                denom = pred.sum((1, 2)).float() + target.sum((1, 2)).float()
                batch_dice = torch.where(denom > 0, 2 * inter / denom.clamp_min(1e-8), torch.ones_like(denom)).mean()
            n = images.shape[0]
            loss_sum += float(loss.detach()) * n
            dice_sum += float(batch_dice) * n
            seen += n
            progress.set_postfix(loss=f"{loss_sum/seen:.4f}", dice=f"{dice_sum/seen:.4f}", domain=domain, lr=f"{lr:.1e}")

        validation = evaluate(model, val_dataset, text_bank, device, args.val_batch_size, amp_enabled, args.threshold)
        improved = validation["dice"] > best + 1e-8
        if improved:
            best = validation["dice"]
            patience_counter = 0
        else:
            patience_counter += 1

        checkpoint = {
            "schema_version": 1,
            "method": "lvit_protocol",
            "method_name": method_name,
            "model_class": "OfficialLViTAdapter",
            "model_config": model.export_config(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "best_val_dice": best,
            "patience_counter": patience_counter,
            "amp_skipped_updates": amp_skipped_updates,
            "consecutive_amp_skips": consecutive_amp_skips,
            "amp_scale": float(scaler.get_scale()),
            "validation": validation,
            "text_embeddings": {k: v.cpu() for k, v in text_bank.items()},
            "prompts": prompts,
            "text_metadata": text_metadata,
            "threshold": args.threshold,
            "loss_profile": args.loss_profile,
            "training_args": vars(args),
            "metric_size": DEFAULT_METRIC_SIZE,
            "postprocess": False,
        }
        atomic_save(checkpoint, last_path)
        if improved:
            atomic_save(checkpoint, save_path)

        row = {
            "epoch": epoch + 1,
            "lr": lr,
            "train_loss": loss_sum / max(seen, 1),
            "train_dice": dice_sum / max(seen, 1),
            **flatten_validation(validation),
            "best_val_dice": best,
            "improved": int(improved),
            "patience_counter": patience_counter,
            "amp_skipped_updates": amp_skipped_updates,
            "consecutive_amp_skips": consecutive_amp_skips,
            "amp_scale": float(scaler.get_scale()),
            "epoch_seconds": time.time() - started,
            "samples_seen": seen,
        }
        append_csv(history_path, row)
        print(
            f"Epoch {epoch+1:03d}: train loss={row['train_loss']:.4f} dice={row['train_dice']:.4f} | "
            f"val macro Dice={validation['dice']:.4f} IoU={validation['iou']:.4f} "
            f"HD95={validation['hd95']:.2f} weighted Dice={validation['weighted_dice']:.4f} | "
            f"best={best:.4f} patience={patience_counter}/{args.patience} "
            f"amp_skips={amp_skipped_updates} scale={float(scaler.get_scale()):g}"
        )
        for domain in DOMAIN_NAMES:
            m = validation["per_domain"][domain]
            print(f"  {domain:10s}: Dice={m['dice']:.4f} IoU={m['iou']:.4f} HD95={m['hd95']:.2f}")
        if patience_counter >= args.patience:
            print("[early stop]")
            break

    print(f"Best checkpoint: {save_path}")
    print(f"History: {history_path}")
    print(f"Run config: {config_path}")


if __name__ == "__main__":
    main()

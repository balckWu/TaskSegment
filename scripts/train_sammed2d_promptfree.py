#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paperlab.progress import paper_tqdm as tqdm
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from sammed2d_protocol_utils import (
    amp_context,
    append_csv,
    atomic_torch_save,
    cosine_lr,
    make_scaler,
    resolve_device,
    safe_torch_load,
    sha256_file,
    write_json,
)
from tasksegment.configs import (
    DOMAIN_NAMES,
    DataConfig,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_LOSS_CONFIG,
    DEFAULT_METRIC_SIZE,
)
from tasksegment.data import get_datasets, set_seed
from tasksegment.models.sammed2d_official import OfficialSAMMed2DAdapter
from tasksegment.training.augmentations import augment_episode_medical
from tasksegment.training.losses import segmentation_loss
from tasksegment.training.memory import make_domain_schedule
from tasksegment.training.metrics import binary_dice, binary_hd95, binary_iou


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prompt-free SAM-Med2D downstream adaptation")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--official-repo", default="./third_party/SAM-Med2D")
    parser.add_argument("--base-checkpoint", default="./pretrained/sam-med2d/sam-med2d_b.pth")
    parser.add_argument("--save-path", default="./paper_artifacts/runs/sammed2d/seed_42/checkpoint.pth")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-resolution", type=int, default=512, choices=(256, 512))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--overwrite-history", action="store_true")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument("--episodes-per-epoch", type=int, default=1500)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--domain-sampling-alpha", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cache-dataset", action="store_true", default=True)
    parser.add_argument("--no-cache-dataset", action="store_false", dest="cache_dataset")
    parser.add_argument("--preload-val-cache", action="store_true")
    parser.add_argument("--preload-train-cache", action="store_true")
    parser.add_argument("--augmentation", action="store_true", default=True)
    parser.add_argument("--no-augmentation", action="store_false", dest="augmentation")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    return parser.parse_args()


def sample_domain_batch(dataset, domain: str, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    pool = dataset.get_organ_indices(domain, positive_only=False)
    indices = np.random.choice(
        pool,
        size=int(batch_size),
        replace=len(pool) < int(batch_size),
    ).tolist()
    images = torch.stack([dataset[index][0] for index in indices])
    masks = torch.stack([dataset[index][1] for index in indices])
    return images, masks


def modality(domain: str) -> str:
    if domain in {"thyroid", "TN3K"}:
        return "thyroid"
    if domain in {"BUSI_WHU", "BUS-BRA"}:
        return "BUSI"
    return "generic"


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    amp_enabled: bool,
    threshold: float,
    batch_size: int,
) -> dict:
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    values = defaultdict(lambda: {"dice": [], "iou": [], "hd95": []})
    for images, masks, domains in tqdm(loader, desc="Validating SAM-Med2D", leave=False):
        images = images.to(device, non_blocking=True)
        with amp_context(device, amp_enabled):
            logits = model.forward_prompt_free(images)
        predictions = (
            torch.softmax(logits.float(), dim=1)[:, 1]
            .ge(threshold)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        targets = masks.numpy().astype(np.uint8)
        for prediction, target, domain in zip(predictions, targets, domains):
            domain = str(domain)
            values[domain]["dice"].append(binary_dice(prediction, target))
            values[domain]["iou"].append(binary_iou(prediction, target))
            values[domain]["hd95"].append(binary_hd95(prediction, target))

    per_domain = {
        domain: {name: float(np.mean(sequence)) for name, sequence in values[domain].items()}
        for domain in DOMAIN_NAMES
    }
    counts = {domain: len(values[domain]["dice"]) for domain in DOMAIN_NAMES}
    total = sum(counts.values())
    return {
        "dice": float(np.mean([per_domain[domain]["dice"] for domain in DOMAIN_NAMES])),
        "iou": float(np.mean([per_domain[domain]["iou"] for domain in DOMAIN_NAMES])),
        "hd95": float(np.mean([per_domain[domain]["hd95"] for domain in DOMAIN_NAMES])),
        "weighted_dice": float(
            sum(per_domain[domain]["dice"] * counts[domain] for domain in DOMAIN_NAMES) / total
        ),
        "weighted_iou": float(
            sum(per_domain[domain]["iou"] * counts[domain] for domain in DOMAIN_NAMES) / total
        ),
        "weighted_hd95": float(
            sum(per_domain[domain]["hd95"] * counts[domain] for domain in DOMAIN_NAMES) / total
        ),
        "per_domain": per_domain,
    }


def flatten_validation(result: dict) -> dict:
    row = {
        "val_macro_dice": result["dice"],
        "val_macro_iou": result["iou"],
        "val_macro_hd95": result["hd95"],
        "val_weighted_dice": result["weighted_dice"],
        "val_weighted_iou": result["weighted_iou"],
        "val_weighted_hd95": result["weighted_hd95"],
    }
    for domain in DOMAIN_NAMES:
        for name in ("dice", "iou", "hd95"):
            row[f"val_{domain}_{name}"] = result["per_domain"][domain][name]
    return row


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    set_seed(args.seed, args.deterministic)
    device = resolve_device(args.device)
    amp_enabled = args.amp and device.type == "cuda"

    save_path = Path(args.save_path)
    history_path = Path(str(save_path) + ".history.csv")
    last_path = Path(str(save_path) + ".last.pth")
    config_path = Path(str(save_path) + ".config.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if args.resume is None and history_path.exists():
        if args.overwrite_history:
            history_path.unlink()
        else:
            raise FileExistsError(f"History already exists: {history_path}")

    data_config = DataConfig(data_root=args.data_root, image_size=DEFAULT_IMAGE_SIZE)
    train_dataset, val_dataset, _ = get_datasets(
        data_config.roots(),
        image_size=DEFAULT_IMAGE_SIZE,
        seed=args.seed,
        train_query_positive_ratio=1.0,
        prefer_empty_train_queries=False,
        cache_in_memory=args.cache_dataset,
        use_cpu_clahe=False,
    )
    train_dataset.image_transform = None
    val_dataset.image_transform = None
    if args.preload_train_cache:
        train_dataset.warm_cache()
    if args.preload_val_cache:
        val_dataset.warm_cache()

    model = OfficialSAMMed2DAdapter(
        official_repository_path=args.official_repo,
        base_checkpoint=args.base_checkpoint,
        model_resolution=args.model_resolution,
        output_resolution=DEFAULT_METRIC_SIZE,
    ).to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = make_scaler(amp_enabled)
    loss_config = {
        "ce": DEFAULT_LOSS_CONFIG["ce"],
        "dice": DEFAULT_LOSS_CONFIG["dice"],
        "focal_tversky": DEFAULT_LOSS_CONFIG["focal_tversky"],
        "tversky_alpha": DEFAULT_LOSS_CONFIG["tversky_alpha"],
        "tversky_beta": DEFAULT_LOSS_CONFIG["tversky_beta"],
        "tversky_gamma": DEFAULT_LOSS_CONFIG["tversky_gamma"],
    }
    domain_sizes = {
        domain: len(train_dataset.get_organ_indices(domain))
        for domain in DOMAIN_NAMES
    }

    start_epoch = 0
    best = -1.0
    patience_counter = 0
    if args.resume:
        checkpoint = safe_torch_load(args.resume)
        model.load_trainable_state_dict(checkpoint["trainable_delta"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint.get("scaler_state_dict", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint["best_val_dice"])
        patience_counter = int(checkpoint.get("patience_counter", 0))

    official_commit_path = Path(args.official_repo) / "OFFICIAL_COMMIT_USED.txt"
    official_commit = (
        official_commit_path.read_text().strip()
        if official_commit_path.exists()
        else "unknown"
    )
    run_config = {
        "method": "sammed2d_promptfree_adapted512",
        "classification": "automatic segmentation main-table baseline",
        "architecture": "official OpenGVLab SAM-Med2D ViT-B",
        "official_code_commit": official_commit,
        "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
        "base_checkpoint_sha256": sha256_file(args.base_checkpoint),
        "model_resolution": args.model_resolution,
        "metric_resolution": DEFAULT_METRIC_SIZE,
        "prompt_policy_train": "none",
        "prompt_policy_val": "none",
        "prompt_policy_test": "none",
        "test_mask_used_for_prompt": False,
        "postprocess": False,
        "threshold": args.threshold,
        "selection_metric": "validation domain-macro Dice",
        "external_pretraining": "SAM-Med2D medical pretraining",
        "load_report": model.load_report,
        "parameter_report": model.parameter_report(),
        "loss_config": loss_config,
        "batch_size": args.batch_size,
        "effective_batch_size": args.batch_size * args.grad_accum_steps,
        "domain_sampling_alpha": args.domain_sampling_alpha,
        "training_args": vars(args),
    }
    write_json(run_config, config_path)

    print("=" * 100)
    print("SAM-Med2D prompt-free automatic comparison")
    print(f"logical_device={device}; physical GPU is selected by CUDA_VISIBLE_DEVICES")
    print(f"model_resolution={args.model_resolution}; metric_resolution={DEFAULT_METRIC_SIZE}")
    print(f"batch_size={args.batch_size}; effective_batch={args.batch_size * args.grad_accum_steps}")
    print(f"load_report={model.load_report}")
    print(f"parameters={model.parameter_report()}")
    print("No point, box, mask prompt, test-mask-derived information, threshold tuning, or post-processing.")
    print("=" * 100)

    for epoch in range(start_epoch, args.epochs):
        started = time.time()
        current_lr = cosine_lr(
            optimizer,
            epoch,
            args.epochs,
            args.warmup_epochs,
            args.lr,
            args.min_lr,
        )
        model.train()
        schedule = make_domain_schedule(
            DOMAIN_NAMES,
            args.episodes_per_epoch,
            domain_sizes=domain_sizes,
            alpha=args.domain_sampling_alpha,
        )
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_dice = 0.0

        progress = tqdm(
            enumerate(schedule),
            total=len(schedule),
            desc=f"Epoch {epoch + 1}/{args.epochs} [SAM-Med2D]",
        )
        for step, domain in progress:
            images, masks = sample_domain_batch(train_dataset, domain, args.batch_size)
            images = images.to(device, non_blocking=True)
            masks = masks.long().to(device, non_blocking=True)
            if args.augmentation:
                augmented_images, augmented_masks = augment_episode_medical(
                    images.mul(2).sub(1),
                    masks[:, None].float(),
                    modality=modality(domain),
                )
                images = augmented_images.add(1).mul(0.5).clamp(0, 1)
                masks = augmented_masks[:, 0].long()

            with amp_context(device, amp_enabled):
                logits = model.forward_prompt_free(images)
                loss = segmentation_loss(logits, masks, loss_config)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite SAM-Med2D loss at epoch={epoch + 1}, step={step + 1}"
                )
            scaler.scale(loss / args.grad_accum_steps).backward()
            if (step + 1) % args.grad_accum_steps == 0 or step + 1 == len(schedule):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                prediction = torch.softmax(logits.float(), dim=1)[:, 1].ge(args.threshold)
                target = masks.eq(1)
                intersection = (prediction & target).flatten(1).sum(1).float()
                denominator = (
                    prediction.flatten(1).sum(1).float()
                    + target.flatten(1).sum(1).float()
                )
                batch_dice = float(
                    (2.0 * intersection / denominator.clamp_min(1.0)).mean().item()
                )
            running_loss += float(loss.detach())
            running_dice += batch_dice
            count = step + 1
            progress.set_postfix(
                loss=f"{running_loss / count:.4f}",
                dice=f"{running_dice / count:.4f}",
                domain=domain,
                lr=f"{current_lr:.1e}",
            )

        validation = evaluate(
            model,
            val_dataset,
            device,
            amp_enabled,
            args.threshold,
            args.val_batch_size,
        )
        improved = validation["dice"] > best + 1e-8
        if improved:
            best = validation["dice"]
            patience_counter = 0
        else:
            patience_counter += 1

        payload = {
            "schema_version": 2,
            "method": "sammed2d_promptfree_adapted512",
            "trainable_delta": model.trainable_state_dict(),
            "model_config": model.export_config(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "best_val_dice": best,
            "patience_counter": patience_counter,
            "validation": validation,
            "threshold": args.threshold,
            "metric_resolution": DEFAULT_METRIC_SIZE,
            "postprocess": False,
            "training_args": vars(args),
            "base_checkpoint_sha256": run_config["base_checkpoint_sha256"],
        }
        atomic_torch_save(payload, last_path)
        if improved:
            atomic_torch_save(payload, save_path)

        row = {
            "epoch": epoch + 1,
            "lr": current_lr,
            "train_loss": running_loss / len(schedule),
            "train_dice": running_dice / len(schedule),
            **flatten_validation(validation),
            "best_val_dice": best,
            "improved": int(improved),
            "patience_counter": patience_counter,
            "epoch_seconds": time.time() - started,
            "microbatches_seen": len(schedule),
            "training_samples_seen": len(schedule) * args.batch_size,
            "optimizer_updates": int(np.ceil(len(schedule) / args.grad_accum_steps)),
        }
        append_csv(history_path, row)
        print(
            f"Epoch {epoch + 1:03d}: train loss={row['train_loss']:.4f} "
            f"Dice={row['train_dice']:.4f} | val macro Dice={validation['dice']:.4f} "
            f"IoU={validation['iou']:.4f} HD95={validation['hd95']:.2f} | "
            f"best={best:.4f} patience={patience_counter}/{args.patience}"
        )
        for domain in DOMAIN_NAMES:
            metrics = validation["per_domain"][domain]
            print(
                f"  {domain:10s}: Dice={metrics['dice']:.4f} "
                f"IoU={metrics['iou']:.4f} HD95={metrics['hd95']:.2f}"
            )
        if patience_counter >= args.patience:
            print("[early stop]")
            break

    print("Best delta:", save_path)
    print("History:", history_path)


if __name__ == "__main__":
    main()

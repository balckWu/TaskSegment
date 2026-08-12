#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paperlab.progress import paper_tqdm as tqdm

from tasksegment.configs import (
    DOMAIN_NAMES,
    DataConfig,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_METRIC_SIZE,
)
from tasksegment.data import get_datasets, set_seed
from tasksegment.models.lvit_official import OfficialLViTAdapter
from tasksegment.training.metrics import binary_dice, binary_hd95, binary_iou


def safe_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def amp_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def summarize(values):
    per_domain = {}
    counts = {}
    for domain in DOMAIN_NAMES:
        per_domain[domain] = {
            name: float(np.mean(values[domain][name]))
            for name in ("dice", "iou", "hd95")
        }
        counts[domain] = len(values[domain]["dice"])
    return {
        "macro_dice": float(np.mean([per_domain[d]["dice"] for d in DOMAIN_NAMES])),
        "macro_iou": float(np.mean([per_domain[d]["iou"] for d in DOMAIN_NAMES])),
        "macro_hd95": float(np.mean([per_domain[d]["hd95"] for d in DOMAIN_NAMES])),
        "weighted_dice": float(
            sum(per_domain[d]["dice"] * counts[d] for d in DOMAIN_NAMES)
            / sum(counts.values())
        ),
        "per_domain": per_domain,
    }


@torch.inference_mode()
def evaluate(model, dataset, text_bank, device, batch_size, use_amp, bn_batch_stats):
    model.eval()

    # Diagnostic only: keep the whole network in eval mode, but make BatchNorm
    # use current validation-batch statistics. momentum=0 prevents running
    # mean/variance from being changed in memory.
    if bn_batch_stats:
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train()
                module.momentum = 0.0

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    values = defaultdict(lambda: {"dice": [], "iou": [], "hd95": []})

    for images, masks, domains in tqdm(
        loader,
        desc=("BN batch stats" if bn_batch_stats else "Normal eval"),
        leave=False,
    ):
        images = images.to(device, non_blocking=True)
        text = torch.cat(
            [text_bank[str(domain)] for domain in domains],
            dim=0,
        ).to(device, non_blocking=True)

        with amp_context(device, use_amp):
            logits = model(images, text)

        predictions = (
            torch.softmax(logits.float(), dim=1)[:, 1]
            .ge(0.5)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        targets = masks.numpy().astype(np.uint8)

        for pred, target, domain in zip(predictions, targets, domains):
            domain = str(domain)
            values[domain]["dice"].append(binary_dice(pred, target))
            values[domain]["iou"].append(binary_iou(pred, target))
            values[domain]["hd95"].append(binary_hd95(pred, target))

    return summarize(values)


def print_result(name, result):
    print("=" * 88)
    print(
        f"{name}: macro Dice={result['macro_dice']:.4f} "
        f"IoU={result['macro_iou']:.4f} "
        f"HD95={result['macro_hd95']:.2f} "
        f"weighted Dice={result['weighted_dice']:.4f}"
    )
    for domain in DOMAIN_NAMES:
        metric = result["per_domain"][domain]
        print(
            f"  {domain:10s}: Dice={metric['dice']:.4f} "
            f"IoU={metric['iou']:.4f} HD95={metric['hd95']:.2f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="paper_artifacts/runs/lvit_native224/seed_42/checkpoint.pth",
    )
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--official-repo", default="./third_party/LViT")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint = safe_load(checkpoint_path)
    training_args = checkpoint.get("training_args", {})
    seed = int(training_args.get("seed", 42))
    set_seed(seed, deterministic=False)

    device = torch.device(args.device)
    model_config = checkpoint.get("model_config", {})
    resolution = int(model_config.get("model_resolution", 512))

    data_config = DataConfig(
        data_root=args.data_root,
        image_size=DEFAULT_IMAGE_SIZE,
    )
    _, val_dataset, _ = get_datasets(
        data_config.roots(),
        image_size=DEFAULT_IMAGE_SIZE,
        seed=seed,
        train_query_positive_ratio=1.0,
        prefer_empty_train_queries=False,
        cache_in_memory=True,
        use_cpu_clahe=False,
    )
    val_dataset.image_transform = None
    val_dataset.warm_cache()

    model = OfficialLViTAdapter(
        official_repository_path=args.official_repo,
        model_resolution=resolution,
        output_resolution=DEFAULT_METRIC_SIZE,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    text_bank = {
        domain: tensor.float().cpu()
        for domain, tensor in checkpoint["text_embeddings"].items()
    }

    normal = evaluate(
        model,
        val_dataset,
        text_bank,
        device,
        args.batch_size,
        not args.no_amp,
        bn_batch_stats=False,
    )
    print_result("Normal model.eval()", normal)

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    batch_stats = evaluate(
        model,
        val_dataset,
        text_bank,
        device,
        args.batch_size,
        not args.no_amp,
        bn_batch_stats=True,
    )
    print_result("Eval with BN batch statistics (diagnostic only)", batch_stats)

    gain = batch_stats["macro_dice"] - normal["macro_dice"]
    print("=" * 88)
    print(f"BN diagnostic Dice gain: {gain:+.4f}")
    if gain >= 0.10:
        print("Conclusion: strong BatchNorm running-statistics mismatch.")
    elif gain >= 0.03:
        print("Conclusion: BatchNorm mismatch is materially contributing.")
    else:
        print("Conclusion: BatchNorm mismatch is not the main explanation.")


if __name__ == "__main__":
    main()

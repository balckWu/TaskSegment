#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paperlab.progress import paper_tqdm as tqdm

from tasksegment.configs import DOMAIN_NAMES, DataConfig, DEFAULT_IMAGE_SIZE
from tasksegment.data import get_datasets, set_seed
from tasksegment.models.lvit_official import OfficialLViTAdapter
from tasksegment.training.metrics import binary_dice, binary_hd95, binary_iou


def safe_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def amp_context(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def write_csv(path: Path, rows):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def text_for_domains(bank: Mapping[str, torch.Tensor], domains: Sequence[str], device):
    return torch.cat([bank[str(d)] for d in domains], dim=0).to(device, non_blocking=True)


def summarize(method: str, rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["domain"]].append(row)
    summary = []
    for domain in DOMAIN_NAMES:
        values = grouped[domain]
        hd = [float(v["hd95"]) for v in values]
        summary.append({
            "method": method,
            "domain": domain,
            "n": len(values),
            "dice": float(np.mean([float(v["dice"]) for v in values])),
            "iou": float(np.mean([float(v["iou"]) for v in values])),
            "hd95": float(np.mean(hd)),
            "hd95_median": float(np.median(hd)),
            "hd95_p90": float(np.percentile(hd, 90)),
        })
    summary.append({
        "method": method,
        "domain": "macro",
        "n": sum(v["n"] for v in summary),
        "dice": float(np.mean([v["dice"] for v in summary])),
        "iou": float(np.mean([v["iou"] for v in summary])),
        "hd95": float(np.mean([v["hd95"] for v in summary])),
        "hd95_median": float(np.mean([v["hd95_median"] for v in summary])),
        "hd95_p90": float(np.mean([v["hd95_p90"] for v in summary])),
    })
    return summary


def update_master(path: Path, method: str, summary):
    existing = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
    existing = [row for row in existing if row.get("method") != method]
    write_csv(path, existing + summary)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="paper_artifacts/manual/lvit_native224_seed42/checkpoint.pth")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--output-root", default="./comparisons/predictions")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", action="store_false", dest="amp")
    p.add_argument("--cache-dataset", action="store_true", default=True)
    p.add_argument("--no-cache-dataset", action="store_false", dest="cache_dataset")
    p.add_argument("--preload-cache", action="store_true")
    p.add_argument("--save-masks", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.chdir(ROOT)
    set_seed(args.seed, False)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = safe_load(args.checkpoint)
    if checkpoint.get("method") != "lvit_protocol":
        raise ValueError("Invalid LViT checkpoint")
    method = checkpoint.get("method_name", "lvit_native224_domain_prompts")
    raw_model_config = dict(checkpoint["model_config"])
    supported_model_keys = {
        "official_repository_path",
        "model_resolution",
        "output_resolution",
    }
    ignored_model_config = sorted(
        set(raw_model_config) - supported_model_keys
    )
    if ignored_model_config:
        print(
            "[checkpoint] ignoring non-constructor model_config fields: "
            + ", ".join(ignored_model_config)
        )
    model_config = {
        key: raw_model_config[key]
        for key in supported_model_keys
        if key in raw_model_config
    }
    model = OfficialLViTAdapter(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    text_bank = checkpoint["text_embeddings"]
    threshold = float(checkpoint.get("threshold", 0.5))

    data_cfg = DataConfig(data_root=args.data_root, image_size=DEFAULT_IMAGE_SIZE)
    _, _, dataset = get_datasets(
        data_cfg.roots(),
        image_size=DEFAULT_IMAGE_SIZE,
        seed=args.seed,
        train_query_positive_ratio=1.0,
        prefer_empty_train_queries=False,
        cache_in_memory=args.cache_dataset,
        use_cpu_clahe=False,
    )
    dataset.image_transform = None
    if args.preload_cache:
        dataset.warm_cache()

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    output_dir = Path(args.output_root) / method
    rows = []
    offset = 0
    with torch.inference_mode():
        for images, masks, domains in tqdm(loader, desc=f"Testing {method}"):
            images = images.to(device, non_blocking=True)
            text = text_for_domains(text_bank, domains, device)
            with amp_context(device, args.amp):
                logits = model(images, text)
            pred = torch.softmax(logits.float(), dim=1)[:, 1].ge(threshold).cpu().numpy().astype(np.uint8)
            target = masks.numpy().astype(np.uint8)
            for j, (p, t, domain) in enumerate(zip(pred, target, domains)):
                info = dataset.get_sample_info(offset + j)
                sample_id = str(info["sample_id"])
                if args.save_masks:
                    mask_path = output_dir / "masks" / str(domain) / f"{sample_id}.png"
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(p * 255, mode="L").save(mask_path)
                rows.append({
                    "method": method,
                    "index": offset + j,
                    "domain": str(domain),
                    "sample_id": sample_id,
                    "image_path": info["image"],
                    "mask_path": info["mask"],
                    "dice": binary_dice(p, t),
                    "iou": binary_iou(p, t),
                    "hd95": binary_hd95(p, t),
                    "pred_foreground_pixels": int(p.sum()),
                    "target_foreground_pixels": int(t.sum()),
                })
            offset += len(domains)

    summary = summarize(method, rows)
    write_csv(output_dir / "per_case_metrics.csv", rows)
    write_csv(output_dir / "summary_metrics.csv", summary)
    (output_dir / "inference_config.json").write_text(json.dumps({
        "method": method,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "threshold": threshold,
        "postprocess": False,
        "test_mask_used_for_prompting": False,
        "model_config": checkpoint["model_config"],
        "prompts": checkpoint["prompts"],
        "text_metadata": checkpoint["text_metadata"],
    }, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    update_master(Path(args.output_root) / "comparison_summary.csv", method, summary)

    print("=" * 90)
    for row in summary:
        print(
            f"[{row['domain']}] n={row['n']} Dice={row['dice']:.4f} "
            f"IoU={row['iou']:.4f} HD95(mean/median/p90)="
            f"{row['hd95']:.2f}/{row['hd95_median']:.2f}/{row['hd95_p90']:.2f}"
        )
    print(f"Per-case metrics: {output_dir / 'per_case_metrics.csv'}")
    print("=" * 90)


if __name__ == "__main__":
    main()

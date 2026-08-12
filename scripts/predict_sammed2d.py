#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paperlab.progress import paper_tqdm as tqdm
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from sammed2d_protocol_utils import (
    amp_context,
    safe_torch_load,
    sha256_file,
    summarize_case_rows,
    update_summary_file,
    write_csv,
)
from tasksegment.configs import DOMAIN_NAMES, DataConfig, DEFAULT_IMAGE_SIZE, DEFAULT_METRIC_SIZE
from tasksegment.data import get_datasets, set_seed
from tasksegment.models.sammed2d_official import OfficialSAMMed2DAdapter
from tasksegment.training.metrics import binary_dice, binary_hd95, binary_iou


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate automatic or oracle SAM-Med2D protocols")
    parser.add_argument("--mode", choices=("prompt_free", "gt_box_oracle"), required=True)
    parser.add_argument("--delta-checkpoint", default=None)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--official-repo", default="./third_party/SAM-Med2D")
    parser.add_argument("--base-checkpoint", default="./pretrained/sam-med2d/sam-med2d_b.pth")
    parser.add_argument("--output-root", default="./paper_artifacts/predictions")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--cache-dataset", action="store_true", default=True)
    parser.add_argument("--no-cache-dataset", action="store_false", dest="cache_dataset")
    parser.add_argument("--preload-cache", action="store_true")
    parser.add_argument("--save-masks", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    set_seed(args.seed, False)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.mode == "prompt_free":
        if not args.delta_checkpoint:
            raise ValueError("--delta-checkpoint is required for prompt_free mode")
        delta = safe_torch_load(args.delta_checkpoint)
        if delta.get("method") != "sammed2d_promptfree_adapted512":
            raise ValueError("The checkpoint is not the automatic SAM-Med2D prompt-free delta")
        model_resolution = int(delta["model_config"]["model_resolution"])
        method = "sammed2d_promptfree_adapted512"
        test_mask_used_for_prompt = False
        summary_file = Path(args.output_root) / "comparison_summary.csv"
    else:
        if args.delta_checkpoint:
            raise ValueError("Oracle mode evaluates the base model; do not pass a delta checkpoint")
        delta = None
        model_resolution = 256
        method = "sammed2d_pretrained_gtbox_oracle256"
        test_mask_used_for_prompt = True
        summary_file = Path(args.output_root) / "comparison_oracle_summary.csv"

    model = OfficialSAMMed2DAdapter(
        official_repository_path=args.official_repo,
        base_checkpoint=args.base_checkpoint,
        model_resolution=model_resolution,
        output_resolution=DEFAULT_METRIC_SIZE,
    )
    if delta is not None:
        if sha256_file(args.base_checkpoint) != delta["base_checkpoint_sha256"]:
            raise ValueError("Base checkpoint SHA differs from the one used during adaptation")
        model.load_trainable_state_dict(delta["trainable_delta"])
    model.to(device).eval()

    data_config = DataConfig(data_root=args.data_root, image_size=DEFAULT_IMAGE_SIZE)
    _, _, dataset = get_datasets(
        data_config.roots(),
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
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    output_dir = Path(args.output_root) / method
    rows: list[dict] = []
    offset = 0
    with torch.inference_mode():
        for images, masks, domains in tqdm(loader, desc=f"Testing {method}"):
            images = images.to(device, non_blocking=True)
            masks_device = masks.to(device, non_blocking=True)
            with amp_context(device, args.amp and device.type == "cuda"):
                if args.mode == "prompt_free":
                    logits = model.forward_prompt_free(images)
                    boxes = None
                else:
                    logits, boxes = model.forward_gt_box_oracle(images, masks_device)
            predictions = (
                torch.softmax(logits.float(), dim=1)[:, 1]
                .ge(args.threshold)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
            targets = masks.numpy().astype(np.uint8)

            for local_index, (prediction, target, domain) in enumerate(
                zip(predictions, targets, domains)
            ):
                index = offset + local_index
                domain = str(domain)
                info = dataset.get_sample_info(index)
                row = {
                    "method": method,
                    "index": index,
                    "domain": domain,
                    "sample_id": str(info["sample_id"]),
                    "image_path": info["image"],
                    "mask_path": info["mask"],
                    "dice": binary_dice(prediction, target),
                    "iou": binary_iou(prediction, target),
                    "hd95": binary_hd95(prediction, target),
                    "pred_foreground_pixels": int(prediction.sum()),
                    "target_foreground_pixels": int(target.sum()),
                    "test_mask_used_for_prompt": int(test_mask_used_for_prompt),
                }
                if boxes is not None:
                    box = boxes[local_index] if getattr(boxes, "ndim", 1) > 1 else boxes
                    box = torch.as_tensor(box).detach().cpu().flatten()
                    if box.numel() >= 4:
                        row.update(
                            {
                                "oracle_box_x0": float(box[0]),
                                "oracle_box_y0": float(box[1]),
                                "oracle_box_x1": float(box[2]),
                                "oracle_box_y1": float(box[3]),
                            }
                        )
                rows.append(row)
                if args.save_masks:
                    mask_path = output_dir / "masks" / domain / f"{info['sample_id']}.png"
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(prediction * 255, mode="L").save(mask_path)
            offset += len(domains)

    summary = summarize_case_rows(method, rows, DOMAIN_NAMES)
    write_csv(output_dir / "per_case_metrics.csv", rows)
    write_csv(output_dir / "summary_metrics.csv", summary)
    update_summary_file(summary_file, method, summary)
    (output_dir / "inference_config.json").write_text(
        json.dumps(
            {
                "method": method,
                "classification": (
                    "automatic main-table baseline"
                    if args.mode == "prompt_free"
                    else "interactive/oracle upper bound; exclude from automatic main table"
                ),
                "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
                "base_checkpoint_sha256": sha256_file(args.base_checkpoint),
                "model_resolution": model_resolution,
                "metric_resolution": DEFAULT_METRIC_SIZE,
                "batch_size": args.batch_size,
                "threshold": args.threshold,
                "postprocess": False,
                "prompt_policy": (
                    "none"
                    if args.mode == "prompt_free"
                    else "tight box deterministically derived from each test ground-truth mask"
                ),
                "test_mask_used_for_prompt": test_mask_used_for_prompt,
                "load_report": model.load_report,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    for row in summary:
        print(
            f"[{row['domain']}] n={row['n']} Dice={row['dice']:.4f} "
            f"IoU={row['iou']:.4f} HD95={row['hd95']:.2f}"
        )
    print("Per-case metrics:", output_dir / "per_case_metrics.csv")
    print("Summary table:", summary_file)


if __name__ == "__main__":
    main()

from __future__ import annotations

"""Evaluate a fine-tuned official CLIPSeg checkpoint on the six-domain test set."""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paperlab.progress import paper_tqdm as tqdm

from tasksegment.configs import DataConfig, DEFAULT_IMAGE_SIZE, DEFAULT_METRIC_SIZE, DOMAIN_NAMES  # noqa: E402
from tasksegment.data import MultiOrganDataset, set_seed  # noqa: E402
from tasksegment.training.metrics import binary_dice, binary_hd95, binary_iou  # noqa: E402

try:
    from transformers import AutoTokenizer, CLIPSegForImageSegmentation
except ImportError as exc:
    raise ImportError(
        "CLIPSeg requires Hugging Face Transformers. Install with: "
        "python -m pip install -r requirements-clipseg.txt"
    ) from exc


def safe_torch_load(path: str | Path, map_location: str | torch.device = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def amp_context(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return nullcontext()
    if hasattr(torch, "autocast"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.cuda.amp.autocast(enabled=True)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L").save(path)


def normalize_clip_images(images: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (images.repeat(1, 3, 1, 1).clamp(0.0, 1.0) - mean) / std


@torch.inference_mode()
def build_prompt_bank(model, tokenizer, prompts: Mapping[str, str], device: torch.device):
    texts = [prompts[domain] for domain in DOMAIN_NAMES]
    tokenized = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    tokenized = {key: value.to(device) for key, value in tokenized.items()}
    embeddings = model.get_conditional_embeddings(
        batch_size=len(texts),
        input_ids=tokenized["input_ids"],
        attention_mask=tokenized.get("attention_mask"),
    )
    return {domain: embeddings[i : i + 1].detach() for i, domain in enumerate(DOMAIN_NAMES)}


def summarize_rows(method: str, rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["domain"])].append(row)
    result: List[Dict[str, Any]] = []
    for domain in DOMAIN_NAMES:
        values = grouped.get(domain, [])
        if not values:
            continue
        hd95_values = [float(v["hd95"]) for v in values]
        result.append(
            {
                "method": method,
                "domain": domain,
                "n": len(values),
                "dice": float(np.mean([float(v["dice"]) for v in values])),
                "iou": float(np.mean([float(v["iou"]) for v in values])),
                "hd95": float(np.mean(hd95_values)),
                "hd95_median": float(np.median(hd95_values)),
            }
        )
    if result:
        result.append(
            {
                "method": method,
                "domain": "macro",
                "n": int(sum(int(v["n"]) for v in result)),
                "dice": float(np.mean([float(v["dice"]) for v in result])),
                "iou": float(np.mean([float(v["iou"]) for v in result])),
                "hd95": float(np.mean([float(v["hd95"]) for v in result])),
                "hd95_median": float(np.mean([float(v["hd95_median"]) for v in result])),
            }
        )
    return result


def refresh_comparison_summary(output_root: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for path in sorted(output_root.glob("*/summary_metrics.csv")):
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    if rows:
        write_csv(output_root / "comparison_summary.csv", rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate official CLIPSeg on the six-domain test split")
    parser.add_argument("--checkpoint", default="comparisons/runs/clipseg_official_seed42/model.pth")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--output-root", default="./comparisons/predictions")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=None, help="Default: checkpoint threshold")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cache-dataset", action="store_true")
    parser.add_argument("--preload-cache", action="store_true")
    parser.add_argument("--no-save-predictions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    set_seed(args.seed, deterministic=True)
    device = torch.device(
        "cpu" if args.device.startswith("cuda") and not torch.cuda.is_available() else args.device
    )
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
    if checkpoint.get("method") != "clipseg_official":
        raise ValueError(f"Not a clipseg_official checkpoint: {checkpoint_path}")

    base_model = str(checkpoint["base_model"])
    prompts = {domain: str(checkpoint["prompts"][domain]) for domain in DOMAIN_NAMES}
    threshold = float(checkpoint.get("threshold", 0.5) if args.threshold is None else args.threshold)
    mean_values = checkpoint.get("image_mean")
    std_values = checkpoint.get("image_std")
    if mean_values is None or std_values is None:
        raise KeyError("Checkpoint is missing CLIP image mean/std statistics")
    mean = torch.tensor(mean_values, device=device).view(1, 3, 1, 1)
    std = torch.tensor(std_values, device=device).view(1, 3, 1, 1)

    model = CLIPSegForImageSegmentation.from_pretrained(base_model)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    prompt_bank = build_prompt_bank(model, tokenizer, prompts, device)

    data_cfg = DataConfig(data_root=args.data_root, image_size=tuple(DEFAULT_IMAGE_SIZE))
    dataset = MultiOrganDataset(
        data_cfg.roots(),
        image_transform=None,
        split="test",
        image_size=tuple(DEFAULT_IMAGE_SIZE),
        seed=args.seed,
        cache_in_memory=args.cache_dataset,
        use_cpu_clahe=False,
    )
    if args.cache_dataset and args.preload_cache:
        dataset.warm_cache(show_progress=True)

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    method = "clipseg_official"
    method_dir = output_root / method
    method_dir.mkdir(parents=True, exist_ok=True)
    (method_dir / "inference_config.json").write_text(
        json.dumps(
            {
                "method": method,
                "checkpoint": str(checkpoint_path),
                "base_model": base_model,
                "train_mode": checkpoint.get("train_mode"),
                "prompts": prompts,
                "threshold": threshold,
                "input_resolution": list(DEFAULT_IMAGE_SIZE),
                "metric_resolution": list(DEFAULT_METRIC_SIZE),
                "postprocess": False,
                "test_mask_used_for_prompting": False,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 96)
    print("CLIPSeg official evaluation")
    print(f"checkpoint={checkpoint_path}")
    print(f"base_model={base_model} train_mode={checkpoint.get('train_mode')}")
    print(f"test_samples={len(dataset)} input=512x512 metric=512x512 threshold={threshold}")
    print(f"prompts={prompts}")
    print("No test mask is used for prompting, threshold selection, or parameter selection.")
    print("No post-processing is applied.")
    print("=" * 96)

    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    rows: List[Dict[str, Any]] = []
    cursor = 0
    with torch.inference_mode():
        for images, masks, domains in tqdm(loader, desc="Testing CLIPSeg", dynamic_ncols=True):
            batch_size = int(images.shape[0])
            images = images.to(device, non_blocking=True)
            conditional = torch.cat([prompt_bank[str(domain)] for domain in domains], dim=0)
            pixel_values = normalize_clip_images(images, mean, std)
            with amp_context(device, bool(args.amp)):
                output = model(
                    pixel_values=pixel_values,
                    conditional_embeddings=conditional,
                    interpolate_pos_encoding=True,
                )
                logits = output.logits
            if logits.shape[-2:] != tuple(DEFAULT_METRIC_SIZE):
                logits = F.interpolate(
                    logits.unsqueeze(1).float(),
                    size=tuple(DEFAULT_METRIC_SIZE),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            predictions = torch.sigmoid(logits.float()).ge(threshold).cpu().numpy().astype(np.uint8)
            targets = masks.numpy().astype(np.uint8)

            for offset in range(batch_size):
                index = cursor + offset
                info = dataset.get_sample_info(index)
                domain = str(domains[offset])
                prediction = predictions[offset]
                target = targets[offset]
                if not args.no_save_predictions:
                    save_mask(method_dir / "masks" / domain / f"{info['sample_id']}.png", prediction)
                rows.append(
                    {
                        "method": method,
                        "domain": domain,
                        "sample_id": info["sample_id"],
                        "dice": binary_dice(prediction, target),
                        "iou": binary_iou(prediction, target),
                        "hd95": binary_hd95(prediction, target),
                        "threshold": threshold,
                    }
                )
            cursor += batch_size

    write_csv(method_dir / "per_case_metrics.csv", rows)
    summary = summarize_rows(method, rows)
    write_csv(method_dir / "summary_metrics.csv", summary)
    refresh_comparison_summary(output_root)

    for item in summary:
        if item["domain"] == "macro":
            continue
        domain_rows = [row for row in rows if row["domain"] == item["domain"]]
        p90 = float(np.percentile([float(row["hd95"]) for row in domain_rows], 90))
        print(
            f"[{item['domain']}] Dice={float(item['dice']):.4f} IoU={float(item['iou']):.4f} "
            f"HD95(mean/median/p90)={float(item['hd95']):.2f}/"
            f"{float(item['hd95_median']):.2f}/{p90:.2f}"
        )
    macro = next(item for item in summary if item["domain"] == "macro")
    print(
        f"[macro] Dice={float(macro['dice']):.4f} IoU={float(macro['iou']):.4f} "
        f"HD95={float(macro['hd95']):.2f}"
    )
    print(f"[info] per-case metrics: {method_dir / 'per_case_metrics.csv'}")
    print(f"[info] comparison summary: {output_root / 'comparison_summary.csv'}")


if __name__ == "__main__":
    main()

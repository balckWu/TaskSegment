#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import json
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paperlab.progress import paper_tqdm as tqdm

from tasksegment.configs import DataConfig, DEFAULT_IMAGE_SIZE, DEFAULT_METRIC_SIZE, DOMAIN_NAMES
from tasksegment.data import MultiOrganDataset, get_image_transform, set_seed
from tasksegment.training.metrics import binary_dice, binary_hd95, binary_iou


METHODS = {
    "unet": {
        "script": ROOT / "scripts" / "train_unet.py",
        "class_name": "UNet2D",
    },
    "unetpp": {
        "script": ROOT / "scripts" / "train_unetpp.py",
        "class_name": "UNetPlusPlus2D",
    },
    "swinunet_official": {
        "module": "tasksegment.models.swinunet_official",
        "class_name": "OfficialSwinUNet2D",
    },
    "transunet_official": {
        "module": "tasksegment.models.transunet_official",
        "class_name": "OfficialTransUNet2D",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained automatic segmentation baseline")
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--cache-dataset", action="store_true", default=True)
    parser.add_argument("--no-cache-dataset", action="store_false", dest="cache_dataset")
    parser.add_argument("--preload-cache", action="store_true")
    parser.add_argument("--save-masks", action="store_true")
    return parser.parse_args()


def safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def import_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def import_model_class(method: str):
    spec = METHODS[method]
    if "module" in spec:
        module = __import__(spec["module"], fromlist=[spec["class_name"]])
    else:
        module = import_module_from_path(f"_paper_{method}", Path(spec["script"]))
    return getattr(module, spec["class_name"])


def sanitize_model_config(model_class, raw_config: Mapping[str, Any]) -> dict[str, Any]:
    config = dict(raw_config or {})
    for key in (
        "implementation",
        "official_repository_path",
        "official_internal_in_channels",
        "official_internal_channels",
        "token_grid",
    ):
        config.pop(key, None)
    if "pretrained_checkpoint" in config:
        config["pretrained_checkpoint"] = None

    signature = inspect.signature(model_class.__init__)
    allowed = {
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    filtered = {key: value for key, value in config.items() if key in allowed}
    if "image_size" in filtered:
        filtered["image_size"] = tuple(int(v) for v in filtered["image_size"])
    return filtered


def resolve_logits(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, Mapping):
        for key in ("logits", "out", "prediction", "pred"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value
        tensors = [value for value in output.values() if isinstance(value, torch.Tensor)]
        if tensors:
            return tensors[-1]
    if isinstance(output, (tuple, list)):
        tensors = [value for value in output if isinstance(value, torch.Tensor)]
        if tensors:
            return tensors[-1]
    raise TypeError(f"Unsupported model output type: {type(output)!r}")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(method: str, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["domain"])].append(row)

    summary: list[dict[str, Any]] = []
    for domain in DOMAIN_NAMES:
        values = grouped.get(domain, [])
        if not values:
            continue
        hd95 = np.asarray([float(v["hd95"]) for v in values], dtype=np.float64)
        summary.append(
            {
                "method": method,
                "domain": domain,
                "n": len(values),
                "dice": float(np.mean([float(v["dice"]) for v in values])),
                "iou": float(np.mean([float(v["iou"]) for v in values])),
                "hd95": float(hd95.mean()),
                "hd95_median": float(np.median(hd95)),
                "hd95_p90": float(np.percentile(hd95, 90)),
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed, deterministic=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = safe_torch_load(checkpoint_path)
    model_class = import_model_class(args.method)
    model_config = sanitize_model_config(model_class, checkpoint.get("model_config", {}))
    model = model_class(**model_config)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise KeyError("Checkpoint does not contain model_state_dict")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    data_config = DataConfig(args.data_root, tuple(DEFAULT_IMAGE_SIZE))
    dataset = MultiOrganDataset(
        data_config.roots(),
        get_image_transform(),
        split="test",
        image_size=tuple(DEFAULT_IMAGE_SIZE),
        seed=args.seed,
        cache_in_memory=args.cache_dataset,
        use_cpu_clahe=False,
    )
    if args.preload_cache:
        dataset.warm_cache(show_progress=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    rows: list[dict[str, Any]] = []
    offset = 0
    def autocast_context():
        if args.amp and device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    with torch.inference_mode():
        for images, masks, domains in tqdm(loader, desc=f"Testing {args.method}"):
            images = images.to(device, non_blocking=True)
            with autocast_context():
                logits = resolve_logits(model(images))
            if tuple(logits.shape[-2:]) != tuple(DEFAULT_METRIC_SIZE):
                logits = F.interpolate(
                    logits.float(),
                    size=tuple(DEFAULT_METRIC_SIZE),
                    mode="bilinear",
                    align_corners=False,
                )
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
                data_index = offset + local_index
                info = dataset.get_sample_info(data_index)
                row = {
                    "method": args.method,
                    "index": data_index,
                    "domain": str(domain),
                    "sample_id": str(info["sample_id"]),
                    "image_path": str(info["image"]),
                    "mask_path": str(info["mask"]),
                    "dice": binary_dice(prediction, target),
                    "iou": binary_iou(prediction, target),
                    "hd95": binary_hd95(prediction, target),
                    "pred_foreground_pixels": int(prediction.sum()),
                    "target_foreground_pixels": int(target.sum()),
                }
                rows.append(row)
                if args.save_masks:
                    mask_path = output_dir / "masks" / str(domain) / f"{info['sample_id']}.png"
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(prediction * 255, mode="L").save(mask_path)
            offset += len(domains)

    summary = summarize(args.method, rows)
    write_csv(output_dir / "per_case_metrics.csv", rows)
    write_csv(output_dir / "summary_metrics.csv", summary)
    (output_dir / "inference_config.json").write_text(
        json.dumps(
            {
                "method": args.method,
                "checkpoint": str(checkpoint_path),
                "model_config": model_config,
                "input_resolution": list(DEFAULT_IMAGE_SIZE),
                "metric_resolution": list(DEFAULT_METRIC_SIZE),
                "batch_size": args.batch_size,
                "threshold": args.threshold,
                "postprocess": False,
                "seed": args.seed,
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


if __name__ == "__main__":
    main()

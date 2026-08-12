#!/usr/bin/env python3
"""Convert the TaskSegment six-domain split into nnU-Net v2 2D PNG format.

The converter intentionally reuses ``tasksegment.data.get_datasets`` so that
nnU-Net sees exactly the same train/validation/test cases and the same 512x512
image/mask tensors as the other baselines.

Output:
    <work-dir>/nnUNet_raw/Dataset<id>_<name>/
        imagesTr/
        labelsTr/
        imagesTs/
        labelsTs/               # project-side ground truth for final evaluation
        dataset.json
        tasksegment_manifest.json
        splits_final.json       # copied to nnUNet_preprocessed after preprocessing

Training cases are the union of the original train and validation sets. A custom
fold 0 split preserves the original train/validation partition.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paperlab.progress import paper_tqdm as tqdm

from tasksegment.configs import DOMAIN_NAMES, DataConfig  # noqa: E402
from tasksegment.data import get_datasets, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the six-domain TaskSegment dataset for nnU-Net v2."
    )
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--work-dir", default="./comparisons/nnunet")
    parser.add_argument("--dataset-id", type=int, default=501)
    parser.add_argument("--dataset-name", default="TaskSegment6Domain")
    parser.add_argument("--image-size", type=int, nargs=2, default=(512, 512))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--cache-dataset",
        action="store_true",
        help="Enable the TaskSegment in-memory cache while exporting.",
    )
    return parser.parse_args()


def sanitize_token(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")
    return value or "unknown"


def tensor_to_image_uint8(value: Any) -> np.ndarray:
    tensor = torch.as_tensor(value).detach().cpu().float()
    while tensor.ndim > 2 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim == 3:
        # Grayscale is expected. For defensive compatibility, average channels.
        tensor = tensor.mean(dim=0)
    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2D image tensor, got shape={tuple(tensor.shape)}")

    array = tensor.numpy()
    finite = np.isfinite(array)
    if not finite.all():
        array = np.where(finite, array, 0.0)

    lo = float(array.min())
    hi = float(array.max())
    if lo >= 0.0 and hi <= 1.0 + 1e-6:
        array = array * 255.0
    elif lo >= 0.0 and hi <= 255.0 + 1e-6:
        pass
    else:
        # Dataset tensors may be standardized. Preserve rank/intensity contrast
        # deterministically by robustly mapping the 0.5-99.5 percentile range.
        p_lo, p_hi = np.percentile(array, [0.5, 99.5])
        if p_hi <= p_lo:
            p_lo, p_hi = lo, hi
        if p_hi > p_lo:
            array = (array - p_lo) / (p_hi - p_lo) * 255.0
        else:
            array = np.zeros_like(array)

    return np.clip(np.rint(array), 0, 255).astype(np.uint8)


def tensor_to_mask_uint8(value: Any) -> np.ndarray:
    tensor = torch.as_tensor(value).detach().cpu()
    while tensor.ndim > 2 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim == 3:
        # Handle one-hot masks if a future dataset implementation returns them.
        if tensor.shape[0] == 2:
            tensor = tensor.argmax(dim=0)
        else:
            tensor = tensor[0]
    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2D mask tensor, got shape={tuple(tensor.shape)}")
    return tensor.gt(0).numpy().astype(np.uint8)


def unpack_sample(sample: Any) -> Tuple[Any, Any, str]:
    if isinstance(sample, dict):
        image = sample.get("image")
        mask = sample.get("mask", sample.get("label"))
        domain = sample.get("domain", sample.get("organ", "unknown"))
        if image is None or mask is None:
            raise ValueError(f"Unsupported sample dictionary keys: {sample.keys()}")
        return image, mask, str(domain)

    if isinstance(sample, (tuple, list)) and len(sample) >= 2:
        image, mask = sample[0], sample[1]
        domain = sample[2] if len(sample) >= 3 else "unknown"
        if isinstance(domain, (tuple, list)) and len(domain) == 1:
            domain = domain[0]
        return image, mask, str(domain)

    raise TypeError(f"Unsupported dataset sample type: {type(sample)!r}")


def expected_domain(dataset: Any, index: int, fallback: str) -> str:
    for attr in ("samples", "records", "items"):
        records = getattr(dataset, attr, None)
        if records is None or index >= len(records):
            continue
        record = records[index]
        if isinstance(record, dict):
            for key in ("domain", "organ", "dataset"):
                if key in record:
                    return str(record[key])
        for key in ("domain", "organ", "dataset"):
            value = getattr(record, key, None)
            if value is not None:
                return str(value)
    return fallback


def export_split(
    dataset: Any,
    split_name: str,
    image_dir: Path,
    label_dir: Path,
    *,
    case_offset: int = 0,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    case_ids: List[str] = []
    records: List[Dict[str, Any]] = []

    for index in tqdm(range(len(dataset)), desc=f"Exporting {split_name}", unit="case"):
        image, mask, domain_from_sample = unpack_sample(dataset[index])
        domain = expected_domain(dataset, index, domain_from_sample)
        domain_token = sanitize_token(domain)
        case_id = f"{domain_token}_{split_name}_{index + case_offset:06d}"

        image_array = tensor_to_image_uint8(image)
        mask_array = tensor_to_mask_uint8(mask)
        if image_array.shape != mask_array.shape:
            raise ValueError(
                f"Geometry mismatch for {case_id}: "
                f"image={image_array.shape}, mask={mask_array.shape}"
            )

        Image.fromarray(image_array, mode="L").save(image_dir / f"{case_id}_0000.png")
        Image.fromarray(mask_array, mode="L").save(label_dir / f"{case_id}.png")

        case_ids.append(case_id)
        records.append(
            {
                "case_id": case_id,
                "split": split_name,
                "domain": domain,
                "height": int(image_array.shape[0]),
                "width": int(image_array.shape[1]),
                "foreground_pixels": int(mask_array.sum()),
            }
        )
    return case_ids, records


def write_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    if not 0 <= args.dataset_id <= 999:
        raise ValueError("--dataset-id must be between 0 and 999")
    if tuple(args.image_size) != (512, 512):
        print(
            f"[warning] requested image size is {tuple(args.image_size)}, "
            "while the controlled comparison protocol uses (512, 512)."
        )

    set_seed(args.seed, deterministic=True)

    work_dir = Path(args.work_dir).resolve()
    raw_root = work_dir / "nnUNet_raw"
    preprocessed_root = work_dir / "nnUNet_preprocessed"
    results_root = work_dir / "nnUNet_results"
    dataset_folder = raw_root / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"

    if dataset_folder.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{dataset_folder} already exists. Pass --overwrite to rebuild it."
            )
        shutil.rmtree(dataset_folder)

    images_tr = dataset_folder / "imagesTr"
    labels_tr = dataset_folder / "labelsTr"
    images_ts = dataset_folder / "imagesTs"
    labels_ts = dataset_folder / "labelsTs"
    for folder in (images_tr, labels_tr, images_ts, labels_ts, preprocessed_root, results_root):
        folder.mkdir(parents=True, exist_ok=True)

    data_cfg = DataConfig(
        data_root=args.data_root,
        image_size=tuple(int(x) for x in args.image_size),
    )
    train_dataset, val_dataset, test_dataset = get_datasets(
        data_cfg.roots(),
        image_size=tuple(int(x) for x in args.image_size),
        seed=args.seed,
        train_query_positive_ratio=1.0,
        prefer_empty_train_queries=False,
        cache_in_memory=args.cache_dataset,
        use_cpu_clahe=False,
    )

    train_ids, train_records = export_split(
        train_dataset, "train", images_tr, labels_tr
    )
    val_ids, val_records = export_split(
        val_dataset, "val", images_tr, labels_tr
    )
    test_ids, test_records = export_split(
        test_dataset, "test", images_ts, labels_ts
    )

    dataset_json = {
        "channel_names": {"0": "Ultrasound"},
        "labels": {"background": 0, "foreground": 1},
        "numTraining": len(train_ids) + len(val_ids),
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
        "name": f"Dataset{args.dataset_id:03d}_{args.dataset_name}",
        "description": (
            "Unified six-domain 2D ultrasound segmentation dataset exported "
            "from the TaskSegment train/validation/test split."
        ),
    }
    (dataset_folder / "dataset.json").write_text(
        json.dumps(dataset_json, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    splits = [{"train": train_ids, "val": val_ids}]
    (dataset_folder / "splits_final.json").write_text(
        json.dumps(splits, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "dataset_id": args.dataset_id,
        "dataset_name": args.dataset_name,
        "source_data_root": str(Path(args.data_root).resolve()),
        "image_size": list(args.image_size),
        "seed": args.seed,
        "domain_names": list(DOMAIN_NAMES),
        "counts": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
        "environment": {
            "nnUNet_raw": str(raw_root),
            "nnUNet_preprocessed": str(preprocessed_root),
            "nnUNet_results": str(results_root),
        },
        "records": train_records + val_records + test_records,
    }
    (dataset_folder / "tasksegment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(dataset_folder / "tasksegment_manifest.csv", manifest["records"])

    env_script = work_dir / "env.sh"
    env_script.write_text(
        "#!/usr/bin/env bash\n"
        f'export nnUNet_raw="{raw_root}"\n'
        f'export nnUNet_preprocessed="{preprocessed_root}"\n'
        f'export nnUNet_results="{results_root}"\n',
        encoding="utf-8",
    )
    env_script.chmod(0o755)

    print("\nPreparation complete.")
    print(f"dataset_folder={dataset_folder}")
    print(f"train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")
    print(f"environment_script={env_script}")
    print("\nNext:")
    print(f"  source {env_script}")
    print(
        f"  nnUNetv2_plan_and_preprocess -d {args.dataset_id} "
        "-c 2d --verify_dataset_integrity"
    )


if __name__ == "__main__":
    main()

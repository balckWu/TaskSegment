from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tasksegment.checkpoint import load_checkpoint_model, safe_torch_load
from tasksegment.configs import (
    DataConfig,
    DEFAULT_DOMAIN_THRESHOLDS,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_METRIC_SIZE,
    DEFAULT_POSTPROCESS_CFG,
)
from tasksegment.data import MultiOrganDataset, get_image_transform, set_seed
from tasksegment.text import load_text_bank
from tasksegment.training.evaluation import (
    get_task_tokens,
    logits_to_foreground_probability,
    postprocess_mask,
    resize_binary_mask,
)
from tasksegment.training.metrics import binary_dice, binary_hd95, binary_iou


def _parse_float_list(value: str) -> List[float]:
    result = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected at least one float")
    return result


def _parse_int_list(value: str) -> List[int]:
    result = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return result


def _parse_bool_list(value: str) -> List[bool]:
    mapping = {"1": True, "true": True, "yes": True, "0": False, "false": False, "no": False}
    result: List[bool] = []
    for token in value.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token not in mapping:
            raise argparse.ArgumentTypeError(f"invalid boolean value: {token}")
        result.append(mapping[token])
    if not result:
        raise argparse.ArgumentTypeError("expected at least one boolean")
    return result


def _unique(values: Iterable[Any]) -> List[Any]:
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Calibrate per-domain thresholds and optional morphology using the validation split only. "
            "All metrics are computed on a unified 512x512 grid."
        )
    )
    p.add_argument("--model-path", default="USTS.pth")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--text-dir", default=None)
    p.add_argument("--output-json", default="validation_calibration_512.json")
    p.add_argument("--output-checkpoint", default=None, help="Optional new checkpoint; source checkpoint is never overwritten")
    p.add_argument("--threshold-min", type=float, default=0.35)
    p.add_argument("--threshold-max", type=float, default=0.65)
    p.add_argument("--threshold-step", type=float, default=0.025)
    p.add_argument("--min-area-ratios", type=_parse_float_list, default=_parse_float_list("0,0.00005,0.0001,0.0002,0.0005"))
    p.add_argument("--closing-kernels", type=_parse_int_list, default=_parse_int_list("0,3"))
    p.add_argument("--fill-holes-options", type=_parse_bool_list, default=_parse_bool_list("false,true"))
    p.add_argument("--keep-largest-options", type=_parse_bool_list, default=_parse_bool_list("false,true"))
    p.add_argument(
        "--dice-tolerance",
        type=float,
        default=0.002,
        help="Among candidates within this absolute Dice of the best validation Dice, choose the lowest HD95.",
    )
    p.add_argument("--only-domain", action="append", default=[], help="May be repeated; default calibrates every domain")
    search = p.add_mutually_exclusive_group()
    search.add_argument("--search-postprocess", dest="search_postprocess", action="store_true")
    search.add_argument("--threshold-only", dest="search_postprocess", action="store_false")
    p.set_defaults(search_postprocess=True)
    p.add_argument("--max-grid-size", type=int, default=5000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)

    cache = p.add_mutually_exclusive_group()
    cache.add_argument("--cache-dataset", dest="cache_dataset", action="store_true")
    cache.add_argument("--no-cache-dataset", dest="cache_dataset", action="store_false")
    p.set_defaults(cache_dataset=True)

    preload = p.add_mutually_exclusive_group()
    preload.add_argument("--preload-cache", dest="preload_cache", action="store_true")
    preload.add_argument("--no-preload-cache", dest="preload_cache", action="store_false")
    p.set_defaults(preload_cache=True)
    return p.parse_args()


def _compute_metrics(predictions: Sequence[np.ndarray], targets: Sequence[np.ndarray]) -> Dict[str, float]:
    dice = [binary_dice(pred, target) for pred, target in zip(predictions, targets)]
    iou = [binary_iou(pred, target) for pred, target in zip(predictions, targets)]
    hd95 = [binary_hd95(pred, target) for pred, target in zip(predictions, targets)]
    hd = np.asarray(hd95, dtype=np.float64)
    return {
        "dice": float(np.mean(dice)),
        "iou": float(np.mean(iou)),
        "hd95": float(np.mean(hd)),
        "hd95_median": float(np.median(hd)),
        "hd95_p90": float(np.percentile(hd, 90)),
    }


def _config_complexity(cfg: Dict[str, Any]) -> tuple:
    active = int(float(cfg["min_area_ratio"]) > 0.0)
    active += int(bool(cfg["fill_holes"]))
    active += int(bool(cfg["keep_largest"]))
    active += int(int(cfg["closing_kernel"]) > 1)
    return (
        active,
        float(cfg["min_area_ratio"]),
        int(bool(cfg["keep_largest"])),
        int(bool(cfg["fill_holes"])),
        int(cfg["closing_kernel"]),
    )


def _candidate_key(candidate: Dict[str, Any]) -> tuple:
    # First minimize HD95 within the allowed Dice band. Then prefer higher Dice,
    # higher IoU, simpler morphology and a threshold closer to 0.5.
    return (
        float(candidate["hd95"]),
        -float(candidate["dice"]),
        -float(candidate["iou"]),
        _config_complexity(candidate["postprocess"]),
        abs(float(candidate["threshold"]) - 0.5),
    )


def main():
    args = parse_args()
    if args.threshold_step <= 0:
        raise ValueError("--threshold-step must be positive")
    if args.threshold_max < args.threshold_min:
        raise ValueError("--threshold-max must be >= --threshold-min")
    if args.dice_tolerance < 0:
        raise ValueError("--dice-tolerance must be non-negative")

    set_seed(args.seed, deterministic=True)
    device = torch.device("cpu" if args.device == "cuda" and not torch.cuda.is_available() else args.device)
    model, bank, model_config, checkpoint = load_checkpoint_model(args.model_path, device, allow_partial_load=False)
    model.eval()
    train_cfg = dict(checkpoint.get("train_config", {}))
    image_size = tuple(DEFAULT_IMAGE_SIZE)
    metric_size = tuple(DEFAULT_METRIC_SIZE)
    text_dir = args.text_dir or train_cfg.get("text_dir", "./text_features_llm")

    data_cfg = DataConfig(args.data_root, image_size)
    val = MultiOrganDataset(
        data_cfg.roots(),
        get_image_transform(),
        "val",
        image_size,
        args.seed,
        cache_in_memory=args.cache_dataset,
    )
    if args.cache_dataset and args.preload_cache:
        val.warm_cache(show_progress=True)

    center_text_features = bool(train_cfg.get("center_text_features", True))
    text_bank = (
        load_text_bank(
            text_dir,
            device,
            expected_text_dim=int(model_config["text_dim"]),
            center_across_domains=center_text_features,
        )
        if model_config["use_ftext"]
        else {}
    )

    selected_domains = set(args.only_domain)
    cases: Dict[str, List[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    with torch.inference_mode():
        for idx in range(len(val)):
            img, resized_label, organ = val[idx]
            if selected_domains and organ not in selected_domains:
                continue
            target = resize_binary_mask(resized_label.numpy(), metric_size)
            x = img.unsqueeze(0).to(device)
            tokens = get_task_tokens(model, bank, text_bank, organ, device)
            out = model.segment_with_task(xq=x, task_tokens=tokens, output_size=x.shape[-2:])
            prob = logits_to_foreground_probability(out["pred_masks"], metric_size)
            cases[organ].append((prob, target))

    if not cases:
        raise RuntimeError("No validation cases matched the requested domains")

    threshold_grid = np.arange(
        args.threshold_min,
        args.threshold_max + 0.5 * args.threshold_step,
        args.threshold_step,
        dtype=np.float64,
    )
    threshold_grid = [float(np.clip(x, 0.0, 1.0)) for x in threshold_grid]

    if args.search_postprocess:
        pp_grid = [
            {
                "min_area": 0,
                "min_area_ratio": float(ratio),
                "keep_largest": bool(keep),
                "fill_holes": bool(fill),
                "closing_kernel": int(kernel),
            }
            for ratio, keep, fill, kernel in itertools.product(
                _unique(args.min_area_ratios),
                _unique(args.keep_largest_options),
                _unique(args.fill_holes_options),
                _unique(args.closing_kernels),
            )
        ]
    else:
        pp_grid = [dict(DEFAULT_POSTPROCESS_CFG)]

    grid_size = len(threshold_grid) * len(pp_grid)
    if grid_size > int(args.max_grid_size):
        raise ValueError(
            f"calibration grid has {grid_size} candidates/domain, exceeding --max-grid-size={args.max_grid_size}"
        )

    calibration: Dict[str, Dict[str, Any]] = {}
    domain_thresholds: Dict[str, float] = dict(train_cfg.get("domain_thresholds", DEFAULT_DOMAIN_THRESHOLDS))
    domain_thresholds.setdefault("default", 0.5)
    existing_pp = train_cfg.get("postprocess_cfg", {"default": dict(DEFAULT_POSTPROCESS_CFG)})
    if isinstance(existing_pp, dict) and any(isinstance(v, dict) for v in existing_pp.values()):
        postprocess_by_domain: Dict[str, Dict[str, Any]] = {
            str(k): dict(v) for k, v in existing_pp.items() if isinstance(v, dict)
        }
        postprocess_by_domain.setdefault("default", dict(DEFAULT_POSTPROCESS_CFG))
    else:
        postprocess_by_domain = {"default": dict(existing_pp) if isinstance(existing_pp, dict) else dict(DEFAULT_POSTPROCESS_CFG)}

    for organ, organ_cases in sorted(cases.items()):
        probabilities = [case[0] for case in organ_cases]
        targets = [case[1] for case in organ_cases]
        baseline_predictions = [(prob >= 0.5).astype(np.uint8) for prob in probabilities]
        baseline = _compute_metrics(baseline_predictions, targets)

        candidates: List[Dict[str, Any]] = []
        for threshold in threshold_grid:
            raw_predictions = [(prob >= threshold).astype(np.uint8) for prob in probabilities]
            for pp_cfg in pp_grid:
                if pp_cfg == DEFAULT_POSTPROCESS_CFG:
                    predictions = raw_predictions
                else:
                    predictions = [postprocess_mask(pred, **pp_cfg) for pred in raw_predictions]
                metrics = _compute_metrics(predictions, targets)
                candidates.append({
                    "threshold": float(threshold),
                    "postprocess": dict(pp_cfg),
                    **metrics,
                })

        best_dice = max(float(c["dice"]) for c in candidates)
        eligible = [c for c in candidates if float(c["dice"]) >= best_dice - float(args.dice_tolerance) - 1e-12]
        chosen = min(eligible, key=_candidate_key)

        domain_thresholds[organ] = float(chosen["threshold"])
        postprocess_by_domain[organ] = dict(chosen["postprocess"])
        calibration[organ] = {
            "cases": len(organ_cases),
            "baseline_raw_threshold_0_5": baseline,
            "best_validation_dice_over_grid": best_dice,
            "dice_tolerance": float(args.dice_tolerance),
            "eligible_candidates": len(eligible),
            "candidates_evaluated": len(candidates),
            "selected": chosen,
        }
        print(
            f"[{organ}] threshold={chosen['threshold']:.3f} "
            f"Dice={chosen['dice']:.4f} IoU={chosen['iou']:.4f} "
            f"HD95(mean/median/p90)={chosen['hd95']:.2f}/{chosen['hd95_median']:.2f}/{chosen['hd95_p90']:.2f} "
            f"post={chosen['postprocess']}"
        )

    payload = {
        "source_checkpoint": str(Path(args.model_path).resolve()),
        "split": "val",
        "input_resolution": list(image_size),
        "metric_resolution": list(metric_size),
        "selection_rule": (
            "Find maximum mean per-case Dice; among configurations within dice_tolerance of that maximum, "
            "select minimum mean HD95, then higher Dice/IoU and simpler morphology."
        ),
        "search_postprocess": bool(args.search_postprocess),
        "threshold_grid": threshold_grid,
        "postprocess_grid": pp_grid,
        "domain_thresholds": domain_thresholds,
        "postprocess_cfg": postprocess_by_domain,
        "per_domain": calibration,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"calibration saved to {args.output_json}")

    if args.output_checkpoint:
        source = safe_torch_load(args.model_path, map_location="cpu")
        if not isinstance(source, dict):
            raise ValueError("invalid checkpoint")
        updated = dict(source)
        updated_train_cfg = dict(updated.get("train_config", {}))
        updated_train_cfg["image_size"] = tuple(image_size)
        updated_train_cfg["metric_size"] = tuple(metric_size)
        updated_train_cfg["domain_thresholds"] = domain_thresholds
        updated_train_cfg["postprocess_cfg"] = postprocess_by_domain
        updated_train_cfg["validation_calibration"] = payload
        updated_train_cfg["validation_original_size"] = False  # legacy field for explicit compatibility
        updated["train_config"] = updated_train_cfg
        Path(args.output_checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save(updated, args.output_checkpoint)
        print(f"calibrated checkpoint saved to {args.output_checkpoint}")


if __name__ == "__main__":
    main()

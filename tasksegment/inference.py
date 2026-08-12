from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from paperlab.progress import paper_tqdm as tqdm

from .checkpoint import load_checkpoint_model
from .configs import DEFAULT_DOMAIN_THRESHOLDS, DEFAULT_METRIC_SIZE, DEFAULT_POSTPROCESS_CFG_BY_DOMAIN
from .data.datasets import MultiOrganDataset
from .training.evaluation import (
    PostprocessConfig,
    get_task_tokens,
    logits_to_foreground_probability,
    normalize_metric_size,
    probability_to_binary_mask,
    resize_binary_mask,
    resolve_postprocess_cfg,
)
from .training.memory import EMATaskMemoryBank
from .training.metrics import binary_dice, binary_hd95, binary_iou


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def denormalize_for_vis(img_tensor: torch.Tensor) -> np.ndarray:
    arr = img_tensor.squeeze().detach().cpu().numpy().astype(np.float32)
    lo, hi = np.percentile(arr, [1, 99])
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max() + 1e-6)
    return (255.0 * np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)).astype(np.uint8)


def to_heatmap_np(x: torch.Tensor | None, target_shape=None) -> Optional[np.ndarray]:
    if x is None:
        return None
    prob = torch.sigmoid(x)
    if target_shape is not None and tuple(prob.shape[-2:]) != tuple(target_shape):
        prob = torch.nn.functional.interpolate(prob, size=target_shape, mode="bilinear", align_corners=False)
    return np.clip(prob.squeeze().detach().cpu().numpy(), 0.0, 1.0)


def load_model(model_path: str, device: torch.device, allow_partial_load: bool = False):
    model, bank, _, checkpoint = load_checkpoint_model(model_path, device, allow_partial_load)
    return model, bank, checkpoint.get("train_config", {})


def save_interpretability_panel(
    save_path: str,
    img_np: np.ndarray,
    gt_np: np.ndarray,
    pred_np: np.ndarray,
    fg_prob: np.ndarray,
    guidance: np.ndarray | None,
    dice_val: float,
    hd95_val: float,
) -> None:
    pred_bool, gt_bool = pred_np.astype(bool), gt_np.astype(bool)
    err = np.zeros_like(gt_np, dtype=np.uint8)
    err[pred_bool & gt_bool] = 1
    err[pred_bool & ~gt_bool] = 2
    err[~pred_bool & gt_bool] = 3
    panels = [
        ("Image", img_np, "gray"),
        ("GT", gt_np, "gray"),
        ("Pred", pred_np, "gray"),
        ("Error 1=TP 2=FP 3=FN", err, "viridis"),
        ("FG probability", fg_prob, "viridis"),
        ("Dense guidance", guidance, "magma"),
    ]
    plt.figure(figsize=(19.2, 3.4))
    for i, (title, arr, cmap) in enumerate(panels, 1):
        plt.subplot(1, len(panels), i)
        plt.imshow(np.zeros_like(gt_np) if arr is None else arr, cmap=cmap)
        plt.title(title)
        plt.axis("off")
    plt.suptitle(f"Dice={dice_val:.3f} | HD95={hd95_val:.2f} px @ {gt_np.shape[0]}x{gt_np.shape[1]}", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def _metric_triplet(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    return {
        "dice": binary_dice(pred, target),
        "iou": binary_iou(pred, target),
        "hd95": binary_hd95(pred, target),
    }


def _component_stats(mask: np.ndarray) -> Dict[str, float]:
    binary = (mask > 0).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return {"components": 0, "largest_component_area": 0, "largest_component_ratio": 0.0}
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
    total = int(areas.sum())
    largest = int(areas.max()) if areas.size else 0
    return {
        "components": int(n - 1),
        "largest_component_area": largest,
        "largest_component_ratio": float(largest / max(total, 1)),
    }


def _summary(values: Dict[str, List[float]], prefix: str) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for key, seq in values.items():
        arr = np.asarray(seq, dtype=np.float64)
        result[f"{prefix}_{key}"] = float(np.mean(arr))
        if key == "hd95":
            result[f"{prefix}_{key}_median"] = float(np.median(arr))
            result[f"{prefix}_{key}_p90"] = float(np.percentile(arr, 90))
            result[f"{prefix}_{key}_p95"] = float(np.percentile(arr, 95))
    return result


def predict_all_organs(
    model: torch.nn.Module,
    test_dataset: MultiOrganDataset,
    device: torch.device,
    text_bank: Dict[str, torch.Tensor],
    ema_bank: EMATaskMemoryBank,
    save_dir: str = "./vis",
    domain_thresholds: Optional[Dict[str, float]] = None,
    postprocess_cfg: Optional[PostprocessConfig] = None,
    use_postprocess: bool = False,
    num_vis_per_organ: int = 5,
    vis_data_indices: Optional[List[int]] = None,
    guidance_scale: Optional[float] = None,
    report_both: bool = True,
    metric_size: Sequence[int] = DEFAULT_METRIC_SIZE,
) -> Dict[str, Dict[str, float]]:
    """Evaluate every dataset at a common metric resolution (default 512x512)."""
    os.makedirs(save_dir, exist_ok=True)
    metric_shape = normalize_metric_size(metric_size)
    thresholds = dict(DEFAULT_DOMAIN_THRESHOLDS if domain_thresholds is None else domain_thresholds)
    pp_cfg: PostprocessConfig = DEFAULT_POSTPROCESS_CFG_BY_DOMAIN if postprocess_cfg is None else postprocess_cfg
    if guidance_scale is not None:
        model.decoder.dense_guidance_strength = float(guidance_scale)
    model.eval()

    results: Dict[str, Dict[str, float]] = {}
    records: List[Dict[str, Any]] = []
    targets_to_vis = set(int(x) for x in (vis_data_indices or []))

    with torch.inference_mode():
        for organ in sorted(test_dataset.organ_to_indices):
            indices = test_dataset.get_organ_indices(organ)
            organ_dir = os.path.join(save_dir, organ)
            os.makedirs(organ_dir, exist_ok=True)
            raw_values = {"dice": [], "iou": [], "hd95": []}
            post_values = {"dice": [], "iou": [], "hd95": []}
            resolved_pp = resolve_postprocess_cfg(organ, pp_cfg)

            for sample_i, data_idx in enumerate(tqdm(indices, desc=f"测试 {organ}")):
                img, resized_label, _ = test_dataset[data_idx]
                info = test_dataset.get_sample_info(data_idx)
                target = resize_binary_mask(resized_label.numpy(), metric_shape)
                xq = img.unsqueeze(0).to(device)
                tokens = get_task_tokens(model, ema_bank, text_bank, organ, device)
                out = model.segment_with_task(xq=xq, task_tokens=tokens, output_size=xq.shape[-2:])
                prob = logits_to_foreground_probability(out["pred_masks"], target_shape=metric_shape)
                raw_pred = probability_to_binary_mask(prob, organ, thresholds, pp_cfg, use_postprocess=False)
                post_pred = probability_to_binary_mask(prob, organ, thresholds, pp_cfg, use_postprocess=True)
                raw = _metric_triplet(raw_pred, target)
                post = _metric_triplet(post_pred, target)
                for key in raw_values:
                    raw_values[key].append(raw[key])
                    post_values[key].append(post[key])

                selected_pred = post_pred if use_postprocess else raw_pred
                selected = post if use_postprocess else raw
                pred_bool, target_bool = selected_pred.astype(bool), target.astype(bool)
                tp = int((pred_bool & target_bool).sum())
                fp = int((pred_bool & ~target_bool).sum())
                fn = int((~pred_bool & target_bool).sum())
                raw_comp = _component_stats(raw_pred)
                post_comp = _component_stats(post_pred)
                gt_comp = _component_stats(target)
                records.append({
                    "organ": organ,
                    "sample_id": info["sample_id"],
                    "data_idx": int(data_idx),
                    "original_height": int(info["original_shape"][0]),
                    "original_width": int(info["original_shape"][1]),
                    "metric_height": int(metric_shape[0]),
                    "metric_width": int(metric_shape[1]),
                    "raw_dice": raw["dice"],
                    "raw_iou": raw["iou"],
                    "raw_hd95": raw["hd95"],
                    "post_dice": post["dice"],
                    "post_iou": post["iou"],
                    "post_hd95": post["hd95"],
                    "reported_variant": "postprocess" if use_postprocess else "raw",
                    "dice": selected["dice"],
                    "iou": selected["iou"],
                    "hd95": selected["hd95"],
                    "gt_area": int(target_bool.sum()),
                    "pred_area": int(pred_bool.sum()),
                    "tp_area": tp,
                    "fp_area": fp,
                    "fn_area": fn,
                    "precision": tp / max(tp + fp, 1),
                    "recall": tp / max(tp + fn, 1),
                    "gt_components": gt_comp["components"],
                    "raw_pred_components": raw_comp["components"],
                    "raw_largest_component_ratio": raw_comp["largest_component_ratio"],
                    "post_pred_components": post_comp["components"],
                    "post_largest_component_ratio": post_comp["largest_component_ratio"],
                    "threshold": float(thresholds.get(organ, thresholds.get("default", 0.5))),
                    "postprocess_cfg": json.dumps(resolved_pp, ensure_ascii=False, sort_keys=True),
                    "guidance_scale": float(model.decoder.dense_guidance_strength),
                    "task_token_source": str(
                        getattr(model, "task_token_source", None)
                        or (
                            "training_support_visual_prototype_domain_text"
                            if bool(getattr(model, "use_visual_prototype", True))
                            else "null_visual_tokens_domain_text_ablation"
                        )
                    ),
                    "ablation_name": str(getattr(model, "ablation_name", "full")),
                    "condition_token_source": str(
                        getattr(
                            model,
                            "condition_token_source",
                            "structured_semantic_text_plus_domain_group_embeddings",
                        )
                    ),
                })

                if sample_i < num_vis_per_organ or data_idx in targets_to_vis:
                    guidance = to_heatmap_np(out.get("dense_guidance_logits_lowres"), metric_shape)
                    save_interpretability_panel(
                        os.path.join(organ_dir, f"{sample_i:04d}_{info['sample_id']}.png"),
                        denormalize_for_vis(xq),
                        target,
                        selected_pred,
                        prob,
                        guidance,
                        selected["dice"],
                        selected["hd95"],
                    )

            raw_summary = _summary(raw_values, "raw")
            post_summary = _summary(post_values, "post")
            active_prefix = "post" if use_postprocess else "raw"
            results[organ] = {
                "dice": post_summary["post_dice"] if use_postprocess else raw_summary["raw_dice"],
                "iou": post_summary["post_iou"] if use_postprocess else raw_summary["raw_iou"],
                "hd95": post_summary["post_hd95"] if use_postprocess else raw_summary["raw_hd95"],
                "hd95_median": post_summary["post_hd95_median"] if use_postprocess else raw_summary["raw_hd95_median"],
                "hd95_p90": post_summary["post_hd95_p90"] if use_postprocess else raw_summary["raw_hd95_p90"],
                "hd95_p95": post_summary["post_hd95_p95"] if use_postprocess else raw_summary["raw_hd95_p95"],
                **raw_summary,
                **post_summary,
                "count": len(indices),
                "metric_height": metric_shape[0],
                "metric_width": metric_shape[1],
            }
            print(
                f"[{organ}] {active_prefix}: Dice={results[organ]['dice']:.4f} "
                f"IoU={results[organ]['iou']:.4f} HD95(mean/median/p90)="
                f"{results[organ]['hd95']:.2f}/{results[organ]['hd95_median']:.2f}/{results[organ]['hd95_p90']:.2f}"
            )
            if report_both:
                print(
                    f"  raw Dice={raw_summary['raw_dice']:.4f}, post Dice={post_summary['post_dice']:.4f}; "
                    f"post_cfg={resolved_pp}"
                )

    _write_csv(os.path.join(save_dir, "per_case_metrics.csv"), records)
    summary_rows = [{"organ": organ, **values} for organ, values in results.items()]
    _write_csv(os.path.join(save_dir, "summary_metrics.csv"), summary_rows)
    print(f"[info] per-case metrics: {os.path.join(save_dir, 'per_case_metrics.csv')}")
    return results


def predict(*args: Any, **kwargs: Any):
    return predict_all_organs(*args, **kwargs)

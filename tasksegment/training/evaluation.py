from __future__ import annotations

from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Mapping, Optional, Sequence, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_fill_holes
from paperlab.progress import paper_tqdm as tqdm

from ..configs import (
    DEFAULT_DOMAIN_THRESHOLDS,
    DEFAULT_METRIC_SIZE,
    DEFAULT_POSTPROCESS_CFG,
)
from ..data.datasets import MultiOrganDataset
from ..text.bank import make_Ftext_batch
from .memory import EMATaskMemoryBank
from .metrics import binary_dice, binary_hd95, binary_iou

PostprocessValue = Union[int, float, bool]
PostprocessConfig = Mapping[str, Union[PostprocessValue, Mapping[str, PostprocessValue]]]


def normalize_metric_size(metric_size: Sequence[int] | None = None) -> tuple[int, int]:
    values = DEFAULT_METRIC_SIZE if metric_size is None else metric_size
    if len(values) != 2:
        raise ValueError("metric_size must contain exactly (height, width)")
    h, w = int(values[0]), int(values[1])
    if h <= 0 or w <= 0:
        raise ValueError("metric_size values must be positive")
    return h, w


def resize_binary_mask(mask: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    h, w = normalize_metric_size(target_shape)
    arr = (np.asarray(mask) > 0).astype(np.uint8)
    if tuple(arr.shape[-2:]) == (h, w):
        return arr
    return cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)


def resolve_postprocess_cfg(
    organ_name: str,
    postprocess_cfg: Optional[PostprocessConfig] = None,
) -> Dict[str, PostprocessValue]:
    """Resolve a flat or per-domain post-processing configuration.

    Supported formats:

    Flat/legacy::
        {"min_area_ratio": 0.0001, "fill_holes": True, ...}

    Per-domain::
        {
            "default": {...},
            "OTU": {"min_area_ratio": 0.0002, "keep_largest": True},
        }
    """
    resolved: Dict[str, PostprocessValue] = dict(DEFAULT_POSTPROCESS_CFG)
    if postprocess_cfg is None:
        return resolved

    cfg = dict(postprocess_cfg)
    nested = any(isinstance(value, Mapping) for value in cfg.values())
    if nested:
        default_cfg = cfg.get("default", {})
        domain_cfg = cfg.get(str(organ_name), {})
        if isinstance(default_cfg, Mapping):
            resolved.update(default_cfg)
        if isinstance(domain_cfg, Mapping):
            resolved.update(domain_cfg)
    else:
        resolved.update(cfg)  # legacy global configuration

    return {
        "min_area": max(0, int(resolved.get("min_area", 0))),
        "min_area_ratio": max(0.0, float(resolved.get("min_area_ratio", 0.0))),
        "keep_largest": bool(resolved.get("keep_largest", False)),
        "fill_holes": bool(resolved.get("fill_holes", False)),
        "closing_kernel": max(0, int(resolved.get("closing_kernel", 0))),
    }


def postprocess_mask(
    pred: np.ndarray,
    min_area: int = 0,
    min_area_ratio: float = 0.0,
    keep_largest: bool = False,
    fill_holes: bool = False,
    closing_kernel: int = 0,
) -> np.ndarray:
    pred = (pred > 0).astype(np.uint8)

    # Closing is applied before connected-component filtering so fragmented parts
    # that should form one lesion can be joined before small-component removal.
    if int(closing_kernel) > 1:
        k = int(closing_kernel)
        kernel = np.ones((k, k), np.uint8)
        pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel)

    if fill_holes:
        pred = binary_fill_holes(pred).astype(np.uint8)

    effective_min_area = max(int(min_area), int(round(float(min_area_ratio) * pred.size)))
    if effective_min_area > 0 or keep_largest:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred, connectivity=8)
        if num_labels > 1:
            areas = stats[:, cv2.CC_STAT_AREA]
            valid_ids = [i for i in range(1, num_labels) if areas[i] >= effective_min_area]
            out = np.zeros_like(pred)
            if valid_ids:
                if keep_largest:
                    valid_ids = [max(valid_ids, key=lambda i: areas[i])]
                for comp_id in valid_ids:
                    out[labels == comp_id] = 1
            pred = out

    return pred.astype(np.uint8)


def logits_to_foreground_probability(
    logits: torch.Tensor,
    target_shape: Sequence[int] | None = None,
) -> np.ndarray:
    prob = torch.softmax(logits, dim=1)[:, 1:2]
    if target_shape is not None:
        shape = normalize_metric_size(target_shape)
        if tuple(prob.shape[-2:]) != shape:
            prob = F.interpolate(prob, size=shape, mode="bilinear", align_corners=False)
    return prob.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)


def probability_to_binary_mask(
    fg_prob: np.ndarray,
    organ_name: str,
    domain_thresholds: Optional[Dict[str, float]] = None,
    postprocess_cfg: Optional[PostprocessConfig] = None,
    use_postprocess: bool = False,
) -> np.ndarray:
    thresholds = dict(DEFAULT_DOMAIN_THRESHOLDS if domain_thresholds is None else domain_thresholds)
    tau = float(thresholds.get(organ_name, thresholds.get("default", 0.5)))
    pred = (fg_prob >= tau).astype(np.uint8)
    if not use_postprocess:
        return pred
    cfg = resolve_postprocess_cfg(organ_name, postprocess_cfg)
    return postprocess_mask(pred, **cfg)


def logits_to_postprocessed_binary_mask(
    logits: torch.Tensor,
    organ_name: str,
    domain_thresholds: Optional[Dict[str, float]] = None,
    postprocess_cfg: Optional[PostprocessConfig] = None,
    use_postprocess: bool = True,
    target_shape: Sequence[int] | None = None,
) -> np.ndarray:
    prob = logits_to_foreground_probability(logits, target_shape=target_shape)
    return probability_to_binary_mask(prob, organ_name, domain_thresholds, postprocess_cfg, use_postprocess)


def get_task_tokens(
    model: torch.nn.Module,
    prototype_bank: Optional[EMATaskMemoryBank],
    text_bank: Dict[str, torch.Tensor],
    organ_name: str,
    device: torch.device,
    batch_size: int = 1,
) -> torch.Tensor | None:
    use_task_conditioned_decoder = bool(
        getattr(model, "use_task_conditioned_decoder", True)
    )
    use_task_specific_representation = bool(
        getattr(model, "use_task_specific_representation", True)
    )
    use_visual_prototype = bool(getattr(model, "use_visual_prototype", True))
    use_semantic_text = bool(getattr(model, "use_semantic_text", True))

    if not use_task_conditioned_decoder:
        return None
    if not use_task_specific_representation:
        return model.shared_task_tokens(batch_size, device=device)

    if use_visual_prototype:
        if prototype_bank is None or not prototype_bank.has(organ_name):
            raise ValueError(f"No training-derived task prototype found for domain '{organ_name}'.")
        stored = prototype_bank.get(organ_name, device=device)
        if stored is None:
            raise ValueError(f"Task prototype for domain '{organ_name}' is empty.")

        # Backward compatibility: an old checkpoint may store complete task tokens.
        if stored.shape[1] > model.visual_token_count:
            return stored.expand(batch_size, -1, -1) if stored.shape[0] == 1 and batch_size > 1 else stored
        visual = stored.expand(batch_size, -1, -1) if stored.shape[0] == 1 and batch_size > 1 else stored
    else:
        visual = model.null_visual_tokens(batch_size, device=device)

    organ_text = text_bank.get(organ_name) if use_semantic_text else None
    ftext = make_Ftext_batch(organ_text, batch_size) if organ_text is not None else None
    domain_ids = model.domain_ids_from_names(
        [organ_name] * batch_size,
        device=device,
    )
    return model.compose_task_tokens(
        visual,
        ftext,
        domain_ids=domain_ids,
        output_batch_size=batch_size,
    )


# Legacy name retained for imports from old scripts. It only works for old full-token banks.
def get_ema_task_tokens(ema_memory_bank, organ_name, device):
    if ema_memory_bank is None or not ema_memory_bank.has(organ_name):
        raise ValueError(f"No EMA task prototype found for domain '{organ_name}'.")
    return ema_memory_bank.get(organ_name, device=device)


def evaluate(
    model: torch.nn.Module,
    query_dataset: MultiOrganDataset,
    device: torch.device,
    text_bank: Dict[str, torch.Tensor],
    ema_memory_bank: Optional[EMATaskMemoryBank] = None,
    domain_thresholds: Optional[Dict[str, float]] = None,
    postprocess_cfg: Optional[PostprocessConfig] = None,
    use_postprocess: bool = False,
    show_progress: bool = False,
    metric_size: Sequence[int] = DEFAULT_METRIC_SIZE,
) -> Dict[str, Any]:
    """Evaluate every case on one shared spatial grid, default 512x512."""
    model.eval()
    metric_shape = normalize_metric_size(metric_size)
    metrics: DefaultDict[str, DefaultDict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    thresholds = dict(DEFAULT_DOMAIN_THRESHOLDS if domain_thresholds is None else domain_thresholds)

    with torch.inference_mode():
        indices = range(len(query_dataset))
        if show_progress:
            indices = tqdm(indices, desc="Validating", leave=False)
        for idx in indices:
            img, resized_label, organ = query_dataset[idx]
            xq = img.unsqueeze(0).to(device)
            target = resize_binary_mask(resized_label.numpy(), metric_shape)
            query_pyramid = model.encode_query(xq)
            task_tokens = get_task_tokens(model, ema_memory_bank, text_bank, organ, device)
            out = model.segment_with_task(
                query_feats=query_pyramid,
                task_tokens=task_tokens,
                output_size=xq.shape[-2:],
            )
            prob = logits_to_foreground_probability(out["pred_masks"], target_shape=metric_shape)
            pred = probability_to_binary_mask(
                prob,
                organ,
                thresholds,
                postprocess_cfg,
                use_postprocess,
            )
            metrics[organ]["dice"].append(binary_dice(pred, target))
            metrics[organ]["iou"].append(binary_iou(pred, target))
            metrics[organ]["hd95"].append(binary_hd95(pred, target))

    per_domain = {
        domain: {name: float(np.mean(values)) for name, values in domain_values.items()}
        for domain, domain_values in metrics.items()
    }
    if not per_domain:
        raise RuntimeError("No validation results were produced.")
    counts = {d: len(metrics[d]["dice"]) for d in per_domain}
    total = sum(counts.values())
    return {
        "dice": float(np.mean([x["dice"] for x in per_domain.values()])),
        "iou": float(np.mean([x["iou"] for x in per_domain.values()])),
        "hd95": float(np.mean([x["hd95"] for x in per_domain.values()])),
        "weighted_dice": float(sum(per_domain[d]["dice"] * counts[d] for d in per_domain) / max(total, 1)),
        "weighted_iou": float(sum(per_domain[d]["iou"] * counts[d] for d in per_domain) / max(total, 1)),
        "weighted_hd95": float(sum(per_domain[d]["hd95"] * counts[d] for d in per_domain) / max(total, 1)),
        "per_domain": per_domain,
        "counts": counts,
        "metric_size": metric_shape,
        "use_postprocess": bool(use_postprocess),
    }

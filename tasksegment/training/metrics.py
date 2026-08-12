from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt


def _max_possible_hd_from_shape(shape, spacing: Sequence[float] | None = None) -> float:
    if len(shape) < 2:
        return 0.0
    h, w = int(shape[-2]), int(shape[-1])
    sy, sx = (1.0, 1.0) if spacing is None else (float(spacing[-2]), float(spacing[-1]))
    return float(np.hypot(h * sy, w * sx))


def _binary_surface(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(bool)
    if not mask.any():
        return mask
    eroded = binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
    surface = mask ^ eroded
    return surface if surface.any() else mask


def _surface_distances(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: Sequence[float] | None = None,
) -> np.ndarray:
    pred_surface = _binary_surface(pred)
    target_surface = _binary_surface(target)
    dist_to_target = distance_transform_edt(~target_surface, sampling=spacing)
    dist_to_pred = distance_transform_edt(~pred_surface, sampling=spacing)
    return np.concatenate([dist_to_target[pred_surface], dist_to_pred[target_surface]]).astype(np.float32)


def binary_hd95(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: Sequence[float] | None = None,
) -> float:
    pred = (pred > 0).astype(np.uint8)
    target = (target > 0).astype(np.uint8)
    if pred.sum() == 0 and target.sum() == 0:
        return 0.0
    if pred.sum() == 0 or target.sum() == 0:
        return _max_possible_hd_from_shape(target.shape, spacing)
    distances = _surface_distances(pred, target, spacing)
    return 0.0 if distances.size == 0 else float(np.percentile(distances, 95))


def compute_hausdorff_distance95(logits, target) -> float:
    pred_labels = torch.softmax(logits, dim=1).argmax(dim=1).detach().cpu().numpy()
    target_labels = target.detach().cpu().numpy()
    return float(np.mean([binary_hd95(p, t) for p, t in zip(pred_labels, target_labels)]))


def _per_sample_overlap(logits: torch.Tensor, target: torch.Tensor):
    pred_fg = torch.softmax(logits, dim=1).argmax(dim=1).eq(1)
    target_fg = target.eq(1)
    intersection = (pred_fg & target_fg).sum(dim=(1, 2)).float()
    pred_area = pred_fg.sum(dim=(1, 2)).float()
    target_area = target_fg.sum(dim=(1, 2)).float()
    return intersection, pred_area, target_area


def compute_per_sample_hard_dice(logits, target, eps: float = 1e-8):
    inter, pred_area, target_area = _per_sample_overlap(logits, target)
    denom = pred_area + target_area
    return torch.where(denom > 0, (2.0 * inter) / denom.clamp_min(eps), torch.ones_like(denom))


def compute_hard_dice(logits, target, eps: float = 1e-8):
    return compute_per_sample_hard_dice(logits, target, eps).mean()


def compute_iou(logits, target, eps: float = 1e-8):
    inter, pred_area, target_area = _per_sample_overlap(logits, target)
    union = pred_area + target_area - inter
    return torch.where(union > 0, inter / union.clamp_min(eps), torch.ones_like(union)).mean()


def binary_dice(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    pred = (pred > 0).astype(bool)
    target = (target > 0).astype(bool)
    denom = float(pred.sum() + target.sum())
    if denom == 0.0:
        return 1.0
    return float(2.0 * np.logical_and(pred, target).sum() / max(denom, eps))


def binary_iou(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    pred = (pred > 0).astype(bool)
    target = (target > 0).astype(bool)
    union = float(np.logical_or(pred, target).sum())
    if union == 0.0:
        return 1.0
    return float(np.logical_and(pred, target).sum() / max(union, eps))


def binary_hd(pred: np.ndarray, target: np.ndarray, spacing: Sequence[float] | None = None) -> float:
    pred = (pred > 0).astype(np.uint8)
    target = (target > 0).astype(np.uint8)
    if pred.sum() == 0 and target.sum() == 0:
        return 0.0
    if pred.sum() == 0 or target.sum() == 0:
        return _max_possible_hd_from_shape(target.shape, spacing)
    distances = _surface_distances(pred, target, spacing)
    return float(distances.max()) if distances.size else 0.0


def compute_hausdorff_distance(logits, target):
    pred_labels = torch.softmax(logits, dim=1).argmax(dim=1).detach().cpu().numpy()
    target_labels = target.detach().cpu().numpy()
    return float(np.mean([binary_hd(p, t) for p, t in zip(pred_labels, target_labels)]))

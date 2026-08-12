from __future__ import annotations

from typing import Dict, Mapping, Sequence

import torch
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)[:, 1]
    target_fg = (target == 1).float()
    intersection = (probs * target_fg).sum(dim=(1, 2))
    denominator = probs.sum(dim=(1, 2)) + target_fg.sum(dim=(1, 2))
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


# Legacy import name kept for external scripts.
dice_loss = soft_dice_loss


def focal_tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.30,
    beta: float = 0.70,
    gamma: float = 0.75,
    eps: float = 1.0,
) -> torch.Tensor:
    """Focal Tversky loss for binary segmentation.

    beta > alpha penalizes false negatives more strongly, which is generally
    useful for lesion recall and Dice on small/low-contrast targets.
    """
    prob = torch.softmax(logits, dim=1)[:, 1]
    tgt = (target == 1).float()
    tp = (prob * tgt).sum(dim=(1, 2))
    fp = (prob * (1.0 - tgt)).sum(dim=(1, 2))
    fn = ((1.0 - prob) * tgt).sum(dim=(1, 2))
    tversky = (tp + eps) / (tp + float(alpha) * fp + float(beta) * fn + eps)
    return torch.pow(1.0 - tversky, float(gamma)).mean()


def segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    cfg: Mapping[str, float],
) -> torch.Tensor:
    ce_w = float(cfg.get("ce", 0.0))
    dice_w = float(cfg.get("dice", 1.0))
    tv_w = float(cfg.get("focal_tversky", 0.0))

    total = logits.new_zeros(())
    if ce_w > 0.0:
        total = total + ce_w * F.cross_entropy(logits, target.long())
    if dice_w > 0.0:
        total = total + dice_w * soft_dice_loss(logits, target)
    if tv_w > 0.0:
        total = total + tv_w * focal_tversky_loss(
            logits,
            target,
            alpha=float(cfg.get("tversky_alpha", 0.30)),
            beta=float(cfg.get("tversky_beta", 0.70)),
            gamma=float(cfg.get("tversky_gamma", 0.75)),
        )
    return total


def deep_supervision_loss(
    aux_logits: Sequence[torch.Tensor] | None,
    target: torch.Tensor,
    cfg: Mapping[str, float],
) -> torch.Tensor:
    if not aux_logits:
        return target.new_zeros((), dtype=torch.float32)
    weights = tuple(float(v) for v in cfg.get("aux_weights", ()))
    if len(weights) != len(aux_logits):
        raise ValueError(
            f"aux_weights length {len(weights)} does not match aux outputs {len(aux_logits)}"
        )

    total = aux_logits[0].new_zeros(())
    # Auxiliary heads use the same Dice-oriented objective, but omit Focal
    # Tversky to keep their gradients smooth at coarse resolutions.
    aux_cfg = dict(cfg)
    aux_cfg["focal_tversky"] = 0.0
    for weight, logits in zip(weights, aux_logits):
        if weight <= 0.0:
            continue
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
        total = total + weight * segmentation_loss(logits, target, aux_cfg)
    return total


def _soft_dice_from_probs(prob: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    prob = prob.float()
    target = target.float()
    intersection = (prob * target).sum(dim=(1, 2, 3))
    denominator = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1.0 - ((2.0 * intersection + eps) / (denominator + eps)).mean()


def dense_guidance_loss(
    output_dict: Dict[str, torch.Tensor],
    target_mask: torch.Tensor,
) -> torch.Tensor:
    pred = output_dict.get("dense_guidance_logits_lowres")
    if pred is None:
        return target_mask.new_zeros((), dtype=torch.float32)
    if target_mask.ndim == 3:
        target_mask = target_mask.unsqueeze(1)
    target = (target_mask > 0).float()
    if pred.shape[-2:] != target.shape[-2:]:
        target = F.interpolate(target, size=pred.shape[-2:], mode="nearest")
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dice = _soft_dice_from_probs(torch.sigmoid(pred), target)
    return bce + dice


# Backward-compatible wrapper used by older imports. Weighting is intentionally
# no longer applied inside this function; the trainer applies one explicit
# ``loss_cfg['dense']`` coefficient.
def compute_dense_guidance_supervision(
    output_dict: Dict[str, torch.Tensor],
    target_mask: torch.Tensor,
    loss_cfg: Dict[str, float] | None = None,
) -> torch.Tensor:
    del loss_cfg
    return dense_guidance_loss(output_dict, target_mask)

from __future__ import annotations

import os
from typing import Dict, Optional, Union

import torch

from ..configs import DOMAIN_NAMES, TEXT_GROUP_NAMES


EXPECTED_SUMMARY_TOKENS = len(TEXT_GROUP_NAMES)


def _safe_torch_load(path: str, map_location: Union[str, torch.device] = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _extract_features(
    obj: object,
    path: str,
    expected_text_dim: Optional[int],
    strict_dim: bool,
) -> torch.Tensor:
    if not isinstance(obj, dict) or "text_features" not in obj:
        raise ValueError(f"Invalid text feature schema: {path}")
    feats = obj["text_features"]
    if not torch.is_tensor(feats):
        raise TypeError(f"text_features is not a tensor: {path}")
    feats = feats.float()
    if feats.ndim == 1:
        feats = feats.unsqueeze(0)
    if feats.ndim == 3:
        feats = feats.reshape(-1, feats.shape[-1])
    if feats.ndim != 2:
        raise ValueError(f"Expected [N,D] text features, got {tuple(feats.shape)} in {path}")
    if feats.shape[0] < EXPECTED_SUMMARY_TOKENS:
        raise ValueError(
            f"Requires {EXPECTED_SUMMARY_TOKENS} summary tokens, got {feats.shape[0]} in {path}"
        )
    feats = feats[:EXPECTED_SUMMARY_TOKENS].contiguous()
    if not torch.isfinite(feats).all():
        raise ValueError(f"Non-finite text features found in {path}")
    if expected_text_dim is not None and feats.shape[-1] != int(expected_text_dim):
        message = (
            f"Text dimension mismatch: got={feats.shape[-1]}, "
            f"expected={int(expected_text_dim)} in {path}"
        )
        if strict_dim:
            raise ValueError(message)
        print(f"[warning] {message}")
        return torch.empty(0)
    return feats


def load_text_bank(
    text_dir: str,
    device: torch.device,
    expected_text_dim: Optional[int] = None,
    strict_dim: bool = True,
    center_across_domains: bool = True,
) -> Dict[str, torch.Tensor]:
    """Load five summary tokens per domain.

    ``center_across_domains`` removes one global common direction estimated from
    all available domain/group tokens. This is a deterministic preprocessing
    step applied identically during training, validation, calibration and test.
    It does not force domains to be orthogonal; it only reduces the dominant
    shared BioBERT direction that otherwise produces cosine similarities near 1.
    """

    mapping = {
        "thyroid": "text_features_thyroid.pt",
        "TN3K": "text_features_TN3K.pt",
        "BUSI_WHU": "text_features_BUSI_WHU.pt",
        "BUS-BRA": "text_features_BUS-BRA.pt",
        "OTU": "text_features_OTU.pt",
        "prostate": "text_features_prostate.pt",
    }
    if not os.path.exists(text_dir):
        return {}

    cpu_bank: Dict[str, torch.Tensor] = {}
    for domain in DOMAIN_NAMES:
        filename = mapping[domain]
        path = os.path.join(text_dir, filename)
        if not os.path.exists(path):
            continue
        feats = _extract_features(
            _safe_torch_load(path, map_location="cpu"),
            path,
            expected_text_dim,
            strict_dim,
        )
        if feats.numel() > 0:
            cpu_bank[domain] = feats

    missing = [domain for domain in DOMAIN_NAMES if domain not in cpu_bank]
    if missing:
        print(f"[warning] missing text features for: {missing}")

    if center_across_domains and cpu_bank:
        common_direction = torch.cat(list(cpu_bank.values()), dim=0).mean(dim=0, keepdim=True)
        cpu_bank = {
            domain: (feats - common_direction).contiguous()
            for domain, feats in cpu_bank.items()
        }
        print(
            f"[text] common-direction centering enabled over "
            f"{sum(x.shape[0] for x in cpu_bank.values())} tokens"
        )
    else:
        print("[text] common-direction centering disabled")

    return {domain: feats.to(device) for domain, feats in cpu_bank.items()}


def make_Ftext_batch(text_feats: torch.Tensor, batch_size: int) -> torch.Tensor:
    if text_feats.ndim != 2:
        raise ValueError(f"text_feats must be [N,D], got {tuple(text_feats.shape)}")
    return text_feats.unsqueeze(0).expand(int(batch_size), -1, -1)

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tasksegment.configs import DOMAIN_NAMES, TEXT_GROUP_NAMES


DOMAINS = tuple(DOMAIN_NAMES)
EXPECTED_GROUPS = tuple(TEXT_GROUP_NAMES)


def safe_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def offdiag_cosine_mean(feats: torch.Tensor) -> float:
    matrix = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
    n = feats.shape[0]
    denom = max(n * (n - 1), 1)
    return float((matrix.sum() - torch.diagonal(matrix).sum()).item() / denom)


def pairwise_domain_cosine(bank: dict[str, torch.Tensor]) -> dict[str, float]:
    result: dict[str, float] = {}
    for index, first in enumerate(DOMAINS):
        for second in DOMAINS[index + 1 :]:
            a = bank[first].mean(dim=0, keepdim=True)
            b = bank[second].mean(dim=0, keepdim=True)
            result[f"{first}__{second}"] = float(F.cosine_similarity(a, b).item())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit raw and common-direction-centered five-token text features"
    )
    parser.add_argument("--text-dir", default="./text_features_llm")
    parser.add_argument("--output", default="text_feature_audit.json")
    args = parser.parse_args()

    root = Path(args.text_dir)
    raw_bank: dict[str, torch.Tensor] = {}
    files: dict[str, dict] = {}
    exact_duplicate_pairs: list[list[str]] = []

    for domain in DOMAINS:
        path = root / f"text_features_{domain}.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        obj = safe_load(path)
        feats = obj.get("text_features") if isinstance(obj, dict) else None
        if not torch.is_tensor(feats):
            raise ValueError(f"invalid text feature file: {path}")
        feats = feats.float()
        if feats.ndim != 2 or feats.shape[0] < len(EXPECTED_GROUPS):
            raise ValueError(
                f"{path}: expected at least [{len(EXPECTED_GROUPS)},D], got {tuple(feats.shape)}"
            )
        feats = feats[: len(EXPECTED_GROUPS)].contiguous()
        if not torch.isfinite(feats).all():
            raise ValueError(f"{path}: contains NaN or Inf")
        raw_bank[domain] = feats

        groups = obj.get("group_names") if isinstance(obj, dict) else None
        files[domain] = {
            "path": str(path),
            "shape_used": list(feats.shape),
            "raw_within_domain_offdiag_cosine_mean": offdiag_cosine_mean(feats),
            "raw_mean_token_norm": float(feats.norm(dim=-1).mean().item()),
            "groups": groups,
            "expected_groups": list(EXPECTED_GROUPS),
            "group_order_matches": list(groups or []) == list(EXPECTED_GROUPS),
            "encoder": obj.get("encoder") if isinstance(obj, dict) else None,
            "schema_version": obj.get("schema_version") if isinstance(obj, dict) else None,
            "embedding_type": obj.get("embedding_type") if isinstance(obj, dict) else None,
        }

    for i, first in enumerate(DOMAINS):
        for second in DOMAINS[i + 1 :]:
            if torch.equal(raw_bank[first], raw_bank[second]):
                exact_duplicate_pairs.append([first, second])

    common_direction = torch.cat(list(raw_bank.values()), dim=0).mean(dim=0, keepdim=True)
    centered_bank = {
        domain: feats - common_direction
        for domain, feats in raw_bank.items()
    }
    for domain in DOMAINS:
        files[domain]["centered_within_domain_offdiag_cosine_mean"] = offdiag_cosine_mean(
            centered_bank[domain]
        )
        files[domain]["centered_mean_token_norm"] = float(
            centered_bank[domain].norm(dim=-1).mean().item()
        )

    raw_pairwise = pairwise_domain_cosine(raw_bank)
    centered_pairwise = pairwise_domain_cosine(centered_bank)
    report = {
        "files": files,
        "raw_pairwise_domain_mean_cosine": raw_pairwise,
        "centered_pairwise_domain_mean_cosine": centered_pairwise,
        "exact_duplicate_pairs": exact_duplicate_pairs,
        "centering": {
            "enabled_in_default_training": True,
            "definition": "subtract global mean across all available domain/group tokens",
            "common_direction_norm": float(common_direction.norm().item()),
        },
    }

    output = Path(args.output)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    for domain in DOMAINS:
        item = files[domain]
        print(
            f"{domain:10s}: shape={item['shape_used']} "
            f"raw_within={item['raw_within_domain_offdiag_cosine_mean']:.4f} "
            f"centered_within={item['centered_within_domain_offdiag_cosine_mean']:.4f}"
        )

    print("\nRaw cross-domain mean-token cosine:")
    for key, value in raw_pairwise.items():
        print(f"  {key:24s}: {value:.4f}")

    print("\nCentered cross-domain mean-token cosine used by default training:")
    for key, value in centered_pairwise.items():
        print(f"  {key:24s}: {value:.4f}")

    if exact_duplicate_pairs:
        print("WARNING exact duplicate domains:", exact_duplicate_pairs)
    if any(not item["group_order_matches"] for item in files.values()):
        print("WARNING: at least one file does not use the expected group order")
    print(f"report: {output.resolve()}")


if __name__ == "__main__":
    main()

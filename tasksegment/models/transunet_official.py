from __future__ import annotations

"""Adapter for the official Beckschen/TransUNet 2D implementation."""

import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_ROOT = ROOT / "third_party" / "TransUNet"

if not OFFICIAL_ROOT.exists():
    raise FileNotFoundError(
        f"Official TransUNet repository not found at {OFFICIAL_ROOT}. "
        "Run scripts/install_official_transunet.sh first."
    )

if str(OFFICIAL_ROOT) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_ROOT))

try:
    from networks.vit_seg_modeling import (
        CONFIGS as OFFICIAL_CONFIGS,
        VisionTransformer as OfficialVisionTransformer,
    )
except Exception as exc:
    raise ImportError(
        "Could not import official TransUNet. Verify "
        "third_party/TransUNet/networks/vit_seg_modeling.py and dependencies."
    ) from exc


class OfficialTransUNet2D(nn.Module):
    """TaskSegment wrapper around the official 2D TransUNet model."""

    def __init__(
        self,
        *,
        image_size: Tuple[int, int] = (512, 512),
        in_channels: int = 1,
        num_classes: int = 2,
        vit_name: str = "R50-ViT-B_16",
        vit_patch_size: int = 16,
        n_skip: int = 3,
        pretrained_checkpoint: str | None = None,
        zero_head: bool = False,
        vis: bool = False,
    ) -> None:
        super().__init__()
        image_size = tuple(int(v) for v in image_size)
        if image_size[0] != image_size[1]:
            raise ValueError("Official 2D TransUNet requires square input.")
        if vit_name not in OFFICIAL_CONFIGS:
            raise ValueError(
                f"Unknown official configuration {vit_name!r}; "
                f"available: {sorted(OFFICIAL_CONFIGS)}"
            )
        if image_size[0] % vit_patch_size != 0:
            raise ValueError("vit_patch_size must divide image_size.")
        if in_channels != 1:
            raise ValueError(
                "This TaskSegment adapter expects one-channel ultrasound input. "
                "The official forward method repeats it to three channels."
            )

        # Copy because the official CONFIGS dictionary stores mutable objects.
        config = OFFICIAL_CONFIGS[vit_name].copy_and_resolve_references()
        config.n_classes = int(num_classes)
        config.n_skip = int(n_skip)

        # This follows the official train.py rule exactly. At 512 and patch 16,
        # the hybrid R50 model produces a 32x32 Transformer token grid.
        if "R50" in vit_name:
            grid = int(image_size[0] // vit_patch_size)
            config.patches.grid = (grid, grid)

        self.model = OfficialVisionTransformer(
            config,
            img_size=int(image_size[0]),
            num_classes=int(num_classes),
            zero_head=bool(zero_head),
            vis=bool(vis),
        )

        self.pretrained_sha256: str | None = None
        self.pretraining_source: str | None = None
        if pretrained_checkpoint:
            checkpoint = Path(pretrained_checkpoint).expanduser().resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"Official NumPy checkpoint not found: {checkpoint}"
                )
            if checkpoint.suffix.lower() != ".npz":
                raise ValueError(
                    "TransUNet pretraining requires the official .npz file; "
                    f"got: {checkpoint}"
                )
            required_keys = {
                "embedding/kernel",
                "embedding/bias",
                "Transformer/encoder_norm/scale",
                "Transformer/posembed_input/pos_embedding",
                "conv_root/kernel",
                "gn_root/scale",
            }
            with np.load(str(checkpoint), allow_pickle=False) as weights:
                missing = sorted(required_keys - set(weights.files))
                if missing:
                    raise KeyError(
                        "The supplied file is not a complete R50+ViT-B_16 "
                        f"ImageNet-21k checkpoint. Missing keys: {missing}"
                    )
                self.model.load_from(weights=weights)
            digest = hashlib.sha256()
            with checkpoint.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            pretrained_checkpoint = str(checkpoint)
            self.pretrained_sha256 = digest.hexdigest()
            self.pretraining_source = "ImageNet-21k R50+ViT-B_16.npz"
            print(
                "[pretrained] loaded TransUNet ImageNet-21k weights "
                f"from {checkpoint} sha256={self.pretrained_sha256}"
            )

        self.image_size = image_size
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.vit_name = str(vit_name)
        self.vit_patch_size = int(vit_patch_size)
        self.n_skip = int(n_skip)
        self.pretrained_checkpoint = pretrained_checkpoint
        self.zero_head = bool(zero_head)
        self.vis = bool(vis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"Expected Bx1xHxW input, got {tuple(x.shape)}")
        if tuple(x.shape[-2:]) != self.image_size:
            raise ValueError(
                f"Expected {self.image_size}, got {tuple(x.shape[-2:])}"
            )
        return self.model(x)

    def export_config(self) -> Dict[str, Any]:
        return {
            "implementation": "official Beckschen/TransUNet",
            "official_repository_path": str(OFFICIAL_ROOT),
            "image_size": self.image_size,
            "in_channels": self.in_channels,
            "official_internal_channels": 3,
            "num_classes": self.num_classes,
            "vit_name": self.vit_name,
            "vit_patch_size": self.vit_patch_size,
            "token_grid": (
                self.image_size[0] // self.vit_patch_size,
                self.image_size[1] // self.vit_patch_size,
            ),
            "n_skip": self.n_skip,
            "pretrained_checkpoint": self.pretrained_checkpoint,
            "pretrained_sha256": self.pretrained_sha256,
            "pretraining_source": self.pretraining_source,
            "zero_head": self.zero_head,
            "vis": self.vis,
        }

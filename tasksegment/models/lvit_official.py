from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class OfficialLViTAdapter(nn.Module):
    """Auditable adapter around the official HUANGLIZI/LViT architecture.

    The official implementation applies a Sigmoid before returning its binary
    output. For numerically stable mixed-precision training, this adapter
    replaces only that terminal Sigmoid with Identity and exposes native logits.
    The architecture and all trainable layers remain unchanged.
    """

    def __init__(
        self,
        official_repository_path: str = "third_party/LViT",
        model_resolution: int = 512,
        output_resolution: Tuple[int, int] = (512, 512),
        binary_output: str | None = None,
    ) -> None:
        super().__init__()
        self.official_repository_path = str(official_repository_path)
        self.model_resolution = int(model_resolution)
        self.output_resolution = tuple(int(v) for v in output_resolution)
        self.binary_output = (
            "native_pre_sigmoid_logit"
            if binary_output is None
            else str(binary_output)
        )
        if self.binary_output != "native_pre_sigmoid_logit":
            raise ValueError(
                "Unsupported LViT binary_output policy: "
                f"{self.binary_output!r}"
            )

        if self.model_resolution <= 0 or self.model_resolution % 16 != 0:
            raise ValueError("model_resolution must be positive and divisible by 16")
        if len(self.output_resolution) != 2:
            raise ValueError("output_resolution must be H,W")

        repo = Path(self.official_repository_path).expanduser().resolve()
        if not (repo / "nets" / "LViT.py").is_file():
            raise FileNotFoundError(
                f"Official LViT repository not found at {repo}. "
                "Run scripts/install_official_lvit.sh first."
            )
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        module = importlib.import_module("nets.LViT")
        config = SimpleNamespace(base_channel=64)
        self.network = module.LViT(
            config,
            n_channels=3,
            n_classes=1,
            img_size=self.model_resolution,
            vis=False,
        )

        expected_patches = (self.model_resolution // 16) ** 2
        actual_patches = int(self.network.downVit.embeddings.position_embeddings.shape[1])
        text_projection = int(self.network.downVit.CTBN3.conv.out_channels)
        if actual_patches != expected_patches or text_projection != expected_patches:
            raise RuntimeError(
                "The official repository has not received the TaskSegment "
                "dynamic-resolution patch. Run scripts/patch_lvit_dynamic_resolution.py. "
                f"expected={expected_patches}, position={actual_patches}, "
                f"text_projection={text_projection}"
            )

        # Official LViT uses Sigmoid for n_classes == 1. Replacing only this
        # terminal activation avoids the unstable FP16 probability->logit
        # inversion while preserving the model architecture.
        self.network.last_activation = nn.Identity()

    def export_config(self) -> dict:
        return {
            "official_repository_path": self.official_repository_path,
            "model_resolution": self.model_resolution,
            "output_resolution": self.output_resolution,
            "binary_output": self.binary_output,
        }

    def forward_raw_logit(
        self,
        images: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError(f"Expected Bx1xHxW grayscale input, got {tuple(images.shape)}")
        if text_embeddings.ndim != 3 or text_embeddings.shape[1:] != (10, 768):
            raise ValueError(
                "Official LViT requires Bx10x768 text embeddings; "
                f"got {tuple(text_embeddings.shape)}"
            )

        x = images.float().clamp(0.0, 1.0)
        if x.shape[-2:] != (self.model_resolution, self.model_resolution):
            x = F.interpolate(
                x,
                size=(self.model_resolution, self.model_resolution),
                mode="bilinear",
                align_corners=False,
            )
        x = x.repeat(1, 3, 1, 1)
        raw_logit = self.network(x, text_embeddings)
        if raw_logit.ndim != 4 or raw_logit.shape[1] != 1:
            raise ValueError(f"Unexpected official LViT output: {tuple(raw_logit.shape)}")

        # Losses and interpolation are kept in FP32 even when the backbone uses AMP.
        raw_logit = raw_logit.float()
        if raw_logit.shape[-2:] != self.output_resolution:
            raw_logit = F.interpolate(
                raw_logit,
                size=self.output_resolution,
                mode="bilinear",
                align_corners=False,
            )
        return raw_logit[:, 0]

    def forward_probability(
        self,
        images: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(self.forward_raw_logit(images, text_embeddings))

    def forward(self, images: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        positive_logit = self.forward_raw_logit(images, text_embeddings)
        return torch.stack([torch.zeros_like(positive_logit), positive_logit], dim=1)

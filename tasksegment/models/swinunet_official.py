from __future__ import annotations

"""Adapter for the official HuCaoFighting/Swin-Unet implementation.

The official repository must be installed at:

    third_party/Swin-Unet/

Only the model is imported from the official repository. Dataset loading,
sampling, augmentation, loss, validation metrics, checkpoint selection, and
early stopping remain in the TaskSegment comparison pipeline.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Sequence, Tuple

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_ROOT = ROOT / "third_party" / "Swin-Unet"

if not OFFICIAL_ROOT.exists():
    raise FileNotFoundError(
        "Official Swin-Unet repository not found at "
        f"{OFFICIAL_ROOT}. Run scripts/install_official_swinunet.sh first."
    )

if str(OFFICIAL_ROOT) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_ROOT))

try:
    from networks.vision_transformer import SwinUnet as OfficialSwinUnet
except Exception as exc:
    raise ImportError(
        "Failed to import the official Swin-Unet code. Confirm that "
        "third_party/Swin-Unet/networks/vision_transformer.py exists and that "
        "the official dependencies are installed."
    ) from exc


def _ns(**kwargs):
    return SimpleNamespace(**kwargs)


def build_official_config(
    *,
    image_size: Tuple[int, int],
    patch_size: int,
    in_channels: int,
    embed_dim: int,
    depths: Sequence[int],
    num_heads: Sequence[int],
    window_size: int,
    mlp_ratio: float,
    qkv_bias: bool,
    qk_scale: float | None,
    drop_rate: float,
    drop_path_rate: float,
    ape: bool,
    patch_norm: bool,
    use_checkpoint: bool,
    pretrained_checkpoint: str | None,
):
    if image_size[0] != image_size[1]:
        raise ValueError("The official Swin-Unet implementation requires square input.")
    return _ns(
        DATA=_ns(IMG_SIZE=int(image_size[0])),
        MODEL=_ns(
            PRETRAIN_CKPT=pretrained_checkpoint,
            DROP_RATE=float(drop_rate),
            DROP_PATH_RATE=float(drop_path_rate),
            SWIN=_ns(
                PATCH_SIZE=int(patch_size),
                IN_CHANS=int(in_channels),
                EMBED_DIM=int(embed_dim),
                DEPTHS=list(int(v) for v in depths),
                NUM_HEADS=list(int(v) for v in num_heads),
                WINDOW_SIZE=int(window_size),
                MLP_RATIO=float(mlp_ratio),
                QKV_BIAS=bool(qkv_bias),
                QK_SCALE=qk_scale,
                APE=bool(ape),
                PATCH_NORM=bool(patch_norm),
            ),
        ),
        TRAIN=_ns(USE_CHECKPOINT=bool(use_checkpoint)),
    )


class OfficialSwinUNet2D(nn.Module):
    """TaskSegment wrapper around the official Swin-Unet model class."""

    def __init__(
        self,
        *,
        image_size: Tuple[int, int] = (512, 512),
        in_channels: int = 1,
        num_classes: int = 2,
        patch_size: int = 4,
        embed_dim: int = 96,
        depths: Sequence[int] = (2, 2, 6, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        ape: bool = False,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        pretrained_checkpoint: str | None = None,
    ) -> None:
        super().__init__()
        image_size = tuple(int(v) for v in image_size)
        depths = tuple(int(v) for v in depths)
        num_heads = tuple(int(v) for v in num_heads)

        if image_size != (512, 512):
            raise ValueError("This comparison protocol requires image_size=(512, 512).")
        if len(depths) != 4 or len(num_heads) != 4:
            raise ValueError("depths and num_heads must contain four values.")
        if image_size[0] % patch_size != 0:
            raise ValueError("patch_size must divide image_size.")

        patch_grid = image_size[0] // patch_size
        for stage in range(4):
            resolution = patch_grid // (2 ** stage)
            effective_window = min(int(window_size), resolution)
            if resolution % effective_window != 0:
                raise ValueError(
                    f"Stage {stage + 1} resolution {resolution} is not divisible "
                    f"by window_size={window_size}. For 512x512 with patch_size=4, "
                    "use window_size=8."
                )

        config = build_official_config(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=3,  # official wrapper repeats grayscale input to RGB
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            ape=ape,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            pretrained_checkpoint=pretrained_checkpoint,
        )

        self.model = OfficialSwinUnet(
            config=config,
            img_size=image_size[0],
            num_classes=int(num_classes),
        )
        if pretrained_checkpoint is not None:
            checkpoint_path = Path(pretrained_checkpoint)
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Pretrained checkpoint not found: {checkpoint_path}"
                )
            self.model.load_from(config)

        self.image_size = image_size
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = int(window_size)
        self.mlp_ratio = float(mlp_ratio)
        self.qkv_bias = bool(qkv_bias)
        self.qk_scale = qk_scale
        self.drop_rate = float(drop_rate)
        self.drop_path_rate = float(drop_path_rate)
        self.ape = bool(ape)
        self.patch_norm = bool(patch_norm)
        self.use_checkpoint = bool(use_checkpoint)
        self.pretrained_checkpoint = pretrained_checkpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected BCHW input with {self.in_channels} channel, got "
                f"{tuple(x.shape)}."
            )
        if tuple(x.shape[-2:]) != self.image_size:
            raise ValueError(
                f"Expected spatial size {self.image_size}, got "
                f"{tuple(x.shape[-2:])}."
            )
        # Official SwinUnet.forward repeats a one-channel tensor to three channels.
        return self.model(x)

    def export_config(self) -> Dict[str, Any]:
        return {
            "implementation": "HuCaoFighting/Swin-Unet official repository",
            "official_repository_path": str(OFFICIAL_ROOT),
            "image_size": self.image_size,
            "in_channels": self.in_channels,
            "official_internal_in_channels": 3,
            "num_classes": self.num_classes,
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "depths": self.depths,
            "num_heads": self.num_heads,
            "window_size": self.window_size,
            "mlp_ratio": self.mlp_ratio,
            "qkv_bias": self.qkv_bias,
            "qk_scale": self.qk_scale,
            "drop_rate": self.drop_rate,
            "drop_path_rate": self.drop_path_rate,
            "ape": self.ape,
            "patch_norm": self.patch_norm,
            "use_checkpoint": self.use_checkpoint,
            "pretrained_checkpoint": self.pretrained_checkpoint,
        }

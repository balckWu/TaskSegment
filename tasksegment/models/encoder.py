from __future__ import annotations

import torch
import torch.nn as nn

from .layers import UNetConv2


class UNetEncoder2D(nn.Module):
    """2D U-Net encoder exposing bottleneck and multi-scale skip features."""

    def __init__(
        self,
        feature_scale: int = 1,
        in_channels: int = 1,
        norm_type: str = "group",
        group_norm_groups: int = 16,
        is_batchnorm: bool | None = None,
    ) -> None:
        super().__init__()
        # Old checkpoints used a boolean. Preserve their architecture when this
        # legacy argument is explicitly supplied.
        if is_batchnorm is not None:
            norm_type = "batch" if bool(is_batchnorm) else "none"

        self.in_channels = int(in_channels)
        self.feature_scale = int(feature_scale)
        self.norm_type = str(norm_type)
        self.group_norm_groups = int(group_norm_groups)

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]
        self.filters = filters

        self.pre_conv = nn.Conv2d(self.in_channels, filters[0], kernel_size=3, padding=1)

        self.conv1 = UNetConv2(filters[0], filters[0], self.norm_type, self.group_norm_groups)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = UNetConv2(filters[0], filters[1], self.norm_type, self.group_norm_groups)
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = UNetConv2(filters[1], filters[2], self.norm_type, self.group_norm_groups)
        self.maxpool3 = nn.MaxPool2d(2)
        self.conv4 = UNetConv2(filters[2], filters[3], self.norm_type, self.group_norm_groups)
        self.maxpool4 = nn.MaxPool2d(2)
        self.center = UNetConv2(filters[3], filters[4], self.norm_type, self.group_norm_groups)

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.pre_conv(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.maxpool1(conv1))
        conv3 = self.conv3(self.maxpool2(conv2))
        conv4 = self.conv4(self.maxpool3(conv3))
        bottleneck = self.center(self.maxpool4(conv4))
        return {
            "skip1": conv1,
            "skip2": conv2,
            "skip3": conv3,
            "skip4": conv4,
            "bottleneck": bottleneck,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)["bottleneck"]

from __future__ import annotations

import torch
import torch.nn as nn


def build_2d_norm(num_channels: int, norm_type: str = "group", group_norm_groups: int = 16) -> nn.Module:
    norm_type = str(norm_type).lower()
    if norm_type in {"none", "identity", "off"}:
        return nn.Identity()
    if norm_type == "batch":
        return nn.BatchNorm2d(num_channels)
    if norm_type == "instance":
        return nn.InstanceNorm2d(num_channels, affine=True)
    if norm_type == "group":
        groups = min(int(group_norm_groups), int(num_channels))
        while groups > 1 and num_channels % groups:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)
    raise ValueError(f"Unsupported norm_type: {norm_type}")


def _conv_norm_act(in_channels: int, out_channels: int, norm_type: str, groups: int) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Conv2d(in_channels, out_channels, 3, padding=1)]
    if str(norm_type).lower() not in {"none", "identity", "off"}:
        layers.append(build_2d_norm(out_channels, norm_type, groups))
    # Keep ReLU to preserve the exact old architecture for migrated checkpoints.
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class UNetConv2(nn.Module):
    """Two U-Net convolutions; parameter names remain old-checkpoint compatible."""

    def __init__(self, in_size: int, out_size: int, norm_type: str = "group", group_norm_groups: int = 16) -> None:
        super().__init__()
        self.conv1 = _conv_norm_act(in_size, out_size, norm_type, group_norm_groups)
        self.conv2 = _conv_norm_act(out_size, out_size, norm_type, group_norm_groups)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(inputs))


unetConv2 = UNetConv2

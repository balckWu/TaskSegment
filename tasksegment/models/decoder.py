from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dense_guidance import ConvNormAct, TextInducedDenseGuidance2D
from .layers import build_2d_norm


class SkipFusionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        task_dim: int,
        norm_type: str = "group",
        group_norm_groups: int = 16,
        skip_gate_mode: str = "centered",
        guidance_modulation_mode: str = "centered",
    ) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            build_2d_norm(out_channels, norm_type, group_norm_groups),
            nn.GELU(),
        )
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_channels, out_channels, kernel_size=1, bias=False),
            build_2d_norm(out_channels, norm_type, group_norm_groups),
        )
        self.skip_gate = nn.Sequential(nn.Linear(task_dim, out_channels), nn.Sigmoid())
        self.skip_gate_mode = str(skip_gate_mode)
        self.guidance_modulation_mode = str(guidance_modulation_mode)
        self.fuse = ConvNormAct(
            out_channels * 2,
            out_channels,
            norm_type=norm_type,
            group_norm_groups=group_norm_groups,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        task_summary: torch.Tensor | None,
        spatial_guidance: torch.Tensor | None = None,
        spatial_strength: float = 0.0,
        use_task_gate: bool = True,
    ) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        skip = self.skip_proj(skip)
        if use_task_gate:
            if task_summary is None:
                raise ValueError("task_summary is required when task gating is enabled")
            gate = self.skip_gate(task_summary).unsqueeze(-1).unsqueeze(-1)
            if self.skip_gate_mode == "legacy":
                skip = skip * (1.0 + gate)
            else:
                # Identity-centred channel gating: [0.75, 1.25].
                skip = skip * (0.75 + 0.5 * gate)

        if spatial_guidance is not None and spatial_strength > 0.0:
            if spatial_guidance.shape[-2:] != skip.shape[-2:]:
                spatial_guidance = F.interpolate(
                    spatial_guidance,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            guidance = spatial_guidance.clamp(0.0, 1.0)
            if self.guidance_modulation_mode == "legacy":
                modulation = 1.0 + float(spatial_strength) * guidance
            else:
                modulation = (1.0 + float(spatial_strength) * (2.0 * guidance - 1.0)).clamp(0.5, 1.5)
            skip = skip * modulation
            x = x * modulation

        return self.fuse(torch.cat([x, skip], dim=1))


class IrisMaskDecoder2D(nn.Module):
    """Task-conditioned U-Net decoder optimized for binary Dice.

    Multi-scale auxiliary heads provide direct supervision to intermediate
    decoder stages. They are used only during training and add negligible
    inference cost.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_heads: int = 8,
        dropout: float = 0.1,
        fg_token_weight: float = 0.30,
        skip_gate_mode: str = "centered",
        num_query_tokens: int = 4,
        use_ftext: bool = True,
        use_task_conditioning: bool = True,
        norm_type: str = "group",
        group_norm_groups: int = 16,
        encoder_filters=(64, 128, 256, 512, 1024),
        use_dense_text_guidance: bool = True,
        dense_guidance_dim: int = 256,
        dense_guidance_strength: float = 0.50,
        guidance_dropout_prob: float = 0.10,
        guidance_scale_min: float = 0.80,
        guidance_scale_max: float = 1.20,
        guidance_clamp_max: float = 0.90,
        text_group_summary_count: int = 5,
        guidance_stage_weights: Sequence[float] = (1.0, 0.50, 0.25, 0.0, 0.0),
        guidance_modulation_mode: str = "centered",
        use_deep_supervision: bool = True,
    ) -> None:
        super().__init__()
        if len(tuple(guidance_stage_weights)) != 5:
            raise ValueError("guidance_stage_weights must contain bottleneck, up4, up3, up2, up1")

        self.num_query_tokens = int(num_query_tokens)
        self.use_ftext = bool(use_ftext)
        self.use_task_conditioning = bool(use_task_conditioning)
        self.dropout = nn.Dropout(dropout)
        self.use_dense_text_guidance = bool(use_dense_text_guidance and use_ftext)
        self.dense_guidance_strength = float(dense_guidance_strength)
        self.guidance_dropout_prob = float(guidance_dropout_prob)
        self.guidance_scale_min = float(guidance_scale_min)
        self.guidance_scale_max = float(guidance_scale_max)
        self.guidance_clamp_max = float(guidance_clamp_max)
        self.text_group_summary_count = int(text_group_summary_count)
        self.guidance_stage_weights = tuple(float(v) for v in guidance_stage_weights)
        self.guidance_modulation_mode = str(guidance_modulation_mode)
        self.use_deep_supervision = bool(use_deep_supervision)

        self.query_to_task = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.task_summary_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        f1, f2, f3, f4, f5 = encoder_filters
        self.bottleneck_proj = nn.Sequential(
            nn.Conv2d(f5, f5, kernel_size=3, padding=1, bias=False),
            build_2d_norm(f5, norm_type, group_norm_groups),
            nn.GELU(),
        )

        self.dense_guidance = None
        if self.use_dense_text_guidance:
            self.dense_guidance = TextInducedDenseGuidance2D(
                in_channels=f5,
                text_dim=hidden_dim,
                guidance_dim=dense_guidance_dim,
                text_group_summary_count=text_group_summary_count,
                norm_type=norm_type,
                group_norm_groups=group_norm_groups,
            )

        self.up4 = SkipFusionBlock(
            f5, f4, f4, hidden_dim, norm_type, group_norm_groups,
            skip_gate_mode, guidance_modulation_mode,
        )
        self.up3 = SkipFusionBlock(
            f4, f3, f3, hidden_dim, norm_type, group_norm_groups,
            skip_gate_mode, guidance_modulation_mode,
        )
        self.up2 = SkipFusionBlock(
            f3, f2, f2, hidden_dim, norm_type, group_norm_groups,
            skip_gate_mode, guidance_modulation_mode,
        )
        self.up1 = SkipFusionBlock(
            f2, f1, f1, hidden_dim, norm_type, group_norm_groups,
            skip_gate_mode, guidance_modulation_mode,
        )

        self.head = nn.Sequential(
            nn.Conv2d(f1, f1, kernel_size=3, padding=1, bias=False),
            build_2d_norm(f1, norm_type, group_norm_groups),
            nn.GELU(),
            nn.Conv2d(f1, 2, kernel_size=1),
        )
        if self.use_deep_supervision:
            self.aux4 = nn.Conv2d(f4, 2, kernel_size=1)
            self.aux3 = nn.Conv2d(f3, 2, kernel_size=1)
            self.aux2 = nn.Conv2d(f2, 2, kernel_size=1)

        self.fg_token_weight = float(fg_token_weight)

    def _split_task_tokens(self, task_tokens: torch.Tensor):
        foreground = task_tokens[:, 0]
        visual_end = 1 + self.num_query_tokens
        visual_context = task_tokens[:, 1:visual_end]
        text = task_tokens[:, visual_end:] if self.use_ftext and task_tokens.shape[1] > visual_end else None
        return foreground, visual_context, text

    @staticmethod
    def _unpack_query_feats(query_feats):
        if isinstance(query_feats, dict):
            return (
                query_feats["bottleneck"],
                query_feats.get("skip1"),
                query_feats.get("skip2"),
                query_feats.get("skip3"),
                query_feats.get("skip4"),
            )
        return query_feats, None, None, None, None

    def _summarize_task(self, task_tokens: torch.Tensor):
        foreground, visual_context, text = self._split_task_tokens(task_tokens)
        visual_summary = visual_context.mean(dim=1) if visual_context.shape[1] else foreground
        fused = self.task_summary_proj(torch.cat([foreground, visual_summary], dim=1))
        task_summary = self.fg_token_weight * foreground + (1.0 - self.fg_token_weight) * fused
        return task_summary, text

    def _sample_guidance_strength(self, device: torch.device) -> float:
        strength = float(self.dense_guidance_strength)
        if not self.training:
            return strength
        if torch.rand((), device=device).item() < self.guidance_dropout_prob:
            return 0.0
        lo = min(self.guidance_scale_min, self.guidance_scale_max)
        hi = max(self.guidance_scale_min, self.guidance_scale_max)
        return strength * float(torch.empty((), device=device).uniform_(lo, hi).item())

    def forward(self, query_feats, task_tokens: torch.Tensor | None = None):
        bottleneck, skip1, skip2, skip3, skip4 = self._unpack_query_feats(query_feats)

        dense_logits = None
        guidance_prob = None
        task_summary = None
        strength = 0.0

        if self.use_task_conditioning:
            if task_tokens is None:
                raise ValueError("task_tokens are required by the task-conditioned decoder")
            query_flat = bottleneck.flatten(2).transpose(1, 2)
            query_update, _ = self.query_to_task(
                query_flat,
                task_tokens,
                task_tokens,
                need_weights=False,
            )
            query_flat = self.query_norm(query_flat + self.dropout(query_update))
            x = self.bottleneck_proj(query_flat.transpose(1, 2).reshape_as(bottleneck))

            task_summary, text_tokens = self._summarize_task(task_tokens)
            strength = self._sample_guidance_strength(x.device)
            if self.dense_guidance is not None and text_tokens is not None and text_tokens.shape[1] > 0:
                dense_out = self.dense_guidance(x, text_tokens)
                dense_logits = dense_out["guidance_map"]
                guidance_prob = torch.sigmoid(dense_logits).clamp(0.0, self.guidance_clamp_max)
                w0 = strength * self.guidance_stage_weights[0]
                if w0 > 0.0:
                    if self.guidance_modulation_mode == "legacy":
                        x = x + w0 * dense_out["guidance_feats"]
                        x = x * (1.0 + w0 * guidance_prob)
                    else:
                        centered = 2.0 * guidance_prob - 1.0
                        x = x + w0 * torch.tanh(dense_out["guidance_feats"])
                        x = x * (1.0 + w0 * centered).clamp(0.5, 1.5)
        else:
            # TSV5 grouped ablation: preserve the same bottleneck projection and
            # U-Net decoding capacity while removing all task-token injection.
            x = self.bottleneck_proj(bottleneck)

        stage_strengths = [strength * v for v in self.guidance_stage_weights[1:]]
        use_task_gate = bool(self.use_task_conditioning)

        x4 = self.up4(x, skip4, task_summary, guidance_prob, stage_strengths[0], use_task_gate) if skip4 is not None else F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x3 = self.up3(x4, skip3, task_summary, guidance_prob, stage_strengths[1], use_task_gate) if skip3 is not None else F.interpolate(x4, scale_factor=2, mode="bilinear", align_corners=False)
        x2 = self.up2(x3, skip2, task_summary, guidance_prob, stage_strengths[2], use_task_gate) if skip2 is not None else F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
        x1 = self.up1(x2, skip1, task_summary, guidance_prob, stage_strengths[3], use_task_gate) if skip1 is not None else F.interpolate(x2, scale_factor=2, mode="bilinear", align_corners=False)

        aux_logits = []
        if self.use_deep_supervision and self.training:
            aux_logits = [self.aux4(x4), self.aux3(x3), self.aux2(x2)]

        return {
            "logits": self.head(x1),
            "aux_logits": aux_logits,
            "dense_guidance_lowres": dense_logits,
            "task_tokens": task_tokens,
        }

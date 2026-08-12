from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import build_2d_norm


class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type="group", group_norm_groups=16):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            build_2d_norm(out_channels, norm_type=norm_type, group_norm_groups=group_norm_groups),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            build_2d_norm(out_channels, norm_type=norm_type, group_norm_groups=group_norm_groups),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class TextInducedDenseGuidance2D(nn.Module):
    """
    Lightweight text-induced dense guidance.
    """

    def __init__(
        self,
        in_channels: int,
        text_dim: int,
        guidance_dim: int = 256,
        text_group_summary_count: int = 5,
        norm_type: str = "group",
        group_norm_groups: int = 16,
    ):
        super().__init__()
        self.group_summary_count = int(text_group_summary_count)
        self.guidance_dim = int(guidance_dim)

        self.feature_proj = nn.Sequential(
            nn.Conv2d(in_channels, guidance_dim, kernel_size=1, bias=False),
            build_2d_norm(guidance_dim, norm_type=norm_type, group_norm_groups=group_norm_groups),
            nn.GELU(),
        )

        self.token_key_proj = nn.Linear(text_dim, guidance_dim, bias=False)
        self.token_value_proj = nn.Linear(text_dim, guidance_dim, bias=False)
        self.seed_proj = nn.Linear(text_dim, guidance_dim)

        self.dense_text_refine = nn.Sequential(
            nn.Conv2d(guidance_dim, guidance_dim, kernel_size=3, padding=1, bias=False),
            build_2d_norm(guidance_dim, norm_type=norm_type, group_norm_groups=group_norm_groups),
            nn.GELU(),
        )

        self.fuse = ConvNormAct(
            guidance_dim + 1,
            guidance_dim,
            norm_type=norm_type,
            group_norm_groups=group_norm_groups,
        )
        self.guidance_head = nn.Conv2d(guidance_dim, 1, kernel_size=1)
        self.guidance_feat_head = nn.Sequential(
            nn.Conv2d(guidance_dim, in_channels, kernel_size=1, bias=False),
            build_2d_norm(in_channels, norm_type=norm_type, group_norm_groups=group_norm_groups),
            nn.GELU(),
        )

    def _split_fine_tokens(self, text_tokens: torch.Tensor) -> torch.Tensor:
        if text_tokens.shape[1] <= self.group_summary_count:
            return text_tokens
        return text_tokens[:, self.group_summary_count:, :]

    def _text_similarity_map(self, feat_norm: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
        vec = self.seed_proj(token)
        vec = F.normalize(vec, dim=1).unsqueeze(-1).unsqueeze(-1)
        return torch.sigmoid((feat_norm * vec).sum(dim=1, keepdim=True))

    def forward(self, image_feats: torch.Tensor, text_tokens: torch.Tensor | None):
        batch_size, _, h, w = image_feats.shape
        if text_tokens is None or text_tokens.numel() == 0:
            zero_map = image_feats.new_zeros((batch_size, 1, h, w))
            zero_feats = image_feats.new_zeros_like(image_feats)
            return {
                "guidance_map": zero_map,
                "guidance_feats": zero_feats,
            }

        feat = self.feature_proj(image_feats)
        feat_norm = F.normalize(feat, dim=1)
        prior_map = self._text_similarity_map(feat_norm, text_tokens.mean(dim=1))

        fine_tokens = self._split_fine_tokens(text_tokens)
        
        # 🌟 修复: 移除 Q 和 K 错误的 L2 归一化，使用标准的 Scaled Dot-Product
        q = feat.flatten(2).transpose(1, 2)
        k = self.token_key_proj(fine_tokens)
        v = self.token_value_proj(fine_tokens)

        attn_logits = torch.bmm(q, k.transpose(1, 2)) / (self.guidance_dim ** 0.5)
        attn = torch.softmax(attn_logits, dim=-1)

        dense_text = torch.bmm(attn, v)
        dense_text = dense_text.transpose(1, 2).reshape(batch_size, self.guidance_dim, h, w)
        dense_text = self.dense_text_refine(dense_text)

        fused = self.fuse(torch.cat([feat + dense_text, prior_map], dim=1))
        
        # 🌟 修复: 移除 torch.sigmoid，直接返回 Logits 以便计算稳定的 BCEWithLogitsLoss
        guidance_map_logits = self.guidance_head(fused)
        guidance_feats = self.guidance_feat_head(fused)

        return {
            "guidance_map": guidance_map_logits,
            "guidance_feats": guidance_feats,
        }
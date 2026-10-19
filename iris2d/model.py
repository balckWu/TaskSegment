"""Paper-based Iris-2D adaptation, not author-provided code.

Dimensional adaptation: 2D U-Net, exact high-resolution foreground pooling,
full-resolution pixel shuffle/mask fusion/unshuffle, contextual cross/self
attention, two-way query/task attention, and a token-conditioned mask head.
Widths, normalization, attention depth and the mask head are explicit choices
where the paper does not specify enough detail to reconstruct the authors' code.
"""
from __future__ import annotations

import math
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F


def norm(channels):
    return nn.GroupNorm(math.gcd(channels, 16), channels)


class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), norm(cout), nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), norm(cout), nn.GELU(),
        )


class Encoder(nn.Module):
    def __init__(self, base):
        super().__init__()
        widths = [base * 2 ** i for i in range(5)]
        self.blocks = nn.ModuleList(
            ConvBlock(1 if i == 0 else widths[i - 1], c)
            for i, c in enumerate(widths)
        )

    def forward(self, x):
        features = []
        for i, block in enumerate(self.blocks):
            x = block(F.max_pool2d(x, 2) if i else x)
            features.append(x)
        return features


def interpolation_weights(low, high, device):
    """Weights for align_corners=False bilinear interpolation along one axis."""
    position = ((torch.arange(high, device=device, dtype=torch.float32) + .5)
                * low / high - .5).clamp(0, low - 1)
    left = position.floor().long()
    right = (left + 1).clamp_max(low - 1)
    fraction = position - left
    weights = torch.zeros(high, low, device=device, dtype=torch.float32)
    weights.scatter_add_(1, left[:, None], (1 - fraction)[:, None])
    weights.scatter_add_(1, right[:, None], fraction[:, None])
    return weights


def high_resolution_masked_pool(features, masks):
    """Exactly pool bilinearly upsampled features using the original masks.

    Apply the transpose of the separable interpolation operator to the mask.
    This avoids allocating [B,C,H,W] full-resolution support features and keeps
    gradients to the low-resolution features mathematically unchanged.
    """
    if masks.ndim == 3:
        masks = masks[:, None]
    context = torch.autocast(features.device.type, enabled=False) if features.device.type in {"cpu", "cuda"} else nullcontext()
    with context:
        m = masks[:, 0].float()
        h, w = features.shape[-2:]
        wy = interpolation_weights(h, m.shape[-2], m.device)
        wx = interpolation_weights(w, m.shape[-1], m.device)
        weighted_mask = torch.matmul(torch.matmul(wy.T, m), wx)
        pooled = (features.float() * weighted_mask[:, None]).sum((-2, -1))
        pooled = pooled / m.sum((-2, -1))[:, None].clamp_min(1e-6)
    return pooled[:, None].to(features.dtype)


class TokenBlock(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.cross = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(3))
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim))

    def forward(self, tokens, spatial):
        tokens = self.norms[0](tokens + self.cross(tokens, spatial, spatial, need_weights=False)[0])
        tokens = self.norms[1](tokens + self.self_attention(tokens, tokens, tokens, need_weights=False)[0])
        return self.norms[2](tokens + self.ffn(tokens))


class TaskEncoder(nn.Module):
    def __init__(self, dim, queries, heads, dropout, upscale=16):
        super().__init__()
        if dim % upscale ** 2:
            raise ValueError("Task width must be divisible by shuffle factor squared")
        compact = dim // upscale ** 2
        self.upscale = upscale
        self.mask_fusion = nn.Conv2d(compact + 1, compact, 1)
        self.queries = nn.Parameter(torch.randn(1, queries, dim) * .02)
        self.context = TokenBlock(dim, heads, dropout)

    def forward(self, support_features, masks):
        foreground = high_resolution_masked_pool(support_features, masks)
        up = F.pixel_shuffle(support_features, self.upscale)
        m = F.interpolate(masks.float(), size=up.shape[-2:], mode="nearest").to(up.dtype)
        fused = F.pixel_unshuffle(self.mask_fusion(torch.cat([up, m], 1)), self.upscale)
        spatial = fused.flatten(2).transpose(1, 2)
        context = self.context(self.queries.expand(spatial.shape[0], -1, -1), spatial)
        return torch.cat([foreground, context], 1)


class TwoWayBlock(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.task_self = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.task_to_image = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.image_to_task = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(5))
        self.task_ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.image_ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, image, task):
        task = self.norms[0](task + self.task_self(task, task, task, need_weights=False)[0])
        task = self.norms[1](task + self.task_to_image(task, image, image, need_weights=False)[0])
        task = self.norms[2](task + self.task_ffn(task))
        image = self.norms[3](image + self.image_to_task(image, task, task, need_weights=False)[0])
        image = self.norms[4](image + self.image_ffn(image))
        return image, task


class Iris2D(nn.Module):
    def __init__(self, base_channels=32, context_queries=10, heads=8, attention_depth=2, dropout=.1):
        super().__init__()
        if base_channels <= 0 or base_channels % 16:
            raise ValueError("base_channels must be a positive multiple of 16")
        if context_queries <= 0 or attention_depth <= 0 or heads <= 0:
            raise ValueError("Attention counts must be positive")
        dim = base_channels * 16
        if dim % heads:
            raise ValueError("Task width must be divisible by heads")
        self.config = dict(base_channels=base_channels, context_queries=context_queries,
                           heads=heads, attention_depth=attention_depth, dropout=dropout)
        self.encoder = Encoder(base_channels)
        self.task_encoder = TaskEncoder(dim, context_queries, heads, dropout)
        self.two_way = nn.ModuleList(TwoWayBlock(dim, heads, dropout) for _ in range(attention_depth))
        self.up = nn.ModuleList(
            ConvBlock(base_channels * 2 ** (i + 1) + base_channels * 2 ** i, base_channels * 2 ** i)
            for i in reversed(range(4))
        )
        self.pixel_head = nn.Conv2d(base_channels, base_channels, 1)
        self.mask_token_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, base_channels))

    @staticmethod
    def _check_image(x):
        if x.ndim != 4 or x.shape[1] != 1 or any(v % 16 for v in x.shape[-2:]):
            raise ValueError("Images must be [B,1,H,W] with H,W divisible by 16")

    def encode_support(self, images, masks, aggregate=True):
        self._check_image(images)
        if masks.ndim == 3:
            masks = masks[:, None]
        if masks.shape != images.shape:
            raise ValueError("Support masks and images must have matching [K,1,H,W] shapes")
        if not (masks.flatten(1).sum(1) > 0).all():
            raise ValueError("Each support mask must contain foreground")
        tokens = self.task_encoder(self.encoder(images)[-1], masks)
        return tokens.mean(0, keepdim=True) if aggregate else tokens

    def segment(self, query, tokens):
        self._check_image(query)
        features = self.encoder(query)
        x = features[-1]
        image = x.flatten(2).transpose(1, 2)
        if tokens.shape[0] == 1:
            tokens = tokens.expand(query.shape[0], -1, -1)
        if tokens.shape[0] != query.shape[0]:
            raise ValueError("Task token batch must be 1 or query batch size")
        for block in self.two_way:
            image, tokens = block(image, tokens)
        x = image.transpose(1, 2).reshape_as(x)
        for block, skip in zip(self.up, reversed(features[:-1])):
            x = block(torch.cat([F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False), skip], 1))
        pixel = self.pixel_head(x)
        mask_vector = self.mask_token_head(tokens[:, 0])
        foreground_logit = torch.einsum("bc,bchw->bhw", mask_vector, pixel)[:, None]
        return torch.cat([torch.zeros_like(foreground_logit), foreground_logit], 1)

    def forward(self, support_images, support_masks, query):
        return self.segment(query, self.encode_support(support_images, support_masks))


def segmentation_loss(logits, target):
    """One CE + one foreground soft-Dice; no US-TextSeg-specific losses."""
    ce = F.cross_entropy(logits.float(), target.long())
    probability = logits.float().softmax(1)[:, 1]
    foreground = target.float()
    dice = (2 * (probability * foreground).sum((-2, -1)) + 1) / (
        probability.sum((-2, -1)) + foreground.sum((-2, -1)) + 1)
    return ce + (1 - dice).mean()

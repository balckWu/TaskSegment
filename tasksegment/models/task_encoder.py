from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import build_2d_norm


def _positive_log_init(value: float, name: str) -> torch.Tensor:
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}")
    return torch.tensor(math.log(value), dtype=torch.float32)


class TaskEncoder2D(nn.Module):
    """Build visual task tokens and domain-aware structured text tokens.

    The text branch combines three complementary signals:

    1. projected BioBERT medical semantics;
    2. a learnable closed-set domain residual;
    3. a learnable text-group residual.

    Domain and group embeddings are L2-normalized and rescaled to have the same
    natural norm as a LayerNorm feature (approximately ``sqrt(hidden_dim)``).
    Their learnable scalar gates therefore have an interpretable magnitude.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_query_tokens: int = 4,
        context_upscale: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_ftext: bool = True,
        use_semantic_text: bool = True,
        text_dim: int = 1024,
        text_group_summary_count: int = 5,
        domain_names: Sequence[str] = (
            "thyroid",
            "TN3K",
            "BUSI_WHU",
            "BUS-BRA",
            "OTU",
            "prostate",
        ),
        use_domain_embedding: bool = True,
        use_group_embedding: bool = True,
        domain_embedding_scale_init: float = 0.50,
        group_embedding_scale_init: float = 0.25,
        text_semantic_scale_init: float = 1.00,
        norm_type: str = "group",
        group_norm_groups: int = 16,
    ) -> None:
        super().__init__()
        if hidden_dim % (context_upscale ** 2) != 0:
            raise ValueError("hidden_dim must be divisible by context_upscale^2 for PixelShuffle")
        if text_group_summary_count <= 0:
            raise ValueError("text_group_summary_count must be positive")

        domain_names = tuple(str(name) for name in domain_names)
        if not domain_names or len(set(domain_names)) != len(domain_names):
            raise ValueError("domain_names must be a non-empty sequence of unique names")

        self.hidden_dim = int(hidden_dim)
        self.num_query_tokens = int(num_query_tokens)
        self.context_upscale = int(context_upscale)
        self.use_ftext = bool(use_ftext)
        self.use_semantic_text = bool(use_semantic_text)
        self.text_group_summary_count = int(text_group_summary_count)
        self.domain_names = domain_names
        self.domain_to_index = {name: index for index, name in enumerate(domain_names)}
        self.use_domain_embedding = bool(use_domain_embedding)
        self.use_group_embedding = bool(use_group_embedding)

        self.context_in_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        shuffled_dim = hidden_dim // (context_upscale ** 2)
        self.context_fuse_conv = nn.Sequential(
            nn.Conv2d(shuffled_dim + 1, shuffled_dim, kernel_size=3, padding=1, bias=False),
            build_2d_norm(shuffled_dim, norm_type, group_norm_groups),
            nn.GELU(),
            nn.Conv2d(shuffled_dim, shuffled_dim, kernel_size=3, padding=1, bias=False),
            build_2d_norm(shuffled_dim, norm_type, group_norm_groups),
            nn.GELU(),
        )

        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.query_norm1 = nn.LayerNorm(hidden_dim)
        self.query_norm2 = nn.LayerNorm(hidden_dim)
        self.query_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.query_dropout = nn.Dropout(dropout)

        self.fg_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        if self.use_ftext:
            self.ftext_proj = nn.Sequential(
                nn.Linear(text_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.ftext_norm = nn.LayerNorm(hidden_dim)

        if self.use_domain_embedding:
            self.domain_embedding = nn.Embedding(len(domain_names), hidden_dim)
            nn.init.normal_(self.domain_embedding.weight, mean=0.0, std=0.02)
            self.domain_log_scale = nn.Parameter(
                _positive_log_init(domain_embedding_scale_init, "domain_embedding_scale_init")
            )

        if self.use_group_embedding:
            self.group_embedding = nn.Parameter(
                torch.randn(1, text_group_summary_count, hidden_dim) * 0.02
            )
            self.group_log_scale = nn.Parameter(
                _positive_log_init(group_embedding_scale_init, "group_embedding_scale_init")
            )

        self.semantic_log_scale = nn.Parameter(
            _positive_log_init(text_semantic_scale_init, "text_semantic_scale_init")
        )
        self.condition_norm = nn.LayerNorm(hidden_dim)
        self.condition_log_scale = nn.Parameter(torch.zeros((), dtype=torch.float32))

    @property
    def visual_token_count(self) -> int:
        return 1 + self.num_query_tokens

    @property
    def condition_token_count(self) -> int:
        if self.use_ftext or self.use_domain_embedding or self.use_group_embedding:
            return self.text_group_summary_count
        return 0

    def domain_ids_from_names(
        self,
        names: str | Sequence[str],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if isinstance(names, str):
            names = [names]
        indices = []
        for name in names:
            key = str(name)
            if key not in self.domain_to_index:
                raise KeyError(
                    f"Unknown domain '{key}'. Expected one of {tuple(self.domain_to_index)}"
                )
            indices.append(self.domain_to_index[key])
        return torch.tensor(indices, dtype=torch.long, device=device)

    @staticmethod
    def _ensure_4d_mask(mask: torch.Tensor) -> torch.Tensor:
        if mask.dim() == 3:
            return mask.unsqueeze(1)
        if mask.dim() != 4:
            raise ValueError(f"support_mask must have 3 or 4 dims, got {tuple(mask.shape)}")
        return mask

    @staticmethod
    def masked_average(feats: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        masked = feats * mask
        denom = mask.sum(dim=(2, 3)).clamp_min(eps)
        return masked.sum(dim=(2, 3)) / denom

    @staticmethod
    def _unwrap_feats(feats):
        return feats["bottleneck"] if isinstance(feats, dict) else feats

    def _prepare_domain_ids(
        self,
        domain_ids: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if not self.use_domain_embedding:
            return None
        if domain_ids is None:
            raise ValueError(
                "domain_ids are required when use_domain_embedding=True. "
                "Pass the known closed-set domain for every support/query batch."
            )
        ids = torch.as_tensor(domain_ids, dtype=torch.long, device=device).reshape(-1)
        if ids.numel() == 1 and batch_size > 1:
            ids = ids.expand(batch_size)
        if ids.numel() != batch_size:
            raise ValueError(
                f"domain_ids batch mismatch: got {ids.numel()}, expected {batch_size}"
            )
        if int(ids.min()) < 0 or int(ids.max()) >= len(self.domain_names):
            raise ValueError("domain_ids contain an out-of-range value")
        return ids

    def project_text_tokens(
        self,
        text_tokens: torch.Tensor | None,
        domain_ids: torch.Tensor | None = None,
        batch_size: int | None = None,
    ) -> torch.Tensor | None:
        if text_tokens is None:
            if batch_size is None:
                if domain_ids is None:
                    return None
                batch_size = int(torch.as_tensor(domain_ids).numel())
            device = (
                domain_ids.device
                if torch.is_tensor(domain_ids)
                else self.condition_norm.weight.device
            )
            semantic = torch.zeros(
                int(batch_size),
                self.text_group_summary_count,
                self.hidden_dim,
                device=device,
                dtype=self.condition_norm.weight.dtype,
            )
        else:
            if text_tokens.ndim == 2:
                text_tokens = text_tokens.unsqueeze(0)
            if text_tokens.ndim != 3:
                raise ValueError(f"text_tokens must be [B,N,D], got {tuple(text_tokens.shape)}")
            if text_tokens.shape[1] < self.text_group_summary_count:
                raise ValueError(
                    f"Expected at least {self.text_group_summary_count} text summary tokens, "
                    f"got {text_tokens.shape[1]}"
                )
            if batch_size is not None and text_tokens.shape[0] == 1 and int(batch_size) > 1:
                text_tokens = text_tokens.expand(int(batch_size), -1, -1)
            batch_size = int(text_tokens.shape[0])
            device = text_tokens.device
            if self.use_ftext and self.use_semantic_text:
                summaries = text_tokens[:, : self.text_group_summary_count].float()
                semantic = self.ftext_norm(self.ftext_proj(summaries))
            else:
                # Formal semantic-text ablation: preserve the complete text
                # projection branch and token topology, but remove only its
                # biomedical semantic content. This keeps parameter count
                # identical to the full model.
                semantic = torch.zeros(
                    batch_size,
                    self.text_group_summary_count,
                    self.hidden_dim,
                    device=device,
                    dtype=self.condition_norm.weight.dtype,
                )

        if not (self.use_ftext or self.use_domain_embedding or self.use_group_embedding):
            return None

        semantic_scale = torch.exp(self.semantic_log_scale).clamp(0.10, 4.0).to(semantic.dtype)
        conditioned = semantic_scale * semantic
        target_norm = math.sqrt(float(self.hidden_dim))

        if self.use_group_embedding:
            group = F.normalize(self.group_embedding, dim=-1, eps=1e-6) * target_norm
            group = group.to(conditioned.dtype)
            group_scale = torch.exp(self.group_log_scale).clamp(0.05, 2.0).to(conditioned.dtype)
            conditioned = conditioned + group_scale * group.expand(batch_size, -1, -1)

        if self.use_domain_embedding:
            ids = self._prepare_domain_ids(domain_ids, batch_size, device)
            assert ids is not None
            domain = self.domain_embedding(ids)
            domain = F.normalize(domain, dim=-1, eps=1e-6) * target_norm
            domain = domain.to(conditioned.dtype)
            domain_scale = torch.exp(self.domain_log_scale).clamp(0.05, 2.0).to(conditioned.dtype)
            conditioned = conditioned + domain_scale * domain.unsqueeze(1)

        output_scale = torch.exp(self.condition_log_scale).clamp(0.25, 4.0).to(conditioned.dtype)
        return output_scale * self.condition_norm(conditioned)

    def shared_visual_tokens(
        self,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Return domain-agnostic visual tokens using existing parameters.

        This is used by TSV5's ``no_task_specific_representation`` ablation.
        It intentionally reads neither support images/masks nor domain IDs. The
        four global query tokens are reused, and their mean is projected into
        the foreground-token slot. No new learnable parameter is introduced.
        """
        reference = self.query_tokens
        target_device = reference.device if device is None else device
        target_dtype = reference.dtype if dtype is None else dtype
        query = reference.to(device=target_device, dtype=target_dtype).expand(
            int(batch_size), -1, -1
        )
        foreground = self.fg_proj(query.mean(dim=1)).unsqueeze(1)
        return torch.cat([foreground, query], dim=1)

    def shared_condition_tokens(
        self,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor | None:
        """Return shared condition tokens without semantic/domain leakage.

        Group-position embeddings are shared across every dataset and therefore
        preserve the five-token topology without encoding the dataset identity.
        The semantic projection and domain embedding remain instantiated, so the
        grouped ablation changes information flow rather than backbone capacity.
        """
        if self.condition_token_count <= 0:
            return None
        reference = self.condition_norm.weight
        target_device = reference.device if device is None else device
        target_dtype = reference.dtype if dtype is None else dtype
        conditioned = torch.zeros(
            int(batch_size),
            self.text_group_summary_count,
            self.hidden_dim,
            device=target_device,
            dtype=target_dtype,
        )
        if self.use_group_embedding:
            target_norm = math.sqrt(float(self.hidden_dim))
            group = F.normalize(self.group_embedding, dim=-1, eps=1e-6) * target_norm
            group = group.to(device=target_device, dtype=target_dtype)
            group_scale = torch.exp(self.group_log_scale).clamp(0.05, 2.0).to(target_dtype)
            conditioned = conditioned + group_scale * group.expand(int(batch_size), -1, -1)
        output_scale = torch.exp(self.condition_log_scale).clamp(0.25, 4.0).to(target_dtype)
        return output_scale * self.condition_norm(conditioned)

    def shared_task_tokens(
        self,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        visual = self.shared_visual_tokens(batch_size, device=device, dtype=dtype)
        condition = self.shared_condition_tokens(batch_size, device=device, dtype=dtype)
        if condition is None:
            return visual
        condition = condition.to(device=visual.device, dtype=visual.dtype)
        return torch.cat([visual, condition], dim=1)

    def compose_task_tokens(
        self,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor | None = None,
        domain_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        projected_text = self.project_text_tokens(
            text_tokens,
            domain_ids=domain_ids,
            batch_size=visual_tokens.shape[0],
        )
        if projected_text is None:
            return visual_tokens
        if projected_text.shape[0] == 1 and visual_tokens.shape[0] > 1:
            projected_text = projected_text.expand(visual_tokens.shape[0], -1, -1)
        if projected_text.shape[0] != visual_tokens.shape[0]:
            raise ValueError("visual/text task-token batch sizes do not match")
        return torch.cat([visual_tokens, projected_text], dim=1)

    def forward_visual(self, support_feats, support_mask) -> torch.Tensor:
        support_feats = self._unwrap_feats(support_feats)
        support_mask = self._ensure_4d_mask(support_mask).float()
        support_mask_lowres = F.interpolate(support_mask, size=support_feats.shape[-2:], mode="nearest")

        raw_foreground = self.masked_average(support_feats, support_mask_lowres)
        foreground_token = self.fg_proj(raw_foreground).unsqueeze(1)

        high_res_feats = F.pixel_shuffle(self.context_in_proj(support_feats), self.context_upscale)
        high_res_mask = F.interpolate(support_mask, size=high_res_feats.shape[-2:], mode="nearest")
        fused = self.context_fuse_conv(torch.cat([high_res_feats, high_res_mask], dim=1))
        fused = F.pixel_unshuffle(fused, self.context_upscale)
        fused_flat = fused.flatten(2).transpose(1, 2)

        query_tokens = self.query_tokens.expand(support_feats.shape[0], -1, -1)
        ctx_tokens, _ = self.cross_attn(query_tokens, fused_flat, fused_flat, need_weights=False)
        query_tokens = self.query_norm1(query_tokens + self.query_dropout(ctx_tokens))
        query_tokens = self.query_norm2(query_tokens + self.query_dropout(self.query_ffn(query_tokens)))
        return torch.cat([foreground_token, query_tokens], dim=1)

    def forward(
        self,
        support_feats,
        support_mask,
        text_tokens=None,
        domain_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        visual = self.forward_visual(support_feats, support_mask)
        return self.compose_task_tokens(visual, text_tokens, domain_ids=domain_ids)

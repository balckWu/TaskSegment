from __future__ import annotations

from tasksegment.visualization.export_visual_token_states import (
    maybe_export_visual_token_states,
)

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import IrisMaskDecoder2D
from .encoder import UNetEncoder2D
from .task_encoder import TaskEncoder2D


class TaskSegmentModel(nn.Module):
    """Task-conditioned binary ultrasound segmentation model.

    The domain identity is known in this closed-set setting. It is therefore
    passed explicitly to the text/task encoder instead of expecting highly
    correlated BioBERT sentence vectors to identify the dataset by themselves.
    """

    def __init__(
        self,
        use_ftext: bool = True,
        use_semantic_text: bool = True,
        hidden_dim: int = 1024,
        num_query_tokens: int = 4,
        context_upscale: int = 4,
        dropout: float = 0.1,
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
        task_norm_type: str = "group",
        group_norm_groups: int = 16,
        encoder_norm_type: str = "group",
        encoder_batchnorm: bool | None = None,
        fg_token_weight: float = 0.30,
        skip_gate_mode: str = "centered",
        aggregate_support_tokens: bool = True,
        use_visual_prototype: bool = True,
        use_task_specific_representation: bool = True,
        use_task_conditioned_decoder: bool = True,
        use_dense_text_guidance: bool = True,
        dense_guidance_dim: int = 256,
        dense_guidance_strength: float = 0.50,
        guidance_dropout_prob: float = 0.10,
        guidance_scale_min: float = 0.80,
        guidance_scale_max: float = 1.20,
        guidance_clamp_max: float = 0.90,
        guidance_stage_weights: Sequence[float] = (1.0, 0.50, 0.25, 0.0, 0.0),
        guidance_modulation_mode: str = "centered",
        use_deep_supervision: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = UNetEncoder2D(
            norm_type=encoder_norm_type,
            group_norm_groups=group_norm_groups,
            is_batchnorm=encoder_batchnorm,
        )
        encoder_bottleneck_dim = self.encoder.filters[-1]
        if hidden_dim != encoder_bottleneck_dim:
            raise ValueError(
                f"hidden_dim={hidden_dim} must equal encoder bottleneck channels={encoder_bottleneck_dim}"
            )

        self.use_ftext = bool(use_ftext)
        self.use_semantic_text = bool(use_semantic_text)
        self.use_domain_embedding = bool(use_domain_embedding)
        self.use_group_embedding = bool(use_group_embedding)
        self.aggregate_support_tokens = bool(aggregate_support_tokens)
        self.use_visual_prototype = bool(use_visual_prototype)
        self.use_task_specific_representation = bool(use_task_specific_representation)
        self.use_task_conditioned_decoder = bool(use_task_conditioned_decoder)
        self.task_encoder = TaskEncoder2D(
            hidden_dim=hidden_dim,
            num_query_tokens=num_query_tokens,
            context_upscale=context_upscale,
            dropout=dropout,
            use_ftext=use_ftext,
            use_semantic_text=use_semantic_text,
            text_dim=text_dim,
            text_group_summary_count=text_group_summary_count,
            domain_names=domain_names,
            use_domain_embedding=use_domain_embedding,
            use_group_embedding=use_group_embedding,
            domain_embedding_scale_init=domain_embedding_scale_init,
            group_embedding_scale_init=group_embedding_scale_init,
            text_semantic_scale_init=text_semantic_scale_init,
            norm_type=task_norm_type,
            group_norm_groups=group_norm_groups,
        )

        # The decoder's historical ``use_ftext`` flag really means that tokens
        # exist after the visual-token prefix. Domain/group-only ablations still
        # create such condition tokens, so use the actual condition-token state.
        use_condition_tokens = self.task_encoder.condition_token_count > 0
        self.decoder = IrisMaskDecoder2D(
            hidden_dim=hidden_dim,
            num_query_tokens=num_query_tokens,
            use_ftext=use_condition_tokens,
            use_task_conditioning=self.use_task_conditioned_decoder,
            dropout=dropout,
            fg_token_weight=fg_token_weight,
            skip_gate_mode=skip_gate_mode,
            norm_type=task_norm_type,
            group_norm_groups=group_norm_groups,
            encoder_filters=tuple(self.encoder.filters),
            use_dense_text_guidance=use_dense_text_guidance,
            dense_guidance_dim=dense_guidance_dim,
            dense_guidance_strength=dense_guidance_strength,
            guidance_dropout_prob=guidance_dropout_prob,
            guidance_scale_min=guidance_scale_min,
            guidance_scale_max=guidance_scale_max,
            guidance_clamp_max=guidance_clamp_max,
            text_group_summary_count=text_group_summary_count,
            guidance_stage_weights=guidance_stage_weights,
            guidance_modulation_mode=guidance_modulation_mode,
            use_deep_supervision=use_deep_supervision,
        )

    @property
    def visual_token_count(self) -> int:
        return self.task_encoder.visual_token_count

    @property
    def requires_visual_support(self) -> bool:
        return bool(
            self.use_task_conditioned_decoder
            and self.use_task_specific_representation
            and self.use_visual_prototype
        )

    @property
    def requires_condition_inputs(self) -> bool:
        return bool(
            self.use_task_conditioned_decoder
            and self.use_task_specific_representation
            and self.task_encoder.condition_token_count > 0
        )

    @property
    def requires_visual_prototype_bank(self) -> bool:
        return self.requires_visual_support

    @property
    def domain_names(self) -> tuple[str, ...]:
        return self.task_encoder.domain_names

    def domain_ids_from_names(
        self,
        names: str | Sequence[str],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        return self.task_encoder.domain_ids_from_names(names, device=device)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.forward_features(x)["bottleneck"]

    def encode_image_pyramid(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.encoder.forward_features(x)

    @staticmethod
    def aggregate_tokens(tokens: torch.Tensor, output_batch_size: int | None = None) -> torch.Tensor:
        prototype = tokens.mean(dim=0, keepdim=True)
        if output_batch_size is not None and int(output_batch_size) > 1:
            prototype = prototype.expand(int(output_batch_size), -1, -1)
        return prototype

    def null_visual_tokens(
        self,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Return parameter-free null visual tokens for the prototype ablation.

        The visual task encoder remains instantiated, so parameter count is
        unchanged, but no support image or support mask is read or encoded.
        """
        reference = self.task_encoder.query_tokens
        return torch.zeros(
            int(batch_size),
            self.visual_token_count,
            self.task_encoder.hidden_dim,
            device=reference.device if device is None else device,
            dtype=reference.dtype if dtype is None else dtype,
        )

    def shared_task_tokens(
        self,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return self.task_encoder.shared_task_tokens(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

    def compose_task_tokens(
        self,
        visual_tokens: torch.Tensor,
        Ftext: torch.Tensor | None = None,
        domain_ids: torch.Tensor | None = None,
        output_batch_size: int | None = None,
    ) -> torch.Tensor:
        if output_batch_size is not None and visual_tokens.shape[0] == 1 and output_batch_size > 1:
            visual_tokens = visual_tokens.expand(output_batch_size, -1, -1)
        if Ftext is not None and Ftext.shape[0] == 1 and visual_tokens.shape[0] > 1:
            Ftext = Ftext.expand(visual_tokens.shape[0], -1, -1)
        return self.task_encoder.compose_task_tokens(
            visual_tokens,
            Ftext,
            domain_ids=domain_ids,
        )

    def split_task_tokens(self, task_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        visual = task_tokens[:, : self.visual_token_count]
        text = task_tokens[:, self.visual_token_count :] if task_tokens.shape[1] > self.visual_token_count else None
        return visual, text

    def encode_task(
        self,
        xs: torch.Tensor | None,
        ys: torch.Tensor | None,
        Ftext: torch.Tensor | None = None,
        domain_ids: torch.Tensor | None = None,
        support_feats: torch.Tensor | None = None,
        support_pyramid: dict[str, torch.Tensor] | None = None,
        aggregate_support: bool | None = None,
        output_batch_size: int | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if not self.use_task_conditioned_decoder:
            return {
                "support_feats": None,
                "support_pyramid": None,
                "raw_visual_task_tokens": None,
                "visual_task_tokens": None,
                "task_tokens": None,
                "domain_ids": None,
            }

        if not self.use_task_specific_representation:
            if output_batch_size is None:
                if Ftext is not None:
                    output_batch_size = int(Ftext.shape[0])
                elif domain_ids is not None:
                    output_batch_size = int(torch.as_tensor(domain_ids).numel())
                elif xs is not None:
                    output_batch_size = int(xs.shape[0])
                else:
                    raise ValueError(
                        "output_batch_size is required for shared task tokens"
                    )
            task_tokens = self.shared_task_tokens(int(output_batch_size))
            visual_tokens, condition_tokens = self.split_task_tokens(task_tokens)
            return {
                "support_feats": None,
                "support_pyramid": None,
                "raw_visual_task_tokens": visual_tokens,
                "visual_task_tokens": visual_tokens,
                "condition_task_tokens": condition_tokens,
                "task_tokens": task_tokens,
                "domain_ids": None,
            }

        if not self.use_visual_prototype:
            if output_batch_size is None:
                if domain_ids is not None:
                    output_batch_size = int(torch.as_tensor(domain_ids).numel())
                elif Ftext is not None:
                    output_batch_size = int(Ftext.shape[0])
                else:
                    raise ValueError("output_batch_size is required when visual prototypes are disabled")
            raw_visual_tokens = self.null_visual_tokens(int(output_batch_size))
            visual_tokens = raw_visual_tokens
            support_feats = None
            support_pyramid = None
        else:
            if xs is None or ys is None:
                raise ValueError("support images and masks are required when use_visual_prototype=True")
            if support_pyramid is None and support_feats is None:
                support_pyramid = self.encoder.forward_features(xs)
            if support_feats is None:
                if support_pyramid is None:
                    raise RuntimeError("support feature extraction failed")
                support_feats = support_pyramid["bottleneck"]

            raw_visual_tokens = self.task_encoder.forward_visual(support_feats, ys.float())
            do_aggregate = self.aggregate_support_tokens if aggregate_support is None else bool(aggregate_support)
            visual_tokens = raw_visual_tokens
            if do_aggregate:
                visual_tokens = self.aggregate_tokens(raw_visual_tokens, output_batch_size=output_batch_size)
            elif output_batch_size is not None and visual_tokens.shape[0] != output_batch_size:
                raise ValueError("Non-aggregated support tokens require matching support/query batch sizes")

        task_tokens = self.compose_task_tokens(
            visual_tokens,
            Ftext,
            domain_ids=domain_ids,
            output_batch_size=output_batch_size,
        )
        _, condition_tokens = self.split_task_tokens(task_tokens)

        # Optional paper/debug export of support-derived visual token states.
        # This remains a no-op unless the visualization exporter is explicitly enabled.
        maybe_export_visual_token_states(
            xs=xs,
            ys=ys,
            support_pyramid=support_pyramid,
            raw_visual_tokens=raw_visual_tokens,
            visual_tokens=visual_tokens,
            domain_ids=domain_ids,
            domain_names=self.domain_names,
        )

        return {
            "support_feats": support_feats,
            "support_pyramid": support_pyramid,
            "raw_visual_task_tokens": raw_visual_tokens,
            "visual_task_tokens": visual_tokens,
            "condition_task_tokens": condition_tokens,
            "task_tokens": task_tokens,
            "domain_ids": domain_ids,
        }

    def encode_query(self, xq: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.encoder.forward_features(xq)

    @staticmethod
    def _maybe_resize_map(x: torch.Tensor | None, output_size):
        if x is None or output_size is None or x.shape[-2:] == output_size:
            return x
        return F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)

    def segment_with_task(
        self,
        xq: torch.Tensor | None = None,
        task_tokens: torch.Tensor | None = None,
        query_feats=None,
        output_size=None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | None]:
        if self.use_task_conditioned_decoder and task_tokens is None:
            raise ValueError("task_tokens must be provided for the conditioned decoder")
        if xq is not None:
            query_pyramid = self.encoder.forward_features(xq)
        elif query_feats is not None:
            query_pyramid = query_feats if isinstance(query_feats, dict) else {"bottleneck": query_feats}
        else:
            raise ValueError("Either xq or query_feats must be provided")

        query_batch = query_pyramid["bottleneck"].shape[0]
        if task_tokens is not None:
            if task_tokens.shape[0] == 1 and query_batch > 1:
                task_tokens = task_tokens.expand(query_batch, -1, -1)
            if task_tokens.shape[0] != query_batch:
                raise ValueError("task token batch size must equal query batch size or be 1")

        decoder_out = self.decoder(query_pyramid, task_tokens)
        logits = decoder_out["logits"]
        if output_size is None and xq is not None:
            output_size = xq.shape[-2:]
        if output_size is not None and logits.shape[-2:] != output_size:
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)

        guidance_lowres = decoder_out.get("dense_guidance_lowres")
        return {
            "pred_masks": logits,
            "aux_logits": decoder_out.get("aux_logits", []),
            "query_feats": query_pyramid["bottleneck"],
            "query_pyramid": query_pyramid,
            "dense_guidance_logits_lowres": guidance_lowres,
            "dense_guidance_map": self._maybe_resize_map(guidance_lowres, output_size),
            "refined_task_tokens": decoder_out.get("task_tokens"),
        }

    def forward(
        self,
        xs,
        ys,
        Ftext=None,
        domain_ids: torch.Tensor | None = None,
        xq=None,
        task_tokens_override=None,
    ):
        if xq is None:
            raise ValueError("xq must be provided")
        task_dict = self.encode_task(
            xs=xs,
            ys=ys,
            Ftext=Ftext,
            domain_ids=domain_ids,
            aggregate_support=self.aggregate_support_tokens,
            output_batch_size=xq.shape[0],
        )
        active_task_tokens = (
            task_dict["task_tokens"]
            if task_tokens_override is None
            else task_tokens_override
        )
        pred_dict = self.segment_with_task(xq=xq, task_tokens=active_task_tokens)
        return {
            **pred_dict,
            **task_dict,
            "task_tokens_used": active_task_tokens,
        }

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Tuple


# Every image, mask and pixel-distance metric uses one fixed grid.
DEFAULT_IMAGE_SIZE: Tuple[int, int] = (512, 512)
DEFAULT_METRIC_SIZE: Tuple[int, int] = (512, 512)

DOMAIN_NAMES: Tuple[str, ...] = (
    "thyroid",
    "TN3K",
    "BUSI_WHU",
    "BUS-BRA",
    "OTU",
    "prostate",
)
DOMAIN_TO_INDEX = {name: index for index, name in enumerate(DOMAIN_NAMES)}

# Five semantically distinct groups. The old generic "task" group was removed
# because almost identical wording across all domains increased redundancy.
TEXT_GROUP_NAMES: Tuple[str, ...] = (
    "target",
    "appearance",
    "boundary",
    "exclusion",
    "domain_identity",
)

# Dice-oriented domain-aware text configuration. BioBERT supplies medical
# semantics, while learnable domain/group residuals provide explicit closed-set
# discrimination when raw sentence embeddings occupy a highly anisotropic space.
DEFAULT_MODEL_CONFIG = {
    "use_ftext": True,
    "use_semantic_text": True,
    "hidden_dim": 1024,
    "num_query_tokens": 4,
    "context_upscale": 4,
    "dropout": 0.1,
    "text_dim": 1024,
    "text_group_summary_count": len(TEXT_GROUP_NAMES),
    "domain_names": DOMAIN_NAMES,
    "use_domain_embedding": True,
    "use_group_embedding": True,
    "domain_embedding_scale_init": 0.50,
    "group_embedding_scale_init": 0.25,
    "text_semantic_scale_init": 1.00,
    "task_norm_type": "group",
    "group_norm_groups": 16,
    "encoder_norm_type": "group",
    "fg_token_weight": 0.30,
    "skip_gate_mode": "centered",
    "aggregate_support_tokens": True,
    "use_visual_prototype": True,
    # TSV5 grouped-ablation switches. Both default to the original Full path.
    "use_task_specific_representation": True,
    "use_task_conditioned_decoder": True,
    "use_dense_text_guidance": True,
    "dense_guidance_dim": 256,
    "dense_guidance_strength": 0.50,
    "guidance_dropout_prob": 0.10,
    "guidance_scale_min": 0.80,
    "guidance_scale_max": 1.20,
    "guidance_clamp_max": 0.90,
    "guidance_stage_weights": (1.0, 0.50, 0.25, 0.0, 0.0),
    "guidance_modulation_mode": "centered",
    "use_deep_supervision": True,
}

DEFAULT_LOSS_CONFIG = {
    "ce": 0.50,
    "dice": 1.00,
    "focal_tversky": 0.25,
    "tversky_alpha": 0.30,
    "tversky_beta": 0.70,
    "tversky_gamma": 0.75,
    "dense": 0.15,
    # Aux heads are ordered from low to high resolution: up4, up3, up2.
    "aux_weights": (0.10, 0.15, 0.25),
}

DEFAULT_DOMAIN_THRESHOLDS = {
    "thyroid": 0.50,
    "TN3K": 0.50,
    "BUSI_WHU": 0.50,
    "BUS-BRA": 0.50,
    "OTU": 0.50,
    "prostate": 0.50,
    "default": 0.50,
}

DEFAULT_POSTPROCESS_CFG = {
    "min_area": 0,
    "min_area_ratio": 0.0,
    "keep_largest": False,
    "fill_holes": False,
    "closing_kernel": 0,
}

DEFAULT_POSTPROCESS_CFG_BY_DOMAIN = {
    "default": dict(DEFAULT_POSTPROCESS_CFG),
}

DEFAULT_EVAL_CONFIG = {
    "metric_size": DEFAULT_METRIC_SIZE,
    "use_postprocess": False,
}


@dataclass
class DataConfig:
    data_root: str = "./data"
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE
    domains: Tuple[str, ...] = DOMAIN_NAMES

    def roots(self) -> Dict[str, str]:
        return {domain: os.path.join(self.data_root, domain) for domain in self.domains}


@dataclass
class TrainConfig:
    num_epochs: int = 150
    batch_size: int = 4
    lr: float = 5e-5
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    patience: int = 20
    episodes_per_epoch: int = 1500
    grad_accum_steps: int = 4
    warmup_epochs: int = 5
    prototype_blend_target: float = 0.90
    prototype_warmup_epochs: int = 5
    prototype_ramp_epochs: int = 10
    prototype_refresh_samples: int = 128
    prototype_refresh_batch_size: int = 4
    prototype_refresh_interval: int = 1
    domain_sampling_alpha: float = 0.0
    use_amp: bool = True
    center_text_features: bool = True
    loss_config: dict = field(default_factory=lambda: dict(DEFAULT_LOSS_CONFIG))
    save_path: str = "USTS_DomainTextV3.pth"
    text_dir: str = "./text_features_llm"
    model_config: dict = field(default_factory=lambda: dict(DEFAULT_MODEL_CONFIG))

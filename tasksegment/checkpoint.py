from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, Union

import torch

from .configs import DEFAULT_MODEL_CONFIG
from .models.segmentation_model import TaskSegmentModel
from .training.memory import TaskPrototypeBank


def safe_torch_load(path: Union[str, Path], map_location="cpu") -> object:
    try:
        return torch.load(str(path), map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def migrate_model_config(raw_config: Dict[str, Any] | None) -> Dict[str, Any]:
    raw = dict(raw_config or {})
    cfg = dict(DEFAULT_MODEL_CONFIG)

    if "encoder_batchnorm" in raw and "encoder_norm_type" not in raw:
        raw["encoder_norm_type"] = "batch" if bool(raw["encoder_batchnorm"]) else "none"

    # Removed DiceV1 arguments. They are intentionally ignored rather than
    # passed into the compact DiceV2 model.
    for key in (
        "encoder_batchnorm",
        "decoder_num_classes",
        "text_token_mode",
        "max_text_tokens",
    ):
        raw.pop(key, None)

    cfg.update({key: value for key, value in raw.items() if key in cfg})
    cfg["guidance_stage_weights"] = tuple(float(value) for value in cfg["guidance_stage_weights"])
    cfg["domain_names"] = tuple(str(value) for value in cfg["domain_names"])
    return cfg


def load_checkpoint_model(
    model_path: Union[str, Path],
    device: torch.device,
    allow_partial_load: bool = False,
) -> Tuple[TaskSegmentModel, TaskPrototypeBank, Dict[str, Any], Dict[str, Any]]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path.resolve()}")
    checkpoint = safe_torch_load(path, map_location=device)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError("Checkpoint must contain model_state_dict")

    schema = int(checkpoint.get("schema_version", 0))
    if schema < 4 and not allow_partial_load:
        raise ValueError(
            "This checkpoint predates the domain-aware text architecture (schema 4). "
            "The current model adds learnable domain/group residual embeddings and must be retrained. "
            "Pass --allow-partial-load only for debugging, never for reported results."
        )

    model_config = migrate_model_config(checkpoint.get("model_config"))
    model = TaskSegmentModel(**model_config).to(device)
    if allow_partial_load:
        incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print(
                "[warning] partial checkpoint load: "
                f"missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}"
            )
    else:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    train_config = dict(checkpoint.get("train_config", {}))
    model.ablation_name = str(train_config.get("ablation_name", "full"))
    model.task_token_source = str(
        train_config.get(
            "task_token_source",
            "training_support_visual_prototype_domain_text",
        )
    )
    model.condition_token_source = str(
        train_config.get(
            "condition_token_source",
            "structured_semantic_text_plus_domain_group_embeddings",
        )
    )

    bank = TaskPrototypeBank()
    state = checkpoint.get("prototype_bank", checkpoint.get("ema_memory_bank", {}))
    if not isinstance(state, dict):
        raise ValueError("Checkpoint prototype bank must be a mapping")
    requires_bank = bool(getattr(model, "requires_visual_prototype_bank", True))
    if requires_bank and not state:
        raise ValueError(
            "Checkpoint does not contain a non-empty training-derived prototype bank "
            "required by this task-specific visual-prototype configuration"
        )
    if state:
        bank.load_state_dict(state)
    return model, bank, model_config, checkpoint

from __future__ import annotations

import copy
from contextlib import nullcontext
import csv
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
import torch.optim as optim

from ..configs import (
    DEFAULT_DOMAIN_THRESHOLDS,
    DEFAULT_LOSS_CONFIG,
    DEFAULT_METRIC_SIZE,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_POSTPROCESS_CFG_BY_DOMAIN,
)
from ..data.datasets import MultiOrganDataset
from ..text.bank import make_Ftext_batch
from .augmentations import augment_episode_medical
from .evaluation import evaluate
from .losses import deep_supervision_loss, dense_guidance_loss, segmentation_loss
from .memory import TaskPrototypeBank, make_domain_schedule
from .metrics import compute_hard_dice
from .prototypes import rebuild_visual_prototype_bank


def _make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _append_history(path: str, row: Dict[str, Any]) -> None:
    history_path = Path(path).with_suffix(Path(path).suffix + ".history.csv")
    write_header = not history_path.exists()
    with history_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _set_cosine_lr(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    num_epochs: int,
    base_lr: float,
    min_lr: float,
    warmup_epochs: int,
) -> float:
    if warmup_epochs > 0 and epoch < warmup_epochs:
        current = base_lr * float(epoch + 1) / float(warmup_epochs)
    else:
        remaining = max(1, num_epochs - warmup_epochs - 1)
        progress = min(max((epoch - warmup_epochs) / remaining, 0.0), 1.0)
        current = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = float(current)
    return float(current)


def _prototype_blend_for_epoch(
    epoch: int,
    warmup_epochs: int,
    ramp_epochs: int,
    target: float,
) -> float:
    if epoch < int(warmup_epochs):
        return 0.0
    ramp_epochs = max(1, int(ramp_epochs))
    progress = min(1.0, (epoch - int(warmup_epochs) + 1) / float(ramp_epochs))
    return float(min(max(target, 0.0), 1.0) * progress)


def _blend_with_prototype(
    model: torch.nn.Module,
    current_task_tokens: torch.Tensor,
    prototype_bank: TaskPrototypeBank,
    domain: str,
    blend: float,
) -> torch.Tensor:
    beta = float(min(max(blend, 0.0), 1.0))
    if beta <= 0.0 or not prototype_bank.has(domain):
        return current_task_tokens

    current_visual, current_text = model.split_task_tokens(current_task_tokens)
    bank_visual = prototype_bank.get(domain, device=current_task_tokens.device)
    if bank_visual is None:
        return current_task_tokens
    if bank_visual.shape[1] > model.visual_token_count:
        bank_visual = bank_visual[:, : model.visual_token_count]
    bank_visual = bank_visual.expand(current_visual.shape[0], -1, -1)
    mixed_visual = (1.0 - beta) * current_visual + beta * bank_visual.detach()
    return torch.cat([mixed_visual, current_text], dim=1) if current_text is not None else mixed_visual


def train(
    model: torch.nn.Module,
    train_dataset: MultiOrganDataset,
    val_dataset: MultiOrganDataset,
    device: torch.device,
    text_bank: Dict[str, torch.Tensor],
    num_epochs: int = 150,
    batch_size: int = 2,
    lr: float = 5e-5,
    min_lr: float = 1e-6,
    weight_decay: float = 1e-4,
    patience: int = 20,
    save_path: str = "USTS_DiceV2.pth",
    episodes_per_epoch: Optional[int] = 1500,
    grad_accum_steps: int = 4,
    model_config: Optional[Dict[str, Any]] = None,
    loss_config: Optional[Mapping[str, Any]] = None,
    warmup_epochs: int = 3,
    prototype_blend_target: float = 0.90,
    prototype_warmup_epochs: int = 5,
    prototype_ramp_epochs: int = 10,
    prototype_refresh_samples: int = 128,
    prototype_refresh_batch_size: int = 4,
    prototype_refresh_interval: int = 1,
    prototype_bank_blending_enabled: bool = False,
    prototype_eval_support_samples: int = 4,
    domain_sampling_alpha: float = 0.0,
    use_amp: bool = True,
    seed: int = 42,
    text_dir: str = "./text_features_llm",
    domain_thresholds: Optional[Dict[str, float]] = None,
    postprocess_cfg: Optional[Dict[str, Any]] = None,
    validation_use_postprocess: bool = False,
    validation_metric_size=DEFAULT_METRIC_SIZE,
    center_text_features: bool = True,
    ablation_name: str = "full",
) -> None:
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = _make_grad_scaler(amp_enabled)

    domains = sorted(train_dataset.organ_to_positive_indices)
    domain_sizes = {domain: len(train_dataset.organ_to_positive_indices[domain]) for domain in domains}
    best_val_dice = -1.0
    patience_counter = 0
    prototype_bank = TaskPrototypeBank()
    use_visual_prototype = bool(getattr(model, "use_visual_prototype", True))
    use_semantic_text = bool(getattr(model, "use_semantic_text", True))
    use_task_specific_representation = bool(
        getattr(model, "use_task_specific_representation", True)
    )
    use_task_conditioned_decoder = bool(
        getattr(model, "use_task_conditioned_decoder", True)
    )
    requires_visual_support = bool(
        getattr(model, "requires_visual_support", use_visual_prototype)
    )
    requires_condition_inputs = bool(
        getattr(model, "requires_condition_inputs", True)
    )
    requires_visual_prototype_bank = bool(
        getattr(model, "requires_visual_prototype_bank", requires_visual_support)
    )
    prototype_bank_blending_enabled = bool(prototype_bank_blending_enabled)
    prototype_manifest: Dict[str, Any] = {}

    modality_map = {
        "thyroid": "thyroid",
        "TN3K": "thyroid",
        "BUSI_WHU": "BUSI",
        "BUS-BRA": "BUSI",
        "OTU": "generic",
        "prostate": "generic",
    }
    model_config = dict(DEFAULT_MODEL_CONFIG if model_config is None else model_config)
    loss_cfg = dict(DEFAULT_LOSS_CONFIG if loss_config is None else loss_config)
    thresholds = dict(DEFAULT_DOMAIN_THRESHOLDS if domain_thresholds is None else domain_thresholds)
    pp_cfg = copy.deepcopy(DEFAULT_POSTPROCESS_CFG_BY_DOMAIN if postprocess_cfg is None else postprocess_cfg)

    print(f"训练域: {domains}")
    print(
        "DiceV2: 默认使用固定 K=4 训练支持视觉原型且不做 bank blending；"
        "可选 with_prototype_bank；Dice+CE+Focal-Tversky、多尺度深监督与 dense loss。"
    )

    for epoch in range(int(num_epochs)):
        current_lr = _set_cosine_lr(
            optimizer,
            epoch,
            int(num_epochs),
            float(lr),
            float(min_lr),
            int(warmup_epochs),
        )
        active_blend = (
            _prototype_blend_for_epoch(
                epoch,
                prototype_warmup_epochs,
                prototype_ramp_epochs,
                prototype_blend_target,
            )
            if requires_visual_prototype_bank and prototype_bank_blending_enabled
            else 0.0
        )

        model.train()
        total_iters = int(episodes_per_epoch) if episodes_per_epoch else max(
            1, len(train_dataset) // max(1, 2 * batch_size)
        )
        schedule = make_domain_schedule(
            domains,
            total_iters,
            domain_sizes=domain_sizes,
            alpha=domain_sampling_alpha,
        )
        optimizer.zero_grad(set_to_none=True)

        sums = {
            "loss": 0.0,
            "main": 0.0,
            "aux": 0.0,
            "dense": 0.0,
            "dice": 0.0,
        }

        for step, domain in enumerate(schedule):
            support_indices, query_indices = train_dataset.sample_episode_indices(
                domain,
                batch_size,
                batch_size,
            )
            if requires_visual_support:
                xs = torch.stack([train_dataset[index][0] for index in support_indices]).to(device, non_blocking=True)
                ys = torch.stack([train_dataset[index][1] for index in support_indices]).unsqueeze(1).float().to(device, non_blocking=True)
            else:
                # TSV5 grouped ablations skip support I/O when that information
                # path is disabled, while keeping the query training budget fixed.
                xs = None
                ys = None
            xq = torch.stack([train_dataset[index][0] for index in query_indices]).to(device, non_blocking=True)
            yq = torch.stack([train_dataset[index][1] for index in query_indices]).long().to(device, non_blocking=True)

            modality = modality_map.get(domain, "generic")
            if requires_visual_support:
                xs, ys = augment_episode_medical(xs, ys, modality=modality)
            xq, yq_aug = augment_episode_medical(xq, yq.unsqueeze(1).float(), modality=modality)
            yq = yq_aug.squeeze(1).long()

            organ_text = (
                text_bank.get(domain)
                if use_semantic_text and requires_condition_inputs
                else None
            )
            ftext = make_Ftext_batch(organ_text, batch_size) if organ_text is not None else None
            domain_ids = (
                model.domain_ids_from_names(
                    [domain] * xq.shape[0],
                    device=device,
                )
                if requires_condition_inputs
                else None
            )

            with _autocast(device, amp_enabled):
                task_dict = model.encode_task(
                    xs,
                    ys,
                    Ftext=ftext,
                    domain_ids=domain_ids,
                    aggregate_support=True,
                    output_batch_size=xq.shape[0],
                )
                active_tokens = task_dict.get("task_tokens")
                if active_tokens is not None and requires_visual_prototype_bank:
                    active_tokens = _blend_with_prototype(
                        model,
                        active_tokens,
                        prototype_bank,
                        domain,
                        active_blend,
                    )
                output = model.segment_with_task(xq=xq, task_tokens=active_tokens)
                output.update(task_dict)

                main_loss = segmentation_loss(output["pred_masks"], yq, loss_cfg)
                aux_loss = deep_supervision_loss(output.get("aux_logits"), yq, loss_cfg)
                dense_loss = dense_guidance_loss(output, yq.unsqueeze(1).float())
                loss = main_loss + aux_loss + float(loss_cfg.get("dense", 0.0)) * dense_loss

            scaler.scale(loss / max(1, int(grad_accum_steps))).backward()

            should_step = (
                (step + 1) % max(1, int(grad_accum_steps)) == 0
                or step + 1 == total_iters
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                batch_dice = float(compute_hard_dice(output["pred_masks"], yq).item())
            sums["loss"] += float(loss.detach().item())
            sums["main"] += float(main_loss.detach().item())
            sums["aux"] += float(aux_loss.detach().item())
            sums["dense"] += float(dense_loss.detach().item())
            sums["dice"] += batch_dice

        refresh_due = (
            not prototype_bank.bank
            or (epoch + 1) % max(1, int(prototype_refresh_interval)) == 0
        )
        if refresh_due and requires_visual_prototype_bank:
            prototype_bank, prototype_manifest = rebuild_visual_prototype_bank(
                model,
                train_dataset,
                device,
                samples_per_domain=prototype_refresh_samples,
                batch_size=prototype_refresh_batch_size,
                seed=seed,
            )

        val_metrics = evaluate(
            model=model,
            query_dataset=val_dataset,
            device=device,
            text_bank=text_bank,
            ema_memory_bank=prototype_bank,
            domain_thresholds=thresholds,
            postprocess_cfg=pp_cfg,
            use_postprocess=validation_use_postprocess,
            metric_size=validation_metric_size,
        )

        denom = max(1, total_iters)
        avg = {key: value / denom for key, value in sums.items()}
        improved = val_metrics["dice"] > best_val_dice
        if improved:
            best_val_dice = float(val_metrics["dice"])
            patience_counter = 0
            checkpoint = {
                "schema_version": 5,
                "architecture": "TaskSegment_DomainTextV3",
                "epoch": int(epoch + 1),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_dice": float(best_val_dice),
                "prototype_bank": prototype_bank.state_dict(),
                # Reader compatibility only; this is a deterministic prototype bank.
                "ema_memory_bank": prototype_bank.state_dict(),
                "prototype_manifest": prototype_manifest,
                "model_config": model_config,
                "train_config": {
                    "image_size": tuple(train_dataset.image_size),
                    "metric_size": tuple(int(value) for value in validation_metric_size),
                    "seed": int(seed),
                    "text_dir": str(text_dir),
                    "task_token_source": (
                        "unconditioned_query_decoder"
                        if not use_task_conditioned_decoder
                        else (
                            "shared_domain_agnostic_task_tokens"
                            if not use_task_specific_representation
                            else (
                                "deterministic_domain_prototype_bank_blending"
                                if requires_visual_prototype_bank and prototype_bank_blending_enabled
                                else (
                                    "episode_prototype_only_fixed_training_support_eval"
                                    if requires_visual_prototype_bank
                                    else "null_visual_tokens_domain_text_ablation"
                                )
                            )
                        )
                    ),
                    "ablation_name": str(ablation_name),
                    "condition_token_source": (
                        "not_used_unconditioned_decoder"
                        if not use_task_conditioned_decoder
                        else (
                            "shared_group_tokens_without_domain_or_semantic_signal"
                            if not use_task_specific_representation
                            else (
                                "zero_condition_tokens_no_text_or_domain_signal"
                                if str(ablation_name) == "no_text_conditioning"
                                else (
                                    "domain_group_embeddings_without_semantic_text"
                                    if not bool(use_semantic_text)
                                    else "structured_semantic_text_plus_domain_group_embeddings"
                                )
                            )
                        )
                    ),
                    "use_semantic_text": bool(use_semantic_text),
                    "use_visual_prototype": bool(use_visual_prototype),
                    "use_task_specific_representation": bool(use_task_specific_representation),
                    "use_task_conditioned_decoder": bool(use_task_conditioned_decoder),
                    "requires_visual_support": bool(requires_visual_support),
                    "requires_condition_inputs": bool(requires_condition_inputs),
                    "prototype_blend_target": float(prototype_blend_target),
                    "prototype_bank_blending_enabled": bool(
                        prototype_bank_blending_enabled
                    ),
                    "prototype_eval_support_samples": int(
                        prototype_eval_support_samples
                    ),
                    "prototype_warmup_epochs": int(prototype_warmup_epochs),
                    "prototype_ramp_epochs": int(prototype_ramp_epochs),
                    "prototype_refresh_samples": int(prototype_refresh_samples),
                    "prototype_refresh_interval": int(prototype_refresh_interval),
                    "domain_sampling_alpha": float(domain_sampling_alpha),
                    "warmup_epochs": int(warmup_epochs),
                    "lr": float(lr),
                    "min_lr": float(min_lr),
                    "loss_config": loss_cfg,
                    "domain_thresholds": thresholds,
                    "postprocess_cfg": pp_cfg,
                    "validation_use_postprocess": bool(validation_use_postprocess),
                    "validation_original_size": False,
                    "amp": bool(amp_enabled),
                    "center_text_features": bool(center_text_features),
                },
            }
            torch.save(checkpoint, save_path)
        else:
            patience_counter += 1

        row = {
            "epoch": epoch + 1,
            "train_loss": avg["loss"],
            "train_main_loss": avg["main"],
            "train_aux_loss": avg["aux"],
            "train_dense_loss": avg["dense"],
            "train_dice": avg["dice"],
            "val_macro_dice": val_metrics["dice"],
            "val_macro_iou": val_metrics["iou"],
            "val_macro_hd95": val_metrics["hd95"],
            "val_weighted_dice": val_metrics["weighted_dice"],
            "lr": current_lr,
            "prototype_blend": active_blend,
            "best_val_dice": best_val_dice,
        }
        _append_history(save_path, row)

        status = "saved" if improved else f"no_improve({patience_counter}/{patience})"
        print(
            f"Epoch {epoch+1:03d}/{num_epochs:03d} | loss={avg['loss']:.4f} "
            f"(main={avg['main']:.4f}, aux={avg['aux']:.4f}, dense={avg['dense']:.4f}) | "
            f"train_dice={avg['dice']:.4f} | val_dice={val_metrics['dice']:.4f} | "
            f"val_iou={val_metrics['iou']:.4f} | val_hd95={val_metrics['hd95']:.2f} | "
            f"blend={active_blend:.2f} | bank={int(prototype_bank_blending_enabled)} "
            f"| supportK={prototype_refresh_samples if requires_visual_prototype_bank else 0} "
            f"| task_specific={int(use_task_specific_representation)} "
            f"| task_decoder={int(use_task_conditioned_decoder)} | lr={current_lr:.2e} "
            f"| best={best_val_dice:.4f} | {status}",
            flush=True,
        )

        if not improved and patience_counter >= int(patience):
            print(f"Early stopping after epoch {epoch + 1}.")
            break

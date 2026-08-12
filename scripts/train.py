from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tasksegment.configs import (
    DataConfig,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_LOSS_CONFIG,
    DEFAULT_METRIC_SIZE,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_POSTPROCESS_CFG_BY_DOMAIN,
)
from tasksegment.data import get_datasets, set_seed
from tasksegment.models import TaskSegmentModel
from tasksegment.text import load_text_bank
from tasksegment.training import train


def _parse_image_size(values) -> Tuple[int, int]:
    if len(values) == 1:
        size = (int(values[0]), int(values[0]))
    elif len(values) == 2:
        size = (int(values[0]), int(values[1]))
    else:
        raise ValueError("--image-size accepts H or H W")
    if size != tuple(DEFAULT_IMAGE_SIZE):
        raise ValueError(
            f"This protocol requires all domains to use {DEFAULT_IMAGE_SIZE[0]}x{DEFAULT_IMAGE_SIZE[1]}; got {size}"
        )
    return size


def _parse_float_tuple(value: str, expected: int):
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if len(values) != expected:
        raise argparse.ArgumentTypeError(f"expected {expected} comma-separated values")
    return values


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train TaskSegment DomainTextV3 on the unified 512x512 protocol"
    )
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--text-dir", default="./text_features_llm")
    parser.add_argument("--save-path", default="USTS_DomainTextV3.pth")
    parser.add_argument("--image-size", type=int, nargs="+", default=list(DEFAULT_IMAGE_SIZE))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--ablation",
        choices=(
            "full",
            "with_prototype_bank",
            "no_text_conditioning",
            "no_semantic_text",
            "no_prototype_bank_blending",
            "no_visual_prototype",
            "no_dense_guidance",
            "no_task_specific_representation",
            "no_task_conditioned_decoder",
        ),
        default="full",
        help=(
            "TaskSegment variant. 'full' is the final no-bank K=4 model; "
            "'with_prototype_bank' enables the optional 128-support bank; "
            "'no_task_specific_representation' replaces support/text/domain task signals "
            "with shared domain-agnostic tokens; 'no_task_conditioned_decoder' disconnects "
            "all task-token injection from the query decoder; "
            "'no_prototype_bank_blending' is a deprecated alias of the new full model."
        ),
    )

    cache = parser.add_mutually_exclusive_group()
    cache.add_argument("--cache-dataset", dest="cache_dataset", action="store_true")
    cache.add_argument("--no-cache-dataset", dest="cache_dataset", action="store_false")
    parser.set_defaults(cache_dataset=True)
    parser.add_argument("--preload-train-cache", action="store_true")
    val_cache = parser.add_mutually_exclusive_group()
    val_cache.add_argument("--preload-val-cache", dest="preload_val_cache", action="store_true")
    val_cache.add_argument("--no-preload-val-cache", dest="preload_val_cache", action="store_false")
    parser.set_defaults(preload_val_cache=True)
    parser.add_argument("--cpu-clahe", action="store_true")

    text = parser.add_mutually_exclusive_group()
    text.add_argument("--use-ftext", dest="use_ftext", action="store_true")
    text.add_argument("--no-use-ftext", dest="use_ftext", action="store_false")
    parser.set_defaults(use_ftext=bool(DEFAULT_MODEL_CONFIG["use_ftext"]))
    parser.add_argument("--text-dim", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-query-tokens", type=int, default=4)
    parser.add_argument("--encoder-norm-type", choices=["group", "instance", "batch", "none"], default="group")

    center = parser.add_mutually_exclusive_group()
    center.add_argument("--center-text-features", dest="center_text_features", action="store_true")
    center.add_argument("--no-center-text-features", dest="center_text_features", action="store_false")
    parser.set_defaults(center_text_features=True)

    domain_condition = parser.add_mutually_exclusive_group()
    domain_condition.add_argument("--domain-embedding", dest="use_domain_embedding", action="store_true")
    domain_condition.add_argument("--no-domain-embedding", dest="use_domain_embedding", action="store_false")
    parser.set_defaults(use_domain_embedding=bool(DEFAULT_MODEL_CONFIG["use_domain_embedding"]))

    group_condition = parser.add_mutually_exclusive_group()
    group_condition.add_argument("--group-embedding", dest="use_group_embedding", action="store_true")
    group_condition.add_argument("--no-group-embedding", dest="use_group_embedding", action="store_false")
    parser.set_defaults(use_group_embedding=bool(DEFAULT_MODEL_CONFIG["use_group_embedding"]))

    parser.add_argument(
        "--domain-embedding-scale-init",
        type=float,
        default=DEFAULT_MODEL_CONFIG["domain_embedding_scale_init"],
    )
    parser.add_argument(
        "--group-embedding-scale-init",
        type=float,
        default=DEFAULT_MODEL_CONFIG["group_embedding_scale_init"],
    )
    parser.add_argument(
        "--text-semantic-scale-init",
        type=float,
        default=DEFAULT_MODEL_CONFIG["text_semantic_scale_init"],
    )

    parser.add_argument("--guidance-strength", type=float, default=DEFAULT_MODEL_CONFIG["dense_guidance_strength"])
    parser.add_argument("--guidance-dropout-prob", type=float, default=DEFAULT_MODEL_CONFIG["guidance_dropout_prob"])
    parser.add_argument("--guidance-scale-min", type=float, default=DEFAULT_MODEL_CONFIG["guidance_scale_min"])
    parser.add_argument("--guidance-scale-max", type=float, default=DEFAULT_MODEL_CONFIG["guidance_scale_max"])
    parser.add_argument("--guidance-clamp-max", type=float, default=DEFAULT_MODEL_CONFIG["guidance_clamp_max"])
    parser.add_argument(
        "--guidance-stage-weights",
        type=lambda value: _parse_float_tuple(value, 5),
        default=DEFAULT_MODEL_CONFIG["guidance_stage_weights"],
    )

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--episodes-per-epoch", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--train-query-positive-ratio", type=float, default=0.7)

    amp = parser.add_mutually_exclusive_group()
    amp.add_argument("--amp", dest="amp", action="store_true")
    amp.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)

    parser.add_argument("--ce-weight", type=float, default=DEFAULT_LOSS_CONFIG["ce"])
    parser.add_argument("--dice-weight", type=float, default=DEFAULT_LOSS_CONFIG["dice"])
    parser.add_argument("--focal-tversky-weight", type=float, default=DEFAULT_LOSS_CONFIG["focal_tversky"])
    parser.add_argument("--tversky-alpha", type=float, default=DEFAULT_LOSS_CONFIG["tversky_alpha"])
    parser.add_argument("--tversky-beta", type=float, default=DEFAULT_LOSS_CONFIG["tversky_beta"])
    parser.add_argument("--tversky-gamma", type=float, default=DEFAULT_LOSS_CONFIG["tversky_gamma"])
    parser.add_argument("--dense-loss-weight", type=float, default=DEFAULT_LOSS_CONFIG["dense"])
    parser.add_argument(
        "--aux-weights",
        type=lambda value: _parse_float_tuple(value, 3),
        default=DEFAULT_LOSS_CONFIG["aux_weights"],
        help="Deep-supervision weights for up4,up3,up2",
    )

    parser.add_argument("--prototype-blend-target", type=float, default=0.90)
    parser.add_argument("--prototype-warmup-epochs", type=int, default=5)
    parser.add_argument("--prototype-ramp-epochs", type=int, default=10)
    parser.add_argument("--prototype-refresh-samples", type=int, default=128)
    parser.add_argument(
        "--prototype-eval-support-samples",
        type=int,
        default=4,
        help=(
            "For the final no-bank Full and its ablations, use this many fixed "
            "training supports per domain for validation/test visual task tokens."
        ),
    )
    parser.add_argument("--prototype-refresh-batch-size", type=int, default=4)
    parser.add_argument("--prototype-refresh-interval", type=int, default=1)
    parser.add_argument(
        "--domain-sampling-alpha",
        type=float,
        default=0.0,
        help="0=domain-balanced; 1=sample-count proportional; 0.3 is a mild compromise",
    )
    parser.add_argument("--validation-postprocess", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed, args.deterministic)
    device = torch.device("cpu" if args.device == "cuda" and not torch.cuda.is_available() else args.device)
    image_size = _parse_image_size(args.image_size)
    metric_size = tuple(DEFAULT_METRIC_SIZE)

    if not Path(args.data_root).exists():
        raise FileNotFoundError(args.data_root)
    remove_all_text_conditioning = args.ablation == "no_text_conditioning"
    task_specific_representation_enabled = bool(
        args.ablation != "no_task_specific_representation"
    )
    task_conditioned_decoder_enabled = bool(
        args.ablation != "no_task_conditioned_decoder"
    )
    grouped_ablation_bypasses_external_task_inputs = bool(
        not task_specific_representation_enabled
        or not task_conditioned_decoder_enabled
    )
    condition_token_topology_enabled = bool(
        args.use_ftext or remove_all_text_conditioning
    )
    semantic_text_enabled = bool(
        args.use_ftext
        and args.ablation not in {"no_semantic_text", "no_text_conditioning"}
        and not grouped_ablation_bypasses_external_task_inputs
    )
    # The all-text ablation removes BioBERT semantics, closed-set domain IDs,
    # and text-group embeddings while preserving five zero condition tokens.
    # Keeping use_ftext=True preserves the condition-token topology and dense
    # guidance architecture without leaking any domain or language signal.
    effective_use_domain_embedding = bool(
        args.use_domain_embedding and not remove_all_text_conditioning
    )
    effective_use_group_embedding = bool(
        args.use_group_embedding and not remove_all_text_conditioning
    )
    effective_center_text_features = bool(
        args.center_text_features and not remove_all_text_conditioning
    )

    visual_prototype_enabled = bool(args.ablation != "no_visual_prototype")
    dense_guidance_module_enabled = bool(args.ablation != "no_dense_guidance")
    dense_guidance_active = bool(
        dense_guidance_module_enabled and task_conditioned_decoder_enabled
    )
    # Final Full is deliberately no-bank. The large domain prototype bank is
    # now an optional supplementary variant only.
    prototype_bank_blending_enabled = bool(args.ablation == "with_prototype_bank")
    prototype_support_samples = (
        int(args.prototype_refresh_samples)
        if prototype_bank_blending_enabled
        else int(args.prototype_eval_support_samples)
    )
    if prototype_support_samples <= 0:
        raise ValueError("prototype support sample count must be positive")

    if semantic_text_enabled and not Path(args.text_dir).exists():
        raise FileNotFoundError(args.text_dir)

    data_cfg = DataConfig(args.data_root, image_size)
    train_dataset, val_dataset, _ = get_datasets(
        data_cfg.roots(),
        image_size=image_size,
        seed=args.seed,
        train_query_positive_ratio=args.train_query_positive_ratio,
        cache_in_memory=args.cache_dataset,
        use_cpu_clahe=args.cpu_clahe,
    )
    if args.cache_dataset and args.preload_train_cache:
        train_dataset.warm_cache(show_progress=True)
    if args.cache_dataset and args.preload_val_cache:
        val_dataset.warm_cache(show_progress=True)

    text_bank = (
        load_text_bank(
            args.text_dir,
            device,
            expected_text_dim=args.text_dim,
            center_across_domains=effective_center_text_features,
        )
        if semantic_text_enabled
        else {}
    )

    model_config = dict(DEFAULT_MODEL_CONFIG)
    model_config.update(
        {
            "use_ftext": condition_token_topology_enabled,
            "use_semantic_text": semantic_text_enabled,
            "text_dim": args.text_dim,
            "hidden_dim": args.hidden_dim,
            "num_query_tokens": args.num_query_tokens,
            "encoder_norm_type": args.encoder_norm_type,
            "aggregate_support_tokens": True,
            "use_visual_prototype": visual_prototype_enabled,
            "use_task_specific_representation": task_specific_representation_enabled,
            "use_task_conditioned_decoder": task_conditioned_decoder_enabled,
            "use_domain_embedding": effective_use_domain_embedding,
            "use_group_embedding": effective_use_group_embedding,
            "domain_embedding_scale_init": args.domain_embedding_scale_init,
            "group_embedding_scale_init": args.group_embedding_scale_init,
            "text_semantic_scale_init": args.text_semantic_scale_init,
            "use_dense_text_guidance": dense_guidance_module_enabled,
            "dense_guidance_strength": args.guidance_strength,
            "guidance_dropout_prob": args.guidance_dropout_prob,
            "guidance_scale_min": args.guidance_scale_min,
            "guidance_scale_max": args.guidance_scale_max,
            "guidance_clamp_max": args.guidance_clamp_max,
            "guidance_stage_weights": tuple(args.guidance_stage_weights),
            "use_deep_supervision": True,
        }
    )
    loss_config = {
        "ce": args.ce_weight,
        "dice": args.dice_weight,
        "focal_tversky": args.focal_tversky_weight,
        "tversky_alpha": args.tversky_alpha,
        "tversky_beta": args.tversky_beta,
        "tversky_gamma": args.tversky_gamma,
        "dense": args.dense_loss_weight if dense_guidance_active else 0.0,
        "aux_weights": tuple(args.aux_weights),
    }

    model = TaskSegmentModel(**model_config).to(device)
    model.ablation_name = args.ablation

    print("=" * 88)
    print("TaskSegment DomainTextV3 training")
    print(
        f"device={device} train={len(train_dataset)} val={len(val_dataset)} "
        f"input_size={image_size} metric_size={metric_size}"
    )
    print(
        f"cache_dataset={args.cache_dataset} preload_train={args.preload_train_cache} "
        f"preload_val={args.preload_val_cache} cpu_clahe={args.cpu_clahe} amp={args.amp}"
    )
    print(f"parameters={sum(parameter.numel() for parameter in model.parameters()) / 1e6:.2f}M")
    print(f"loss_config={loss_config}")
    print(
        f"ablation={args.ablation} condition_tokens={condition_token_topology_enabled} "
        f"text_semantics={semantic_text_enabled} "
        f"visual_prototype={visual_prototype_enabled} "
        f"prototype_bank_blending={prototype_bank_blending_enabled} "
        f"prototype_support_samples={prototype_support_samples} "
        f"task_specific_representation={task_specific_representation_enabled} "
        f"task_conditioned_decoder={task_conditioned_decoder_enabled} "
        f"dense_guidance_module={dense_guidance_module_enabled} "
        f"dense_guidance_active={dense_guidance_active} "
        f"center_text={effective_center_text_features} "
        f"domain_embedding={effective_use_domain_embedding} "
        f"group_embedding={effective_use_group_embedding} "
        f"all_text_conditioning_removed={remove_all_text_conditioning}"
    )
    print(
        "Key protocol: fixed K=4 training-support visual prototypes by default; "
        "five structured condition tokens; deep supervision; fixed 512x512 metrics. "
        "The 128-support prototype bank is optional (with_prototype_bank)."
    )
    print("=" * 88)

    train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        text_bank=text_bank,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        save_path=args.save_path,
        episodes_per_epoch=args.episodes_per_epoch,
        grad_accum_steps=args.grad_accum_steps,
        model_config=model_config,
        loss_config=loss_config,
        warmup_epochs=args.warmup_epochs,
        prototype_blend_target=(
            args.prototype_blend_target
            if visual_prototype_enabled and prototype_bank_blending_enabled
            else 0.0
        ),
        prototype_warmup_epochs=args.prototype_warmup_epochs,
        prototype_ramp_epochs=args.prototype_ramp_epochs,
        prototype_refresh_samples=prototype_support_samples,
        prototype_bank_blending_enabled=prototype_bank_blending_enabled,
        prototype_eval_support_samples=args.prototype_eval_support_samples,
        prototype_refresh_batch_size=args.prototype_refresh_batch_size,
        prototype_refresh_interval=args.prototype_refresh_interval,
        domain_sampling_alpha=args.domain_sampling_alpha,
        use_amp=args.amp,
        seed=args.seed,
        text_dir=args.text_dir,
        postprocess_cfg=dict(DEFAULT_POSTPROCESS_CFG_BY_DOMAIN),
        validation_use_postprocess=args.validation_postprocess,
        validation_metric_size=metric_size,
        center_text_features=effective_center_text_features,
        ablation_name=args.ablation,
    )


if __name__ == "__main__":
    main()

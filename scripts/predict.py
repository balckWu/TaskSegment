from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tasksegment.checkpoint import load_checkpoint_model
from tasksegment.configs import (
    DataConfig,
    DEFAULT_DOMAIN_THRESHOLDS,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_METRIC_SIZE,
    DEFAULT_POSTPROCESS_CFG_BY_DOMAIN,
)
from tasksegment.data import MultiOrganDataset, get_image_transform, set_seed
from tasksegment.inference import predict_all_organs
from tasksegment.text import load_text_bank


def _parse_image_size(values) -> Tuple[int, int]:
    if len(values) == 1:
        size = (int(values[0]), int(values[0]))
    elif len(values) == 2:
        size = (int(values[0]), int(values[1]))
    else:
        raise ValueError("--image-size accepts H or H W")
    if size != tuple(DEFAULT_IMAGE_SIZE):
        raise ValueError(
            f"This paper protocol requires a unified {DEFAULT_IMAGE_SIZE[0]}x{DEFAULT_IMAGE_SIZE[1]} grid; got {size}."
        )
    return size


def _parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _load_json_mapping(path: str | None) -> Dict[str, Any] | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("postprocess JSON must contain an object")
    # Accept either a direct config or calibration payload.
    if isinstance(payload.get("postprocess_cfg"), dict):
        return dict(payload["postprocess_cfg"])
    return payload


def _apply_global_overrides(cfg: Mapping[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    if not overrides:
        return copy.deepcopy(dict(cfg))
    result = copy.deepcopy(dict(cfg))
    nested = any(isinstance(value, Mapping) for value in result.values())
    if not nested:
        result.update(overrides)
        return result

    default = dict(result.get("default", {}))
    default.update(overrides)
    result["default"] = default
    # Command-line overrides are explicitly global, including domains that have
    # a validation-calibrated entry in the checkpoint.
    for key, value in list(result.items()):
        if key == "default" or not isinstance(value, Mapping):
            continue
        domain_cfg = dict(value)
        domain_cfg.update(overrides)
        result[key] = domain_cfg
    return result


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate USTS on a unified 512x512 metric grid")
    p.add_argument("--model-path", default="USTS_DomainTextV3.pth")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--text-dir", default=None, help="Defaults to checkpoint train_config.text_dir")
    p.add_argument("--save-dir", default="./vis_dense_512")
    p.add_argument("--image-size", type=int, nargs="+", default=list(DEFAULT_IMAGE_SIZE))
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--ablation",
        default=None,
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
        help=(
            "Optional consistency check. The actual architecture is always restored "
            "from checkpoint model_config; a mismatch raises an error."
        ),
    )

    cache = p.add_mutually_exclusive_group()
    cache.add_argument("--cache-dataset", dest="cache_dataset", action="store_true")
    cache.add_argument("--no-cache-dataset", dest="cache_dataset", action="store_false")
    p.set_defaults(cache_dataset=True)

    preload = p.add_mutually_exclusive_group()
    preload.add_argument("--preload-cache", dest="preload_cache", action="store_true")
    preload.add_argument("--no-preload-cache", dest="preload_cache", action="store_false")
    p.set_defaults(preload_cache=True)

    p.add_argument("--num-vis-per-organ", type=int, default=5)
    p.add_argument("--vis-data-idx", default="")
    p.add_argument("--guidance-scale", type=float, default=None)
    p.add_argument("--threshold", type=float, default=None, help="Global diagnostic override; prefer validation calibration")
    p.add_argument("--postprocess", action="store_true", help="Use validation-frozen postprocessing as the primary variant")
    p.add_argument("--postprocess-config-json", default=None, help="Optional nested per-domain configuration JSON")
    p.add_argument("--min-area", type=int, default=None)
    p.add_argument("--min-area-ratio", type=float, default=None)
    p.add_argument("--closing-kernel", type=int, default=None)
    p.add_argument("--fill-holes", action="store_true")
    p.add_argument("--keep-largest", action="store_true")
    p.add_argument(
        "--allow-partial-load",
        action="store_true",
        help="Diagnostic only. Paper evaluation should never use this flag.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed, deterministic=True)
    device = torch.device("cpu" if args.device == "cuda" and not torch.cuda.is_available() else args.device)
    model, prototype_bank, model_config, checkpoint = load_checkpoint_model(
        args.model_path,
        device,
        allow_partial_load=args.allow_partial_load,
    )
    train_config = dict(checkpoint.get("train_config", {}))
    checkpoint_ablation = str(train_config.get("ablation_name", "full"))
    if args.ablation is not None and args.ablation != checkpoint_ablation:
        raise ValueError(
            "Ablation mismatch: "
            f"checkpoint={checkpoint_ablation}, command={args.ablation}"
        )

    image_size = _parse_image_size(args.image_size)
    metric_size = tuple(DEFAULT_METRIC_SIZE)
    text_dir = args.text_dir or train_config.get("text_dir", "./text_features_llm")
    thresholds = dict(train_config.get("domain_thresholds", DEFAULT_DOMAIN_THRESHOLDS))
    if args.threshold is not None:
        thresholds = {key: float(args.threshold) for key in DEFAULT_DOMAIN_THRESHOLDS}

    checkpoint_pp = train_config.get("postprocess_cfg", DEFAULT_POSTPROCESS_CFG_BY_DOMAIN)
    post_cfg = copy.deepcopy(checkpoint_pp)
    json_cfg = _load_json_mapping(args.postprocess_config_json)
    if json_cfg is not None:
        post_cfg = json_cfg
    overrides: Dict[str, Any] = {}
    if args.min_area is not None:
        overrides["min_area"] = int(args.min_area)
    if args.min_area_ratio is not None:
        overrides["min_area_ratio"] = float(args.min_area_ratio)
    if args.closing_kernel is not None:
        overrides["closing_kernel"] = int(args.closing_kernel)
    if args.fill_holes:
        overrides["fill_holes"] = True
    if args.keep_largest:
        overrides["keep_largest"] = True
    post_cfg = _apply_global_overrides(post_cfg, overrides)

    data_cfg = DataConfig(args.data_root, image_size)
    test_dataset = MultiOrganDataset(
        data_cfg.roots(),
        get_image_transform(),
        split="test",
        image_size=image_size,
        seed=args.seed,
        cache_in_memory=args.cache_dataset,
    )
    if args.cache_dataset and args.preload_cache:
        test_dataset.warm_cache(show_progress=True)

    center_text_features = bool(train_config.get("center_text_features", True))
    text_bank = (
        load_text_bank(
            text_dir,
            device,
            expected_text_dim=int(model_config["text_dim"]),
            center_across_domains=center_text_features,
        )
        if model_config["use_ftext"] and model_config.get("use_semantic_text", True)
        else {}
    )

    print("=" * 80)
    print("USTS evaluation")
    print(f"checkpoint={Path(args.model_path).resolve()}")
    print(
        f"test_samples={len(test_dataset)} input_resolution={image_size[0]}x{image_size[1]} "
        f"metric_resolution={metric_size[0]}x{metric_size[1]} strict_load={not args.allow_partial_load}"
    )
    print(f"cache_dataset={args.cache_dataset} preload_cache={args.preload_cache}")
    print(
        f"center_text_features={center_text_features} "
        f"domain_embedding={model_config.get('use_domain_embedding', False)} "
        f"group_embedding={model_config.get('use_group_embedding', False)}"
    )
    print(
        f"ablation={getattr(model, 'ablation_name', 'full')} "
        f"task_token_source={getattr(model, 'task_token_source', 'unknown')} "
        f"condition_token_source={getattr(model, 'condition_token_source', 'unknown')}"
    )
    print(f"primary_variant={'postprocess' if args.postprocess else 'raw'} thresholds={thresholds}")
    print(f"postprocess={post_cfg}")
    print("No test mask is used for prompting, prototype construction, threshold selection, or parameter selection.")
    print("All reported Dice/IoU/HD95 values are computed on the same 512x512 grid.")
    print("=" * 80)

    predict_all_organs(
        model=model,
        test_dataset=test_dataset,
        device=device,
        text_bank=text_bank,
        ema_bank=prototype_bank,
        save_dir=args.save_dir,
        domain_thresholds=thresholds,
        postprocess_cfg=post_cfg,
        use_postprocess=args.postprocess,
        num_vis_per_organ=args.num_vis_per_organ,
        vis_data_indices=_parse_int_list(args.vis_data_idx),
        guidance_scale=args.guidance_scale,
        report_both=True,
        metric_size=metric_size,
    )


if __name__ == "__main__":
    main()

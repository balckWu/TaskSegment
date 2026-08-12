from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class PaperConfig:
    path: Path
    root: Path
    payload: dict[str, Any]

    @property
    def protocol(self) -> dict[str, Any]:
        return dict(self.payload["protocol"])

    @property
    def paths(self) -> dict[str, Any]:
        return dict(self.payload.get("paths", {}))

    @property
    def artifacts_root(self) -> Path:
        return resolve_path(self.root, self.payload.get("artifacts_root", "./paper_artifacts"))

    @property
    def database_path(self) -> Path:
        return resolve_path(
            self.root,
            self.payload.get("database", "./paper_artifacts/experiment_results.sqlite"),
        )

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(int(seed) for seed in self.protocol["seeds"])

    @property
    def gpus(self) -> tuple[int, ...]:
        return tuple(int(gpu) for gpu in self.protocol["gpus"])

    @property
    def batch_size(self) -> int:
        return int(self.protocol["batch_size"])

    @property
    def episodes_per_epoch(self) -> int:
        return int(self.protocol["episodes_per_epoch"])

    # Backward-compatible name used by the runner and tests.
    @property
    def steps_per_epoch(self) -> int:
        return self.episodes_per_epoch

    @property
    def experiments(self) -> list[dict[str, Any]]:
        return [
            dict(experiment)
            for experiment in self.payload["experiments"]
            if bool(experiment.get("enabled", True))
        ]


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_config(path: str | Path) -> PaperConfig:
    config_path = Path(path).expanduser().resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    root = config_path.parents[1]
    config = PaperConfig(path=config_path, root=root, payload=payload)
    validate_config(config)
    return config


def validate_config(config: PaperConfig) -> None:
    protocol = config.protocol
    required = {
        "data_root",
        "seeds",
        "gpus",
        "batch_size",
        "val_batch_size",
        "grad_accum_steps",
        "effective_batch_size",
        "episodes_per_epoch",
        "epochs",
        "threshold",
        "metric_size",
        "prototype_blend_target",
        "prototype_warmup_epochs",
        "prototype_ramp_epochs",
        "prototype_refresh_samples",
        "prototype_eval_support_samples",
        "prototype_refresh_batch_size",
        "prototype_refresh_interval",
    }
    missing = sorted(required - set(protocol))
    if missing:
        raise ValueError(f"Missing protocol keys: {missing}")

    for key in (
        "batch_size",
        "val_batch_size",
        "grad_accum_steps",
        "episodes_per_epoch",
        "epochs",
        "prototype_warmup_epochs",
        "prototype_ramp_epochs",
        "prototype_refresh_samples",
        "prototype_eval_support_samples",
        "prototype_refresh_batch_size",
        "prototype_refresh_interval",
    ):
        if int(protocol[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    expected_effective = int(protocol["batch_size"]) * int(protocol["grad_accum_steps"])
    if int(protocol["effective_batch_size"]) != expected_effective:
        raise ValueError(
            "effective_batch_size mismatch: "
            f"expected {expected_effective}, got {protocol['effective_batch_size']}"
        )
    if len(set(config.seeds)) != len(config.seeds):
        raise ValueError("Seeds must be unique")
    if len(set(config.gpus)) != len(config.gpus):
        raise ValueError("GPUs must be unique")

    names: set[str] = set()
    for experiment in config.experiments:
        name = str(experiment["name"])
        if name in names:
            raise ValueError(f"Duplicate experiment name: {name}")
        names.add(name)
        allowed = tuple(int(gpu) for gpu in experiment.get("allowed_gpus", config.gpus))
        seed_gpu_map = {
            int(seed): int(gpu)
            for seed, gpu in dict(experiment.get("seed_gpu_map", {})).items()
        }
        unknown = sorted((set(allowed) | set(seed_gpu_map.values())) - set(config.gpus))
        if unknown:
            raise ValueError(f"Experiment {name} references unavailable GPUs: {unknown}")
        unknown_seeds = sorted(set(seed_gpu_map) - set(config.seeds))
        if unknown_seeds:
            raise ValueError(f"Experiment {name} maps unknown seeds: {unknown_seeds}")
        if not allowed and not seed_gpu_map:
            raise ValueError(f"No GPU allowed for experiment {name}")


def selected_experiments(
    config: PaperConfig,
    names: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    experiments = config.experiments
    if names is None:
        return experiments
    requested = {str(name) for name in names}
    result = [experiment for experiment in experiments if experiment["name"] in requested]
    missing = sorted(requested - {experiment["name"] for experiment in result})
    if missing:
        raise KeyError(f"Unknown or disabled experiments: {missing}")
    return result

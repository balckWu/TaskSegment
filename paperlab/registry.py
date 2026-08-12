from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import PaperConfig, resolve_path


@dataclass(frozen=True)
class Job:
    run_id: str
    experiment: str
    family: str
    variant: str
    seed: int
    env: str
    input_size: tuple[int, int]
    metric_size: tuple[int, int]
    batch_size: int
    val_batch_size: int
    grad_accum_steps: int
    effective_batch_size: int
    steps_per_epoch: int
    weight: float
    allowed_gpus: tuple[int, ...]
    preferred_gpu: int | None
    main_table: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["input_size"] = list(self.input_size)
        payload["metric_size"] = list(self.metric_size)
        payload["allowed_gpus"] = list(self.allowed_gpus)
        return payload


@dataclass(frozen=True)
class CommandSet:
    checkpoint: Path
    history: Path
    prediction_root: Path
    train: tuple[str, ...]
    predict: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": str(self.checkpoint),
            "history": str(self.history),
            "prediction_root": str(self.prediction_root),
            "train": list(self.train),
            "predict": list(self.predict),
        }


def expand_jobs(config: PaperConfig) -> list[Job]:
    protocol = config.protocol
    metric_size = tuple(int(v) for v in protocol["metric_size"])
    jobs: list[Job] = []
    for experiment in config.experiments:
        for seed in config.seeds:
            name = str(experiment["name"])
            jobs.append(
                Job(
                    run_id=f"{name}__seed{seed}",
                    experiment=name,
                    family=str(experiment["family"]),
                    variant=str(experiment["variant"]),
                    seed=int(seed),
                    env=str(experiment["env"]),
                    input_size=tuple(int(v) for v in experiment["input_size"]),
                    metric_size=metric_size,
                    batch_size=int(protocol["batch_size"]),
                    val_batch_size=int(protocol["val_batch_size"]),
                    grad_accum_steps=int(protocol["grad_accum_steps"]),
                    effective_batch_size=int(protocol["effective_batch_size"]),
                    steps_per_epoch=config.steps_per_epoch,
                    weight=float(experiment.get("weight", 1.0)),
                    allowed_gpus=tuple(
                        int(gpu) for gpu in experiment.get("allowed_gpus", config.gpus)
                    ),
                    preferred_gpu=(
                        int(dict(experiment.get("seed_gpu_map", {}))[str(seed)])
                        if str(seed) in dict(experiment.get("seed_gpu_map", {}))
                        else None
                    ),
                    main_table=bool(experiment.get("main_table", False)),
                )
            )
    return jobs


def run_directory(config: PaperConfig, job: Job) -> Path:
    return config.artifacts_root / "runs" / job.experiment / f"seed_{job.seed}"


def _python(script: Path, *args: str) -> tuple[str, ...]:
    return ("python", "-u", str(script), *[str(arg) for arg in args])


def _common_training_args(config: PaperConfig, job: Job) -> list[str]:
    protocol = config.protocol
    args = [
        "--data-root",
        str(resolve_path(config.root, protocol["data_root"])),
        "--device",
        "cuda:0",
        "--seed",
        str(job.seed),
        "--epochs",
        str(int(protocol["epochs"])),
        "--batch-size",
        str(job.batch_size),
        "--episodes-per-epoch",
        str(job.steps_per_epoch),
        "--grad-accum-steps",
        str(job.grad_accum_steps),
        "--min-lr",
        "1e-6",
        "--weight-decay",
        "1e-4",
        "--warmup-epochs",
        str(int(protocol["warmup_epochs"])),
        "--patience",
        str(int(protocol["patience"])),
        "--domain-sampling-alpha",
        str(float(protocol["domain_sampling_alpha"])),
        "--threshold",
        str(float(protocol["threshold"])),
        "--cache-dataset",
        "--preload-val-cache",
        "--augmentation",
        "--amp",
    ]
    if bool(protocol.get("deterministic", True)):
        args.append("--deterministic")
    return args


def build_commands(config: PaperConfig, job: Job) -> CommandSet:
    root = config.root
    protocol = config.protocol
    paths = config.paths
    run_dir = run_directory(config, job)
    checkpoint = run_dir / "checkpoint.pth"
    history = Path(str(checkpoint) + ".history.csv")
    prediction_root = run_dir / "predictions"
    data_root = resolve_path(root, protocol["data_root"])
    threshold = str(float(protocol["threshold"]))

    if job.family == "tasksegment":
        train = _python(
            root / "scripts" / "train.py",
            "--data-root",
            str(data_root),
            "--text-dir",
            str(resolve_path(root, protocol["text_dir"])),
            "--save-path",
            str(checkpoint),
            "--image-size",
            "512",
            "512",
            "--device",
            "cuda:0",
            "--seed",
            str(job.seed),
            "--deterministic",
            "--ablation",
            job.variant,
            "--epochs",
            str(int(protocol["epochs"])),
            "--batch-size",
            str(job.batch_size),
            "--episodes-per-epoch",
            str(job.steps_per_epoch),
            "--grad-accum-steps",
            str(job.grad_accum_steps),
            "--lr",
            "5e-5",
            "--min-lr",
            "1e-6",
            "--weight-decay",
            "1e-4",
            "--warmup-epochs",
            str(int(protocol["warmup_epochs"])),
            "--patience",
            str(int(protocol["patience"])),
            "--domain-sampling-alpha",
            str(float(protocol["domain_sampling_alpha"])),
            "--prototype-blend-target",
            str(float(protocol["prototype_blend_target"])),
            "--prototype-warmup-epochs",
            str(int(protocol["prototype_warmup_epochs"])),
            "--prototype-ramp-epochs",
            str(int(protocol["prototype_ramp_epochs"])),
            "--prototype-refresh-samples",
            str(int(protocol["prototype_refresh_samples"])),
            "--prototype-eval-support-samples",
            str(int(protocol["prototype_eval_support_samples"])),
            "--prototype-refresh-batch-size",
            str(int(protocol["prototype_refresh_batch_size"])),
            "--prototype-refresh-interval",
            str(int(protocol["prototype_refresh_interval"])),
            "--cache-dataset",
            "--preload-val-cache",
            "--amp",
        )
        predict = _python(
            root / "scripts" / "predict.py",
            "--model-path",
            str(checkpoint),
            "--data-root",
            str(data_root),
            "--text-dir",
            str(resolve_path(root, protocol["text_dir"])),
            "--save-dir",
            str(prediction_root),
            "--image-size",
            "512",
            "512",
            "--device",
            "cuda:0",
            "--seed",
            str(job.seed),
            "--cache-dataset",
            "--preload-cache",
            "--num-vis-per-organ",
            "0",
            "--threshold",
            threshold,
        )
        return CommandSet(checkpoint, history, prediction_root, train, predict)

    if job.family == "baseline":
        script_by_method = {
            "unet": "train_unet.py",
            "unetpp": "train_unetpp.py",
            "swinunet_official": "train_swinunet_official.py",
            "transunet_pretrained": "train_transunet_official.py",
            "transunet_scratch": "train_transunet_official.py",
        }
        script = root / "scripts" / script_by_method[job.variant]
        train_args = [
            "--save-path",
            str(checkpoint),
            *_common_training_args(config, job),
            "--val-batch-size",
            str(job.val_batch_size),
            "--lr",
            "1e-4",
            "--max-grad-norm",
            "5.0",
        ]
        if job.variant == "swinunet_official":
            train_args.append("--use-checkpoint")
        if job.variant == "unetpp":
            train_args.append("--deep-supervision")
        if job.variant == "transunet_pretrained":
            train_args.extend(
                [
                    "--pretrained-checkpoint",
                    str(resolve_path(root, paths["transunet_pretrained"])),
                ]
            )
        train = _python(script, *train_args)
        prediction_method = (
            "transunet_official"
            if job.variant in {"transunet_pretrained", "transunet_scratch"}
            else job.variant
        )
        predict = _python(
            root / "scripts" / "predict_baseline.py",
            "--method",
            prediction_method,
            "--checkpoint",
            str(checkpoint),
            "--data-root",
            str(data_root),
            "--output-dir",
            str(prediction_root),
            "--device",
            "cuda:0",
            "--seed",
            str(job.seed),
            "--batch-size",
            str(job.val_batch_size),
            "--threshold",
            threshold,
            "--amp",
            "--cache-dataset",
            "--preload-cache",
        )
        return CommandSet(checkpoint, history, prediction_root, train, predict)

    if job.family == "clipseg":
        base_model = resolve_path(root, paths["clipseg_model"])
        train = _python(
            root / "scripts" / "train_clipseg_official.py",
            "--data-root",
            str(data_root),
            "--save-path",
            str(checkpoint),
            "--prompt-file",
            str(root / "clipseg_prompts.json"),
            "--base-model",
            str(base_model),
            "--image-size",
            "512",
            "512",
            "--device",
            "cuda:0",
            "--seed",
            str(job.seed),
            "--deterministic",
            "--train-mode",
            "decoder",
            "--gradient-checkpointing",
            "--epochs",
            str(int(protocol["epochs"])),
            "--batch-size",
            str(job.batch_size),
            "--val-batch-size",
            str(job.val_batch_size),
            "--episodes-per-epoch",
            str(job.steps_per_epoch),
            "--grad-accum-steps",
            str(job.grad_accum_steps),
            "--decoder-lr",
            "1e-4",
            "--vision-lr",
            "1e-5",
            "--min-lr",
            "1e-6",
            "--weight-decay",
            "1e-4",
            "--warmup-epochs",
            str(int(protocol["warmup_epochs"])),
            "--patience",
            str(int(protocol["patience"])),
            "--max-grad-norm",
            "5.0",
            "--domain-sampling-alpha",
            str(float(protocol["domain_sampling_alpha"])),
            "--threshold",
            threshold,
            "--cache-dataset",
            "--preload-val-cache",
            "--augmentation",
            "--amp",
        )
        predict = _python(
            root / "scripts" / "predict_clipseg_official.py",
            "--checkpoint",
            str(checkpoint),
            "--data-root",
            str(data_root),
            "--output-root",
            str(prediction_root),
            "--device",
            "cuda:0",
            "--seed",
            str(job.seed),
            "--batch-size",
            str(job.val_batch_size),
            "--threshold",
            threshold,
            "--amp",
            "--cache-dataset",
            "--preload-cache",
            "--no-save-predictions",
        )
        return CommandSet(checkpoint, history, prediction_root, train, predict)

    if job.family == "lvit":
        train = _python(
            root / "scripts" / "train_lvit_official.py",
            "--data-root",
            str(data_root),
            "--official-repo",
            str(resolve_path(root, paths["lvit_repo"])),
            "--text-encoder",
            str(resolve_path(root, paths["lvit_text_encoder"])),
            "--prompt-file",
            str(root / "lvit_prompts.json"),
            "--save-path",
            str(checkpoint),
            "--model-resolution",
            "224",
            "--device",
            "cuda:0",
            "--seed",
            str(job.seed),
            "--deterministic",
            "--epochs",
            str(int(protocol["epochs"])),
            "--batch-size",
            str(job.batch_size),
            "--val-batch-size",
            str(job.val_batch_size),
            "--episodes-per-epoch",
            str(job.steps_per_epoch),
            "--grad-accum-steps",
            str(job.grad_accum_steps),
            "--lr",
            "1e-4",
            "--min-lr",
            "1e-6",
            "--weight-decay",
            "1e-4",
            "--warmup-epochs",
            str(int(protocol["warmup_epochs"])),
            "--patience",
            str(int(protocol["patience"])),
            "--max-grad-norm",
            "1.0",
            "--domain-sampling-alpha",
            str(float(protocol["domain_sampling_alpha"])),
            "--threshold",
            threshold,
            "--loss-profile",
            "controlled",
            "--amp-init-scale",
            "4096",
            "--amp-growth-interval",
            "4000",
            "--max-consecutive-amp-skips",
            "8",
            "--cache-dataset",
            "--preload-val-cache",
            "--augmentation",
            "--amp",
        )
        predict = _python(
            root / "scripts" / "predict_lvit_official.py",
            "--checkpoint",
            str(checkpoint),
            "--data-root",
            str(data_root),
            "--output-root",
            str(prediction_root),
            "--device",
            "cuda:0",
            "--batch-size",
            str(job.val_batch_size),
            "--seed",
            str(job.seed),
            "--amp",
            "--cache-dataset",
            "--preload-cache",
        )
        return CommandSet(checkpoint, history, prediction_root, train, predict)

    if job.family == "sammed2d":
        repo = resolve_path(root, paths["sammed2d_repo"])
        base_checkpoint = resolve_path(root, paths["sammed2d_checkpoint"])
        train = _python(
            root / "scripts" / "train_sammed2d_promptfree.py",
            "--data-root",
            str(data_root),
            "--official-repo",
            str(repo),
            "--base-checkpoint",
            str(base_checkpoint),
            "--save-path",
            str(checkpoint),
            "--device",
            "cuda:0",
            "--model-resolution",
            "512",
            "--seed",
            str(job.seed),
            "--deterministic",
            "--epochs",
            str(int(protocol["epochs"])),
            "--batch-size",
            str(job.batch_size),
            "--val-batch-size",
            str(job.val_batch_size),
            "--episodes-per-epoch",
            str(job.steps_per_epoch),
            "--grad-accum-steps",
            str(job.grad_accum_steps),
            "--lr",
            "1e-4",
            "--min-lr",
            "1e-6",
            "--weight-decay",
            "1e-4",
            "--warmup-epochs",
            str(int(protocol["warmup_epochs"])),
            "--patience",
            str(int(protocol["patience"])),
            "--max-grad-norm",
            "5.0",
            "--domain-sampling-alpha",
            str(float(protocol["domain_sampling_alpha"])),
            "--threshold",
            threshold,
            "--cache-dataset",
            "--preload-val-cache",
            "--augmentation",
            "--amp",
        )
        predict = _python(
            root / "scripts" / "predict_sammed2d.py",
            "--mode",
            "prompt_free",
            "--delta-checkpoint",
            str(checkpoint),
            "--data-root",
            str(data_root),
            "--official-repo",
            str(repo),
            "--base-checkpoint",
            str(base_checkpoint),
            "--output-root",
            str(prediction_root),
            "--device",
            "cuda:0",
            "--batch-size",
            str(job.val_batch_size),
            "--threshold",
            threshold,
            "--seed",
            str(job.seed),
            "--amp",
            "--cache-dataset",
            "--preload-cache",
        )
        return CommandSet(checkpoint, history, prediction_root, train, predict)

    raise KeyError(f"Unsupported experiment family: {job.family}")

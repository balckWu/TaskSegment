from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import PaperConfig, resolve_path
from .registry import Job, build_commands, expand_jobs, run_directory
from .results import (
    connect,
    ingest_run,
    register_run,
    set_project_metadata,
    set_run_state,
    status_counts,
)
from .provenance import system_manifest, write_manifest


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def create_plan(
    config: PaperConfig,
    selected_names: Iterable[str] | None = None,
    selected_seeds: Iterable[int] | None = None,
) -> dict[str, Any]:
    names = set(str(name) for name in selected_names) if selected_names else None
    seeds = set(int(seed) for seed in selected_seeds) if selected_seeds else None
    jobs = [
        job
        for job in expand_jobs(config)
        if (names is None or job.experiment in names)
        and (seeds is None or job.seed in seeds)
    ]
    if not jobs:
        raise ValueError("No jobs selected")

    loads = {gpu: 0.0 for gpu in config.gpus}
    assignments: dict[int, list[Job]] = {gpu: [] for gpu in config.gpus}
    # Longest jobs first; stable run_id tie-break makes the plan reproducible.
    for job in sorted(jobs, key=lambda item: (-item.weight, item.run_id)):
        if job.preferred_gpu is not None:
            candidates = [job.preferred_gpu]
        else:
            candidates = [gpu for gpu in job.allowed_gpus if gpu in loads]
        if not candidates:
            raise RuntimeError(f"No eligible GPU for {job.run_id}")
        gpu = min(candidates, key=lambda candidate: (loads[candidate], candidate))
        assignments[gpu].append(job)
        loads[gpu] += job.weight

    plan = {
        "schema_version": 1,
        "created_at": utc_now(),
        "config": str(config.path),
        "gpus": list(config.gpus),
        "estimated_load": {str(gpu): loads[gpu] for gpu in config.gpus},
        "assignments": {
            str(gpu): [job.run_id for job in assignments[gpu]]
            for gpu in config.gpus
        },
        "jobs": {
            job.run_id: {
                **job.to_dict(),
                "gpu": next(
                    gpu for gpu, queue in assignments.items() if job in queue
                ),
                "run_directory": str(run_directory(config, job)),
                "commands": build_commands(config, job).to_dict(),
            }
            for job in jobs
        },
    }
    config.artifacts_root.mkdir(parents=True, exist_ok=True)
    provenance = system_manifest(config.root, config.path)
    write_manifest(config.artifacts_root / "provenance.json", provenance)
    atomic_json(config.artifacts_root / "config_snapshot.json", config.payload)
    atomic_json(config.artifacts_root / "plan.json", plan)
    write_plan_tsv(config.artifacts_root / "plan.tsv", plan)

    connection = connect(config.database_path)
    try:
        set_project_metadata(connection, "config", config.payload)
        set_project_metadata(connection, "provenance", provenance)
        set_project_metadata(connection, "plan", plan)
        for job in jobs:
            gpu = int(plan["jobs"][job.run_id]["gpu"])
            register_run(connection, config, job, gpu, "planned")
    finally:
        connection.close()
    return plan


def write_plan_tsv(path: Path, plan: dict[str, Any]) -> None:
    lines = [
        "gpu\trun_id\texperiment\tseed\tenv\tbatch_size\teffective_batch\ttrain_command\tpredict_command"
    ]
    for gpu in plan["gpus"]:
        for run_id in plan["assignments"][str(gpu)]:
            item = plan["jobs"][run_id]
            command = item["commands"]
            lines.append(
                "\t".join(
                    [
                        str(gpu),
                        run_id,
                        str(item["experiment"]),
                        str(item["seed"]),
                        str(item["env"]),
                        str(item["batch_size"]),
                        str(item["effective_batch_size"]),
                        shlex.join(command["train"]),
                        shlex.join(command["predict"]),
                    ]
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_plan(config: PaperConfig) -> dict[str, Any]:
    path = config.artifacts_root / "plan.json"
    if not path.exists():
        return create_plan(config)
    return json.loads(path.read_text(encoding="utf-8"))


def jobs_by_id(config: PaperConfig) -> dict[str, Job]:
    return {job.run_id: job for job in expand_jobs(config)}


def preflight(config: PaperConfig, strict: bool = True) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def check(name: str, path: Path, required: bool = True) -> None:
        exists = path.exists()
        checks.append(
            {
                "name": name,
                "path": str(path),
                "required": required,
                "ok": exists or not required,
            }
        )

    protocol = config.protocol
    paths = config.paths
    check("data_root", resolve_path(config.root, protocol["data_root"]))
    check("tasksegment_data_source", config.root / "tasksegment" / "data" / "datasets.py")
    check("text_features", resolve_path(config.root, protocol["text_dir"]))
    for script in (
        "train.py",
        "predict.py",
        "train_unet.py",
        "train_unetpp.py",
        "train_swinunet_official.py",
        "train_transunet_official.py",
        "train_clipseg_official.py",
        "predict_clipseg_official.py",
        "train_lvit_official.py",
        "predict_lvit_official.py",
        "train_sammed2d_promptfree.py",
        "predict_sammed2d.py",
        "predict_baseline.py",
    ):
        check(f"script:{script}", config.root / "scripts" / script)

    clipseg = resolve_path(config.root, paths["clipseg_model"])
    check("clipseg:config", clipseg / "config.json")
    check("clipseg:preprocessor", clipseg / "preprocessor_config.json")
    check("clipseg:tokenizer_config", clipseg / "tokenizer_config.json")
    check("clipseg:vocab", clipseg / "vocab.json")
    check("clipseg:merges", clipseg / "merges.txt")
    clipseg_weights = (clipseg / "model.safetensors").exists() or (
        clipseg / "pytorch_model.bin"
    ).exists()
    checks.append(
        {
            "name": "clipseg:weights",
            "path": str(clipseg),
            "required": True,
            "ok": clipseg_weights,
        }
    )

    lvit_repo = resolve_path(config.root, paths["lvit_repo"])
    check("lvit:official_repo", lvit_repo / "nets" / "LViT.py")
    lvit_text_encoder = resolve_path(config.root, paths["lvit_text_encoder"])
    check("lvit:text_encoder_config", lvit_text_encoder / "config.json")
    check("lvit:text_encoder_tokenizer", lvit_text_encoder / "tokenizer_config.json")
    sam_repo = resolve_path(config.root, paths["sammed2d_repo"])
    check("sammed2d:official_repo", sam_repo / "segment_anything" / "build_sam.py")
    check(
        "sammed2d:checkpoint",
        resolve_path(config.root, paths["sammed2d_checkpoint"]),
    )
    check(
        "swinunet:official_repo",
        config.root / "third_party" / "Swin-Unet" / "networks" / "vision_transformer.py",
    )
    check(
        "transunet:official_repo",
        config.root / "third_party" / "TransUNet" / "networks" / "vit_seg_modeling.py",
    )
    transunet_pretrained_enabled = any(
        experiment.get("variant") == "transunet_pretrained"
        for experiment in config.experiments
    )
    if transunet_pretrained_enabled:
        transunet_checkpoint = resolve_path(
            config.root,
            paths["transunet_pretrained"],
        )
        check(
            "transunet:imagenet21k_checkpoint",
            transunet_checkpoint,
        )
        checkpoint_members_ok = False
        checkpoint_detail = str(transunet_checkpoint)
        if transunet_checkpoint.is_file():
            required_members = {
                "embedding/kernel.npy",
                "embedding/bias.npy",
                "Transformer/encoder_norm/scale.npy",
                "Transformer/posembed_input/pos_embedding.npy",
                "conv_root/kernel.npy",
                "gn_root/scale.npy",
            }
            try:
                with zipfile.ZipFile(transunet_checkpoint) as archive:
                    members = set(archive.namelist())
                missing_members = sorted(required_members - members)
                checkpoint_members_ok = not missing_members
                if missing_members:
                    checkpoint_detail += "; missing=" + ",".join(missing_members)
            except Exception as error:
                checkpoint_detail += f"; invalid_npz={error}"
        checks.append(
            {
                "name": "transunet:imagenet21k_checkpoint_keys",
                "path": checkpoint_detail,
                "required": True,
                "ok": checkpoint_members_ok,
            }
        )
    for repository in ("LViT", "SAM-Med2D", "Swin-Unet", "TransUNet"):
        check(
            f"official_commit:{repository}",
            config.root / "third_party" / repository / "OFFICIAL_COMMIT_USED.txt",
        )

    conda = shutil.which("conda")
    checks.append(
        {
            "name": "conda",
            "path": conda or "not found",
            "required": True,
            "ok": conda is not None,
        }
    )
    if conda is not None:
        try:
            payload = json.loads(
                subprocess.check_output([conda, "env", "list", "--json"], text=True)
            )
            names = {Path(path).name for path in payload.get("envs", [])}
        except Exception:
            names = set()
        required_imports = {
            "TS": ["torch", "numpy", "cv2", "transformers", "timm", "ml_collections"],
            "TS_LVIT": ["torch", "numpy", "transformers", "timm", "ml_collections"],
            "TS_SAMMED2D": ["torch", "numpy", "cv2", "scipy", "PIL"],
        }
        for env in sorted({experiment["env"] for experiment in config.experiments}):
            env_exists = env in names
            checks.append(
                {
                    "name": f"conda_env:{env}",
                    "path": env,
                    "required": True,
                    "ok": env_exists,
                }
            )
            modules = required_imports.get(env, ["torch", "numpy"])
            import_ok = False
            detail = ",".join(modules)
            if env_exists:
                try:
                    subprocess.check_output(
                        [
                            conda,
                            "run",
                            "--no-capture-output",
                            "-n",
                            env,
                            "python",
                            "-c",
                            ";".join(f"import {module}" for module in modules),
                        ],
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=120,
                    )
                    import_ok = True
                except Exception as error:
                    detail = f"{detail}; {error}"
            checks.append(
                {
                    "name": f"imports:{env}",
                    "path": detail,
                    "required": True,
                    "ok": import_ok,
                }
            )

    gpu_output = None
    try:
        gpu_output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=20,
        )
    except Exception:
        pass
    visible_gpu_indices = set()
    if gpu_output:
        for line in gpu_output.splitlines():
            try:
                visible_gpu_indices.add(int(line.split(",", 1)[0].strip()))
            except (ValueError, IndexError):
                continue
    for gpu in config.gpus:
        checks.append(
            {
                "name": f"gpu:{gpu}",
                "path": f"physical GPU {gpu}",
                "required": True,
                "ok": gpu in visible_gpu_indices,
            }
        )

    report_path = config.artifacts_root / "preflight.json"
    atomic_json(report_path, {"created_at": utc_now(), "checks": checks})
    failures = [item for item in checks if item["required"] and not item["ok"]]
    if strict and failures:
        message = "\n".join(f"- {item['name']}: {item['path']}" for item in failures)
        raise RuntimeError(f"Preflight failed:\n{message}")
    return checks


def reset_artifacts(config: PaperConfig) -> None:
    root = config.artifacts_root
    if root.exists():
        resolved = root.resolve()
        if resolved == config.root.resolve() or config.root.resolve() not in resolved.parents:
            raise RuntimeError(f"Refusing to delete unsafe artifacts path: {resolved}")
        shutil.rmtree(resolved)
    root.mkdir(parents=True, exist_ok=True)


def _conda_command(env_name: str, command: tuple[str, ...]) -> list[str]:
    conda = shutil.which("conda") or "conda"
    return [conda, "run", "--no-capture-output", "-n", env_name, *command]


def _run_command(
    job: Job,
    gpu: int,
    command: tuple[str, ...],
    log_path: Path,
    cwd: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            "PAPERLAB_PROGRESS": "off",
            "PAPERLAB_PROGRESS_MININTERVAL": "5.0",
            "PYTORCH_CUDA_ALLOC_CONF": environment.get(
                "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
            ),
            "OMP_NUM_THREADS": environment.get("OMP_NUM_THREADS", "4"),
            "MKL_NUM_THREADS": environment.get("MKL_NUM_THREADS", "4"),
        }
    )
    full_command = _conda_command(job.env, command)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# started_at={utc_now()}\n")
        handle.write(f"# physical_gpu={gpu}; logical_device=cuda:0\n")
        handle.write(f"# command={shlex.join(full_command)}\n")
        handle.flush()
        process = subprocess.Popen(
            full_command,
            cwd=cwd,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return_code = process.wait()
        handle.write(f"# finished_at={utc_now()} return_code={return_code}\n")
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, full_command)


def _log_finished_successfully(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        tail = path.read_text(encoding="utf-8", errors="replace")[-4096:]
    except OSError:
        return False
    return "return_code=0" in tail


def _archive_incomplete_run(run_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = run_dir.with_name(f"{run_dir.name}_incomplete_{stamp}")
    counter = 1
    while archive.exists():
        archive = run_dir.with_name(f"{run_dir.name}_incomplete_{stamp}_{counter}")
        counter += 1
    run_dir.replace(archive)
    return archive


def run_job(
    config: PaperConfig,
    job: Job,
    gpu: int,
    *,
    force: bool = False,
) -> None:
    run_dir = run_directory(config, job)
    complete_path = run_dir / "completed.json"
    if complete_path.exists() and not force:
        connection = connect(config.database_path)
        try:
            register_run(connection, config, job, gpu, "completed")
            ingest_run(connection, config, job)
        finally:
            connection.close()
        return

    if force and run_dir.exists():
        shutil.rmtree(run_dir)

    commands = build_commands(config, job)
    train_log = run_dir / "train.log"
    prediction_log = run_dir / "predict.log"
    training_complete = (
        commands.checkpoint.exists()
        and commands.history.exists()
        and _log_finished_successfully(train_log)
    )

    # A predict-only retry reuses a fully completed training run. A partially
    # written training run is archived and restarted, never silently appended.
    if run_dir.exists() and not force and not training_complete:
        _archive_incomplete_run(run_dir)
        training_complete = False

    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(
        run_dir / "resolved_run.json",
        {
            "created_at": utc_now(),
            "gpu": gpu,
            "job": job.to_dict(),
            "commands": commands.to_dict(),
            "train_shell": shlex.join(_conda_command(job.env, commands.train)),
            "predict_shell": shlex.join(_conda_command(job.env, commands.predict)),
        },
    )

    connection = connect(config.database_path)
    try:
        register_run(connection, config, job, gpu, "training")
        set_run_state(
            connection,
            job.run_id,
            "training",
            timestamp_field="train_started_at",
        )
    finally:
        connection.close()

    try:
        if not training_complete:
            _run_command(job, gpu, commands.train, train_log, config.root)
            if not commands.checkpoint.exists():
                raise FileNotFoundError(
                    f"Training completed without checkpoint: {commands.checkpoint}"
                )

        connection = connect(config.database_path)
        try:
            set_run_state(
                connection,
                job.run_id,
                "predicting",
                timestamp_field="train_finished_at",
            )
        finally:
            connection.close()

        _run_command(job, gpu, commands.predict, prediction_log, config.root)

        connection = connect(config.database_path)
        try:
            counts = ingest_run(connection, config, job)
            set_run_state(
                connection,
                job.run_id,
                "completed",
                timestamp_field="predict_finished_at",
            )
        finally:
            connection.close()
        atomic_json(
            complete_path,
            {
                "run_id": job.run_id,
                "completed_at": utc_now(),
                "gpu": gpu,
                "checkpoint": str(commands.checkpoint),
                "ingested": counts,
            },
        )
    except Exception as error:
        error_text = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )
        (run_dir / "FAILED.txt").write_text(error_text, encoding="utf-8")
        connection = connect(config.database_path)
        try:
            set_run_state(connection, job.run_id, "failed", error=error_text[-16000:])
        finally:
            connection.close()
        raise


def worker(
    config: PaperConfig,
    gpu: int,
    *,
    force: bool = False,
) -> None:
    plan = load_plan(config)
    mapping = jobs_by_id(config)
    run_ids = plan["assignments"].get(str(gpu), [])
    scheduler_dir = config.artifacts_root / "scheduler"
    scheduler_dir.mkdir(parents=True, exist_ok=True)
    worker_state = scheduler_dir / f"gpu{gpu}.json"
    atomic_json(
        worker_state,
        {
            "gpu": gpu,
            "pid": os.getpid(),
            "status": "running",
            "started_at": utc_now(),
            "jobs": run_ids,
        },
    )
    failures: list[str] = []
    for position, run_id in enumerate(run_ids, start=1):
        job = mapping[run_id]
        atomic_json(
            worker_state,
            {
                "gpu": gpu,
                "pid": os.getpid(),
                "status": "running",
                "started_at": utc_now(),
                "current": run_id,
                "position": position,
                "total": len(run_ids),
                "failures": failures,
            },
        )
        try:
            run_job(config, job, gpu, force=force)
        except Exception:
            failures.append(run_id)
    atomic_json(
        worker_state,
        {
            "gpu": gpu,
            "pid": os.getpid(),
            "status": "finished" if not failures else "finished_with_failures",
            "finished_at": utc_now(),
            "jobs": run_ids,
            "failures": failures,
        },
    )
    if failures:
        raise RuntimeError(f"GPU {gpu} failed jobs: {failures}")



def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def _launch_monitor_detached(config: PaperConfig, script: Path) -> int:
    scheduler_dir = config.artifacts_root / "scheduler"
    scheduler_dir.mkdir(parents=True, exist_ok=True)
    pid_path = scheduler_dir / "monitor.pid"
    if pid_path.exists():
        try:
            existing = int(pid_path.read_text(encoding="utf-8").strip())
        except Exception:
            existing = -1
        if _pid_is_alive(existing):
            return existing

    interval = float(config.protocol.get("monitor_interval_seconds", 30))
    command = [
        sys.executable,
        "-u",
        str(script),
        "monitor",
        "--config",
        str(config.path),
        "--interval",
        str(interval),
    ]
    log_path = scheduler_dir / "monitor.log"
    handle = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=config.root,
        stdout=handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    handle.close()
    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    return int(process.pid)

def launch_detached(
    config: PaperConfig,
    *,
    force: bool = False,
) -> dict[int, int]:
    plan = load_plan(config)
    script = config.root / "scripts" / "paper.py"
    scheduler_dir = config.artifacts_root / "scheduler"
    scheduler_dir.mkdir(parents=True, exist_ok=True)
    pids: dict[int, int] = {}
    for gpu in plan["gpus"]:
        if not plan["assignments"].get(str(gpu)):
            continue
        command = [
            sys.executable,
            "-u",
            str(script),
            "worker",
            "--config",
            str(config.path),
            "--gpu",
            str(gpu),
        ]
        if force:
            command.append("--force")
        log_path = scheduler_dir / f"gpu{gpu}.log"
        handle = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=config.root,
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        handle.close()
        pids[int(gpu)] = process.pid
        (scheduler_dir / f"gpu{gpu}.pid").write_text(f"{process.pid}\n", encoding="utf-8")
    monitor_pid = _launch_monitor_detached(config, script)
    atomic_json(
        scheduler_dir / "launcher.json",
        {
            "launched_at": utc_now(),
            "pids": pids,
            "monitor_pid": monitor_pid,
            "config": str(config.path),
        },
    )
    return pids


def collect_all(config: PaperConfig) -> dict[str, Any]:
    connection = connect(config.database_path)
    successes: list[str] = []
    failures: dict[str, str] = {}
    try:
        for job in expand_jobs(config):
            register_run(connection, config, job, None, "collecting")
            try:
                ingest_run(connection, config, job)
                set_run_state(connection, job.run_id, "completed")
                successes.append(job.run_id)
            except Exception as error:
                failures[job.run_id] = str(error)
        counts = status_counts(connection)
    finally:
        connection.close()
    return {"successes": successes, "failures": failures, "status_counts": counts}


def status_report(config: PaperConfig) -> dict[str, Any]:
    plan = load_plan(config)
    connection = connect(config.database_path)
    try:
        rows = [dict(row) for row in connection.execute(
            "SELECT run_id, experiment, seed, gpu, status, error FROM runs ORDER BY experiment, seed"
        ).fetchall()]
        counts = status_counts(connection)
    finally:
        connection.close()
    workers = {}
    scheduler_dir = config.artifacts_root / "scheduler"
    for gpu in config.gpus:
        path = scheduler_dir / f"gpu{gpu}.json"
        if path.exists():
            workers[str(gpu)] = json.loads(path.read_text(encoding="utf-8"))
    return {"counts": counts, "runs": rows, "workers": workers, "plan": plan}

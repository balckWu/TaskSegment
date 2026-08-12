from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .config import PaperConfig
from .registry import build_commands, expand_jobs, run_directory
from .runner import load_plan, status_report


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _read_history(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except Exception:
        return []


def _first_float(row: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _first_int(row: Mapping[str, Any], *keys: str) -> int | None:
    value = _first_float(row, *keys)
    return None if value is None else int(value)


def _fmt(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _fmt_sci(value: float | None) -> str:
    return "-" if value is None else f"{value:.2e}"


def _alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def build_snapshot(config: PaperConfig) -> dict[str, Any]:
    plan = load_plan(config)
    status = status_report(config)
    db_rows = {str(row["run_id"]): dict(row) for row in status["runs"]}
    jobs = {job.run_id: job for job in expand_jobs(config)}
    worker_states = status["workers"]
    records: list[dict[str, Any]] = []

    for gpu in plan["gpus"]:
        queue = plan["assignments"].get(str(gpu), [])
        for queue_position, run_id in enumerate(queue, start=1):
            job = jobs[run_id]
            commands = build_commands(config, job)
            run_dir = run_directory(config, job)
            history = _read_history(commands.history)
            latest = history[-1] if history else {}
            db_row = db_rows.get(run_id, {})
            current = worker_states.get(str(gpu), {}).get("current") == run_id

            if (run_dir / "completed.json").exists():
                phase = "completed"
            elif (run_dir / "FAILED.txt").exists():
                phase = "failed"
            elif current:
                database_phase = str(db_row.get("status") or "")
                phase = "predicting" if database_phase == "predicting" else "training"
            else:
                phase = str(db_row.get("status") or "planned")

            records.append(
                {
                    "gpu": int(gpu),
                    "queue_position": queue_position,
                    "queue_total": len(queue),
                    "run_id": run_id,
                    "experiment": job.experiment,
                    "seed": job.seed,
                    "phase": phase,
                    "current": bool(current),
                    "epoch": _first_int(latest, "epoch"),
                    "epochs_total": int(config.protocol["epochs"]),
                    "train_loss": _first_float(latest, "train_loss", "loss"),
                    "train_dice": _first_float(latest, "train_dice", "dice"),
                    "val_dice": _first_float(latest, "val_macro_dice", "macro_dice", "val_dice"),
                    "val_iou": _first_float(latest, "val_macro_iou", "macro_iou", "val_iou"),
                    "val_hd95": _first_float(latest, "val_macro_hd95", "macro_hd95", "val_hd95"),
                    "best_val_dice": _first_float(latest, "best_val_dice"),
                    "lr": _first_float(latest, "lr"),
                    "epoch_seconds": _first_float(latest, "epoch_seconds", "time"),
                    "history_path": str(commands.history),
                    "run_directory": str(run_dir),
                    "error": str(db_row.get("error") or ""),
                }
            )

    counts: dict[str, int] = {}
    for record in records:
        counts[record["phase"]] = counts.get(record["phase"], 0) + 1
    return {
        "updated_at": utc_now(),
        "counts": counts,
        "workers": worker_states,
        "records": records,
        "plan": plan,
    }


def render_snapshot(snapshot: Mapping[str, Any], *, show_all: bool = False) -> str:
    records = list(snapshot["records"])
    current_by_gpu = {record["gpu"]: record for record in records if record["current"]}
    lines = [
        "=" * 138,
        f"TaskSegment paper training dashboard | updated {snapshot['updated_at']}",
        "Run counts: " + ", ".join(
            f"{key}={value}" for key, value in sorted(snapshot["counts"].items())
        ),
        "=" * 138,
        "GPU  STATE       CURRENT RUN                                EPOCH       LOSS    TRAIN_D   VAL_D    BEST_D   HD95     LR        QUEUE",
        "-" * 138,
    ]
    for gpu in snapshot["plan"]["gpus"]:
        record = current_by_gpu.get(int(gpu))
        worker = snapshot["workers"].get(str(gpu), {})
        if record is None:
            state = str(worker.get("status") or "idle")
            lines.append(f"{gpu:<4} {state:<11} {'-':<42} {'-':<11} {'-':<7} {'-':<9} {'-':<8} {'-':<8} {'-':<8} {'-':<9} -")
            continue
        epoch = "-" if record["epoch"] is None else f"{record['epoch']}/{record['epochs_total']}"
        run_label = f"{record['experiment']} s{record['seed']}"
        lines.append(
            f"{gpu:<4} {record['phase']:<11} {run_label[:42]:<42} {epoch:<11} "
            f"{_fmt(record['train_loss']):<7} {_fmt(record['train_dice']):<9} "
            f"{_fmt(record['val_dice']):<8} {_fmt(record['best_val_dice']):<8} "
            f"{_fmt(record['val_hd95'], 2):<8} {_fmt_sci(record['lr']):<9} "
            f"{record['queue_position']}/{record['queue_total']}"
        )

    terminal = [record for record in records if record["phase"] in {"completed", "failed"}]
    if terminal:
        lines.extend(["-" * 138, "Latest terminal runs:"])
        for record in terminal[-10:]:
            lines.append(
                f"  {record['phase']:<9} GPU{record['gpu']} {record['experiment']} seed={record['seed']} "
                f"epoch={record['epoch'] or '-'} best={_fmt(record['best_val_dice'])}"
            )

    if show_all:
        lines.extend(["-" * 138, "All runs:"])
        for record in records:
            epoch = "-" if record["epoch"] is None else f"{record['epoch']}/{record['epochs_total']}"
            lines.append(
                f"  GPU{record['gpu']} {record['phase']:<11} {record['experiment']:<45} "
                f"seed={record['seed']:<4} epoch={epoch:<9} val={_fmt(record['val_dice'])} "
                f"best={_fmt(record['best_val_dice'])} hd95={_fmt(record['val_hd95'], 2)}"
            )
    lines.append("=" * 138)
    return "\n".join(lines) + "\n"


def _write_tsv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fields = list(records[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(records)


def update_monitor_files(config: PaperConfig, snapshot: Mapping[str, Any]) -> None:
    monitor_dir = config.artifacts_root / "monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    current_text = render_snapshot(snapshot)
    temporary = monitor_dir / "current.txt.tmp"
    temporary.write_text(current_text, encoding="utf-8")
    temporary.replace(monitor_dir / "current.txt")
    _write_tsv(monitor_dir / "current.tsv", list(snapshot["records"]))
    (monitor_dir / "current.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    state_path = monitor_dir / "state.json"
    state = _read_json(state_path)
    seen_epochs = {str(key): int(value) for key, value in dict(state.get("epochs", {})).items()}
    seen_status = {str(key): str(value) for key, value in dict(state.get("status", {})).items()}
    event_lines: list[str] = []
    epoch_rows: list[dict[str, Any]] = []

    for record in snapshot["records"]:
        run_id = str(record["run_id"])
        rows = _read_history(Path(record["history_path"]))
        last_seen = seen_epochs.get(run_id, 0)
        for row in rows:
            epoch = _first_int(row, "epoch") or 0
            if epoch <= last_seen:
                continue
            event = {
                "timestamp": snapshot["updated_at"],
                "gpu": record["gpu"],
                "run_id": run_id,
                "experiment": record["experiment"],
                "seed": record["seed"],
                "epoch": epoch,
                "train_loss": _first_float(row, "train_loss", "loss"),
                "train_dice": _first_float(row, "train_dice", "dice"),
                "val_dice": _first_float(row, "val_macro_dice", "macro_dice", "val_dice"),
                "val_iou": _first_float(row, "val_macro_iou", "macro_iou", "val_iou"),
                "val_hd95": _first_float(row, "val_macro_hd95", "macro_hd95", "val_hd95"),
                "best_val_dice": _first_float(row, "best_val_dice"),
                "lr": _first_float(row, "lr"),
            }
            epoch_rows.append(event)
            event_lines.append(
                f"{snapshot['updated_at']} | GPU{record['gpu']} | {record['experiment']} seed={record['seed']} "
                f"| epoch={epoch}/{record['epochs_total']} | loss={_fmt(event['train_loss'])} "
                f"train_dice={_fmt(event['train_dice'])} val_dice={_fmt(event['val_dice'])} "
                f"best={_fmt(event['best_val_dice'])} hd95={_fmt(event['val_hd95'], 2)}"
            )
            last_seen = max(last_seen, epoch)
        seen_epochs[run_id] = last_seen

        phase = str(record["phase"])
        if seen_status.get(run_id) != phase:
            event_lines.append(
                f"{snapshot['updated_at']} | GPU{record['gpu']} | {record['experiment']} seed={record['seed']} "
                f"| status={phase}"
            )
            seen_status[run_id] = phase

    if epoch_rows:
        epoch_path = monitor_dir / "epochs.csv"
        fields = list(epoch_rows[0])
        write_header = not epoch_path.exists() or epoch_path.stat().st_size == 0
        with epoch_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerows(epoch_rows)
    if event_lines:
        with (monitor_dir / "events.log").open("a", encoding="utf-8") as handle:
            handle.write("\n".join(event_lines) + "\n")

    state_path.write_text(
        json.dumps(
            {"updated_at": snapshot["updated_at"], "epochs": seen_epochs, "status": seen_status},
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


def all_runs_terminal(snapshot: Mapping[str, Any]) -> bool:
    records = list(snapshot["records"])
    return bool(records) and all(
        record["phase"] in {"completed", "failed"} for record in records
    )


def dashboard_once(config: PaperConfig, *, show_all: bool = False) -> str:
    snapshot = build_snapshot(config)
    update_monitor_files(config, snapshot)
    return render_snapshot(snapshot, show_all=show_all)


def monitor_loop(
    config: PaperConfig,
    *,
    interval: float = 30.0,
    print_to_stdout: bool = False,
    clear_screen: bool = False,
    show_all: bool = False,
    stop_when_finished: bool = True,
) -> None:
    interval = max(2.0, float(interval))
    while True:
        snapshot = build_snapshot(config)
        update_monitor_files(config, snapshot)
        if print_to_stdout:
            if clear_screen:
                print("\033[2J\033[H", end="")
            print(render_snapshot(snapshot, show_all=show_all), end="", flush=True)
        if stop_when_finished and all_runs_terminal(snapshot):
            return
        time.sleep(interval)

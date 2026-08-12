from __future__ import annotations

import csv
import json
import math
import sqlite3
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .config import PaperConfig
from .registry import Job, build_commands, run_directory


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS project_metadata (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    experiment TEXT NOT NULL,
    family TEXT NOT NULL,
    variant TEXT NOT NULL,
    seed INTEGER NOT NULL,
    gpu INTEGER,
    env TEXT NOT NULL,
    status TEXT NOT NULL,
    main_table INTEGER NOT NULL,
    input_height INTEGER NOT NULL,
    input_width INTEGER NOT NULL,
    metric_height INTEGER NOT NULL,
    metric_width INTEGER NOT NULL,
    batch_size INTEGER NOT NULL,
    val_batch_size INTEGER NOT NULL,
    grad_accum_steps INTEGER NOT NULL,
    effective_batch_size INTEGER NOT NULL,
    steps_per_epoch INTEGER NOT NULL,
    checkpoint TEXT,
    run_directory TEXT,
    train_started_at TEXT,
    train_finished_at TEXT,
    predict_finished_at TEXT,
    error TEXT,
    metadata_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS epochs (
    run_id TEXT NOT NULL,
    epoch INTEGER NOT NULL,
    lr REAL,
    train_loss REAL,
    train_dice REAL,
    val_macro_dice REAL,
    val_macro_iou REAL,
    val_macro_hd95 REAL,
    best_val_dice REAL,
    epoch_seconds REAL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (run_id, epoch),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS case_metrics (
    run_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    sample_id TEXT NOT NULL,
    sample_index INTEGER,
    dice REAL NOT NULL,
    iou REAL NOT NULL,
    hd95 REAL NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (run_id, domain, sample_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS domain_metrics (
    run_id TEXT NOT NULL,
    split TEXT NOT NULL,
    domain TEXT NOT NULL,
    n INTEGER NOT NULL,
    dice REAL NOT NULL,
    iou REAL NOT NULL,
    hd95 REAL NOT NULL,
    hd95_median REAL,
    hd95_p90 REAL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (run_id, split, domain),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS paired_statistics (
    reference_experiment TEXT NOT NULL,
    comparison_experiment TEXT NOT NULL,
    metric TEXT NOT NULL,
    n_pairs INTEGER NOT NULL,
    mean_difference REAL NOT NULL,
    bootstrap_ci_low REAL,
    bootstrap_ci_high REAL,
    p_value REAL,
    correction TEXT,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (reference_experiment, comparison_experiment, metric)
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=120.0)
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA)
    return connection




def set_project_metadata(connection: sqlite3.Connection, key: str, value: Any) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO project_metadata (key, value_json) VALUES (?, ?)",
        (key, json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)),
    )
    connection.commit()

def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def register_run(
    connection: sqlite3.Connection,
    config: PaperConfig,
    job: Job,
    gpu: int | None,
    status: str,
    error: str | None = None,
) -> None:
    commands = build_commands(config, job)
    run_dir = run_directory(config, job)
    metadata = {
        "job": job.to_dict(),
        "commands": commands.to_dict(),
        "config": str(config.path),
    }
    connection.execute(
        """
        INSERT INTO runs (
            run_id, experiment, family, variant, seed, gpu, env, status,
            main_table, input_height, input_width, metric_height, metric_width,
            batch_size, val_batch_size, grad_accum_steps, effective_batch_size,
            steps_per_epoch, checkpoint, run_directory, error, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id) DO UPDATE SET
            gpu=excluded.gpu,
            status=excluded.status,
            error=excluded.error,
            checkpoint=excluded.checkpoint,
            run_directory=excluded.run_directory,
            metadata_json=excluded.metadata_json
        """,
        (
            job.run_id,
            job.experiment,
            job.family,
            job.variant,
            job.seed,
            gpu,
            job.env,
            status,
            int(job.main_table),
            job.input_size[0],
            job.input_size[1],
            job.metric_size[0],
            job.metric_size[1],
            job.batch_size,
            job.val_batch_size,
            job.grad_accum_steps,
            job.effective_batch_size,
            job.steps_per_epoch,
            str(commands.checkpoint),
            str(run_dir),
            error,
            json.dumps(metadata, ensure_ascii=False, sort_keys=True),
        ),
    )
    connection.commit()


def set_run_state(
    connection: sqlite3.Connection,
    run_id: str,
    status: str,
    *,
    error: str | None = None,
    timestamp_field: str | None = None,
) -> None:
    allowed_fields = {"train_started_at", "train_finished_at", "predict_finished_at"}
    if timestamp_field is not None and timestamp_field not in allowed_fields:
        raise ValueError(f"Unsupported timestamp field: {timestamp_field}")
    if timestamp_field is None:
        connection.execute(
            "UPDATE runs SET status=?, error=? WHERE run_id=?",
            (status, error, run_id),
        )
    else:
        connection.execute(
            f"UPDATE runs SET status=?, error=?, {timestamp_field}=? WHERE run_id=?",
            (status, error, utc_now(), run_id),
        )
    connection.commit()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def ingest_history(connection: sqlite3.Connection, run_id: str, path: Path) -> int:
    rows = read_csv(path)
    connection.execute("DELETE FROM epochs WHERE run_id=?", (run_id,))
    inserted = 0
    for row_number, row in enumerate(rows, start=1):
        epoch = _int(row.get("epoch")) or row_number
        connection.execute(
            """
            INSERT OR REPLACE INTO epochs (
                run_id, epoch, lr, train_loss, train_dice,
                val_macro_dice, val_macro_iou, val_macro_hd95,
                best_val_dice, epoch_seconds, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                epoch,
                _float(row.get("lr")),
                _float(row.get("train_loss") or row.get("loss")),
                _float(row.get("train_dice") or row.get("dice")),
                _float(row.get("val_macro_dice") or row.get("macro_dice")),
                _float(row.get("val_macro_iou") or row.get("macro_iou")),
                _float(row.get("val_macro_hd95") or row.get("macro_hd95")),
                _float(row.get("best_val_dice")),
                _float(row.get("epoch_seconds") or row.get("time")),
                json.dumps(row, ensure_ascii=False, sort_keys=True),
            ),
        )
        inserted += 1
    connection.commit()
    return inserted


def find_metric_file(root: Path, filename: str) -> Path | None:
    candidates = sorted(root.rglob(filename), key=lambda path: (len(path.parts), str(path)))
    return candidates[0] if candidates else None


def normalize_case_row(row: Mapping[str, Any], fallback_index: int) -> dict[str, Any]:
    domain = str(row.get("domain") or row.get("organ") or "unknown")
    sample_id = str(
        row.get("sample_id")
        or row.get("case_id")
        or row.get("data_idx")
        or row.get("index")
        or fallback_index
    )
    dice = _float(row.get("dice"))
    iou = _float(row.get("iou"))
    hd95 = _float(row.get("hd95"))
    if dice is None or iou is None or hd95 is None:
        raise ValueError(f"Case metric row is missing dice/iou/hd95: {row}")
    return {
        "domain": domain,
        "sample_id": sample_id,
        "sample_index": _int(row.get("index") or row.get("data_idx")),
        "dice": dice,
        "iou": iou,
        "hd95": hd95,
        "payload": dict(row),
    }


def derive_domain_metrics(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[str(case["domain"])].append(case)

    rows: list[dict[str, Any]] = []
    for domain, values in sorted(grouped.items()):
        hd95 = [float(value["hd95"]) for value in values]
        rows.append(
            {
                "domain": domain,
                "n": len(values),
                "dice": statistics.fmean(float(value["dice"]) for value in values),
                "iou": statistics.fmean(float(value["iou"]) for value in values),
                "hd95": statistics.fmean(hd95),
                "hd95_median": statistics.median(hd95),
                "hd95_p90": percentile(hd95, 90.0),
            }
        )

    if rows:
        rows.append(
            {
                "domain": "__macro__",
                "n": sum(row["n"] for row in rows),
                "dice": statistics.fmean(row["dice"] for row in rows),
                "iou": statistics.fmean(row["iou"] for row in rows),
                "hd95": statistics.fmean(row["hd95"] for row in rows),
                "hd95_median": statistics.fmean(row["hd95_median"] for row in rows),
                "hd95_p90": statistics.fmean(row["hd95_p90"] for row in rows),
            }
        )
        total = sum(row["n"] for row in rows if not row["domain"].startswith("__"))
        source = [row for row in rows if not row["domain"].startswith("__")]
        rows.append(
            {
                "domain": "__weighted__",
                "n": total,
                "dice": sum(row["dice"] * row["n"] for row in source) / total,
                "iou": sum(row["iou"] * row["n"] for row in source) / total,
                "hd95": sum(row["hd95"] * row["n"] for row in source) / total,
                "hd95_median": None,
                "hd95_p90": None,
            }
        )
    return rows


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return float("nan")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q / 100.0
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def ingest_predictions(
    connection: sqlite3.Connection,
    run_id: str,
    prediction_root: Path,
) -> tuple[int, int]:
    per_case_path = find_metric_file(prediction_root, "per_case_metrics.csv")
    if per_case_path is None:
        raise FileNotFoundError(f"No per_case_metrics.csv found below {prediction_root}")
    raw_rows = read_csv(per_case_path)
    cases = [normalize_case_row(row, index) for index, row in enumerate(raw_rows)]

    connection.execute("DELETE FROM case_metrics WHERE run_id=?", (run_id,))
    connection.execute("DELETE FROM domain_metrics WHERE run_id=?", (run_id,))
    for case in cases:
        connection.execute(
            """
            INSERT OR REPLACE INTO case_metrics (
                run_id, domain, sample_id, sample_index, dice, iou, hd95, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                case["domain"],
                case["sample_id"],
                case["sample_index"],
                case["dice"],
                case["iou"],
                case["hd95"],
                json.dumps(case["payload"], ensure_ascii=False, sort_keys=True),
            ),
        )

    domain_rows = derive_domain_metrics(cases)
    for row in domain_rows:
        connection.execute(
            """
            INSERT OR REPLACE INTO domain_metrics (
                run_id, split, domain, n, dice, iou, hd95,
                hd95_median, hd95_p90, payload_json
            ) VALUES (?, 'test', ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                row["domain"],
                row["n"],
                row["dice"],
                row["iou"],
                row["hd95"],
                row["hd95_median"],
                row["hd95_p90"],
                json.dumps(row, ensure_ascii=False, sort_keys=True),
            ),
        )
    connection.commit()
    return len(cases), len(domain_rows)


def ingest_run(connection: sqlite3.Connection, config: PaperConfig, job: Job) -> dict[str, int]:
    commands = build_commands(config, job)
    epochs = ingest_history(connection, job.run_id, commands.history)
    cases, domains = ingest_predictions(connection, job.run_id, commands.prediction_root)
    return {"epochs": epochs, "cases": cases, "domains": domains}


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_results(connection: sqlite3.Connection, output_root: Path) -> dict[str, Path]:
    runs_full_path = output_root / "paper_runs.csv"
    write_csv(
        runs_full_path,
        [dict(row) for row in connection.execute(
            "SELECT * FROM runs ORDER BY experiment, seed"
        ).fetchall()],
    )
    epochs_path = output_root / "paper_epochs.csv"
    write_csv(
        epochs_path,
        [dict(row) for row in connection.execute(
            "SELECT * FROM epochs ORDER BY run_id, epoch"
        ).fetchall()],
    )
    cases_path = output_root / "paper_case_metrics.csv"
    write_csv(
        cases_path,
        [dict(row) for row in connection.execute(
            "SELECT * FROM case_metrics ORDER BY run_id, domain, sample_id"
        ).fetchall()],
    )
    domains_path = output_root / "paper_domain_metrics.csv"
    write_csv(
        domains_path,
        [dict(row) for row in connection.execute(
            "SELECT * FROM domain_metrics ORDER BY run_id, split, domain"
        ).fetchall()],
    )

    run_rows = connection.execute(
        """
        SELECT r.run_id, r.experiment, r.variant, r.seed, r.status, r.main_table,
               r.input_height, r.input_width, r.metric_height, r.metric_width,
               r.batch_size, r.effective_batch_size, r.gpu,
               d.domain, d.n, d.dice, d.iou, d.hd95, d.hd95_median, d.hd95_p90
        FROM runs r
        LEFT JOIN domain_metrics d ON d.run_id=r.run_id AND d.split='test'
        ORDER BY r.experiment, r.seed, d.domain
        """
    ).fetchall()
    long_rows = [dict(row) for row in run_rows]
    long_path = output_root / "paper_results_long.csv"
    write_csv(long_path, long_rows)

    macro_rows = [row for row in long_rows if row.get("domain") == "__macro__"]
    run_summary_path = output_root / "paper_run_summary.csv"
    write_csv(run_summary_path, macro_rows)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in macro_rows:
        if row.get("status") == "completed":
            grouped[str(row["experiment"])].append(row)
    method_rows: list[dict[str, Any]] = []
    for experiment, values in sorted(grouped.items()):
        result: dict[str, Any] = {
            "experiment": experiment,
            "n_seeds": len(values),
            "seeds": ",".join(str(value["seed"]) for value in values),
            "input_size": f"{values[0]['input_height']}x{values[0]['input_width']}",
            "metric_size": f"{values[0]['metric_height']}x{values[0]['metric_width']}",
            "batch_size": values[0]["batch_size"],
            "effective_batch_size": values[0]["effective_batch_size"],
            "main_table": values[0]["main_table"],
        }
        for metric in ("dice", "iou", "hd95"):
            sequence = [float(value[metric]) for value in values]
            result[f"macro_{metric}_mean"] = statistics.fmean(sequence)
            result[f"macro_{metric}_sd"] = statistics.stdev(sequence) if len(sequence) > 1 else 0.0
        method_rows.append(result)
    method_path = output_root / "paper_method_summary.csv"
    write_csv(method_path, method_rows)

    return {
        "runs_full": runs_full_path,
        "epochs": epochs_path,
        "cases": cases_path,
        "domains": domains_path,
        "long": long_path,
        "run_summary": run_summary_path,
        "methods": method_path,
    }


def status_counts(connection: sqlite3.Connection) -> dict[str, int]:
    rows = connection.execute(
        "SELECT status, COUNT(*) AS n FROM runs GROUP BY status ORDER BY status"
    ).fetchall()
    return {str(row["status"]): int(row["n"]) for row in rows}

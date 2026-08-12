from __future__ import annotations

import csv
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _case_means(
    connection: sqlite3.Connection,
    experiment: str,
    metric: str,
) -> dict[tuple[str, str], float]:
    if metric not in {"dice", "iou", "hd95"}:
        raise ValueError(metric)
    rows = connection.execute(
        f"""
        SELECT c.domain, c.sample_id, AVG(c.{metric}) AS value
        FROM case_metrics c
        JOIN runs r ON r.run_id=c.run_id
        WHERE r.experiment=? AND r.status='completed'
        GROUP BY c.domain, c.sample_id
        """,
        (experiment,),
    ).fetchall()
    return {
        (str(row["domain"]), str(row["sample_id"])): float(row["value"])
        for row in rows
    }


def bootstrap_ci(
    differences: np.ndarray,
    *,
    iterations: int = 10000,
    seed: int = 2026,
) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    n = differences.size
    if n == 0:
        return float("nan"), float("nan")
    means = np.empty(iterations, dtype=np.float64)
    block = 250
    for start in range(0, iterations, block):
        stop = min(start + block, iterations)
        indices = generator.integers(0, n, size=(stop - start, n))
        means[start:stop] = differences[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def wilcoxon_pvalue(reference: np.ndarray, comparison: np.ndarray) -> float | None:
    try:
        from scipy.stats import wilcoxon
    except Exception:
        return None
    difference = reference - comparison
    if np.allclose(difference, 0.0):
        return 1.0
    result = wilcoxon(
        reference,
        comparison,
        zero_method="pratt",
        alternative="two-sided",
        method="auto",
    )
    return float(result.pvalue)


def holm_adjust(pvalues: list[float | None]) -> list[float | None]:
    valid = [(index, value) for index, value in enumerate(pvalues) if value is not None]
    adjusted: list[float | None] = [None] * len(pvalues)
    if not valid:
        return adjusted
    ordered = sorted(valid, key=lambda item: item[1])
    running = 0.0
    count = len(ordered)
    for rank, (index, value) in enumerate(ordered):
        candidate = min(1.0, (count - rank) * value)
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def paired_statistics(
    connection: sqlite3.Connection,
    reference_experiment: str,
    comparisons: Iterable[str],
    *,
    iterations: int = 10000,
) -> list[dict[str, Any]]:
    raw: list[dict[str, Any]] = []
    for comparison in comparisons:
        for metric in ("dice", "iou", "hd95"):
            reference = _case_means(connection, reference_experiment, metric)
            candidate = _case_means(connection, comparison, metric)
            keys = sorted(set(reference) & set(candidate))
            if not keys:
                continue
            ref_values = np.asarray([reference[key] for key in keys], dtype=np.float64)
            cmp_values = np.asarray([candidate[key] for key in keys], dtype=np.float64)
            differences = ref_values - cmp_values
            low, high = bootstrap_ci(
                differences,
                iterations=iterations,
                seed=2026 + len(raw),
            )
            raw.append(
                {
                    "reference_experiment": reference_experiment,
                    "comparison_experiment": comparison,
                    "metric": metric,
                    "n_pairs": len(keys),
                    "mean_difference": float(differences.mean()),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "p_value": wilcoxon_pvalue(ref_values, cmp_values),
                }
            )

    adjusted = holm_adjust([row["p_value"] for row in raw])
    for row, corrected in zip(raw, adjusted):
        row["p_value_holm"] = corrected
        connection.execute(
            """
            INSERT OR REPLACE INTO paired_statistics (
                reference_experiment, comparison_experiment, metric, n_pairs,
                mean_difference, bootstrap_ci_low, bootstrap_ci_high,
                p_value, correction, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'Holm', ?)
            """,
            (
                row["reference_experiment"],
                row["comparison_experiment"],
                row["metric"],
                row["n_pairs"],
                row["mean_difference"],
                row["bootstrap_ci_low"],
                row["bootstrap_ci_high"],
                row["p_value_holm"],
                json.dumps(row, ensure_ascii=False, sort_keys=True),
            ),
        )
    connection.commit()
    return raw


def write_statistics(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

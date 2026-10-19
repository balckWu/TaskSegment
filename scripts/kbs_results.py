"""Collect domain summaries and reproduce KBS tables without patient identifiers.

This reads completed evaluations; it does not train or rerun inference.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 3407, 2026)
COUNTS = {"thyroid": 286, "TN3K": 614, "BUSI_WHU": 186,
          "BUS-BRA": 283, "OTU": 382, "prostate": 16}
METRICS = ("dice", "iou", "hd95")
# First alias is the historical run name; subsequent aliases are release names.
RUN_NAMES = {
    "tasksegment_full2": ("tasksegment_full2", "tasksegment_full"),
    "unet": ("unet",), "unetpp": ("unetpp",),
    "swinunet_official": ("swinunet_official",),
    "transunet_pretrained": ("transunet_pretrained",),
    "clipseg_official": ("clipseg_official",),
    "lvit_native224": ("lvit_native224",),
    "sammed2d_promptfree": ("sammed2d_promptfree",),
    "iris2d": ("iris2d",),
    "tasksegment_no_text_conditioning": ("tasksegment_no_text_conditioning",),
    "tasksegment_no_visual_prototype": ("tasksegment_no_visual_prototype",),
    "tasksegment_no_dense_guidance_nobank": (
        "tasksegment_no_dense_guidance_nobank", "tasksegment_no_dense_guidance"),
    "tasksegment_no_task_conditioned_decoder": ("tasksegment_no_task_conditioned_decoder",),
}


def read_run(directory: Path, method: str, seed: int) -> list[dict]:
    if method == "iris2d":
        result = json.loads((directory / "predictions/metrics.json").read_text(encoding="utf-8"))
        if result.get("seed") != seed or result.get("eval_support_count") != 4:
            raise ValueError(f"Iris seed/evaluation-support mismatch: {directory}")
        values = result["per_domain"]
    else:
        files = list((directory / "predictions").rglob("summary_metrics.csv"))
        if len(files) != 1:
            raise ValueError(f"Expected one domain summary in {directory}, found {len(files)}")
        with files[0].open(encoding="utf-8-sig", newline="") as handle:
            raw = list(csv.DictReader(handle))
        values = {}
        for row in raw:
            domain = row.get("organ", row.get("domain", ""))
            if domain.lower() in {"macro", "average", "mean", "all"}:
                continue
            if domain in values:
                raise ValueError(f"Duplicate domain {domain}: {files[0]}")
            values[domain] = {**{m: float(row[m]) for m in METRICS},
                              "count": int(row.get("count", row.get("n")))}
    return [{"method": method, "seed": seed, "domain": domain,
             "n": int(row["count"]), **{m: float(row[m]) for m in METRICS}}
            for domain, row in values.items()]


def collect(run_roots: list[Path]) -> list[dict]:
    rows = []
    for method, aliases in RUN_NAMES.items():
        for seed in SEEDS:
            candidates = [root / alias / f"seed_{seed}" for root in run_roots for alias in aliases
                          if (root / alias / f"seed_{seed}").is_dir()]
            if len(candidates) != 1:
                raise ValueError(f"Expected exactly one {method}/seed_{seed}, found {len(candidates)}")
            rows.extend(read_run(candidates[0], method, seed))
    aggregate(rows)
    return rows


def aggregate(rows: list[dict], require_complete: bool = True) -> dict:
    grouped = {}
    for row in rows:
        method, seed, domain = row["method"], row["seed"], row["domain"]
        if method not in RUN_NAMES or seed not in SEEDS or domain not in COUNTS:
            raise ValueError(f"Unexpected method/seed/domain: {(method, seed, domain)}")
        if row["n"] != COUNTS[domain]:
            raise ValueError(f"Test count mismatch: {(method, seed, domain)}")
        key = (method, seed, domain)
        if key in grouped:
            raise ValueError(f"Duplicate result: {key}")
        for metric in METRICS:
            value = float(row[metric])
            if not math.isfinite(value) or value < 0 or (metric != "hd95" and value > 1):
                raise ValueError(f"Invalid {metric}: {key}")
        grouped[key] = row
    methods = sorted({key[0] for key in grouped})
    if require_complete and set(methods) != set(RUN_NAMES):
        raise ValueError("Expected nine main-comparison methods and four ablations")
    stats = {}
    for method in methods:
        expected = {(method, seed, domain) for seed in SEEDS for domain in COUNTS}
        if not expected.issubset(grouped):
            raise ValueError(f"Missing seed/domain: {method}")
        stats[method] = {}
        for domain in (*COUNTS, "macro"):
            stats[method][domain] = {}
            for metric in METRICS:
                values = [(statistics.mean(grouped[method, seed, d][metric] for d in COUNTS)
                           if domain == "macro" else grouped[method, seed, domain][metric])
                          for seed in SEEDS]
                stats[method][domain][metric] = {
                    "mean": statistics.mean(values), "sd": statistics.stdev(values)}
    return stats


def verify(stats: dict, reference: dict) -> dict:
    checked = 0
    for method, domains in reference["stats"].items():
        for domain, metrics in domains.items():
            for metric, values in metrics.items():
                for name, expected in values.items():
                    precision = reference.get("decimal_places", {}).get(method)
                    actual = stats[method][domain][metric][name]
                    if precision is None:
                        ok = abs(actual - expected) <= (5.1e-7 if method == "iris2d" else 1e-10)
                    else:
                        digits = precision[metric]
                        ok = f"{actual:.{digits}f}" == f"{expected:.{digits}f}"
                    if not ok:
                        raise ValueError(f"Manuscript mismatch: {method}/{domain}/{metric}/{name}: {actual} vs {expected}")
                    checked += 1
    return {"passed": True, "checked_values": checked,
            "scope": "Archived domain summaries, not a new evaluation of predictions"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--input", type=Path, default=None)
    group.add_argument("--run-roots", type=Path, nargs="+")
    parser.add_argument("--reference", type=Path, default=ROOT / "paper_results/reference_summary.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.run_roots:
        rows = collect(args.run_roots)
    else:
        rows = json.loads((args.input or ROOT / "paper_results/seed_metrics.json").read_text(encoding="utf-8"))["rows"]
    stats = aggregate(rows)
    result = {"schema_version": 1, "seeds": list(SEEDS), "rows": rows, "stats": stats,
              "verification": verify(stats, json.loads(args.reference.read_text(encoding="utf-8")))}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"methods": len(stats), "domain_rows": len(rows), **result["verification"]}, indent=2))


if __name__ == "__main__":
    main()

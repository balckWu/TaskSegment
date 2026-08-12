from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch


def safe_torch_load(path: str | Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolve_device(value: str) -> torch.device:
    if str(value).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(value)


def amp_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def make_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    rows = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: str | Path, row: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(dict(row))


def cosine_lr(optimizer, epoch: int, epochs: int, warmup: int, base_lr: float, min_lr: float) -> float:
    if warmup > 0 and epoch < warmup:
        value = min_lr + (base_lr - min_lr) * ((epoch + 1) / warmup)
    else:
        denominator = max(1, epochs - warmup - 1)
        progress = min(max((epoch - warmup) / denominator, 0.0), 1.0)
        value = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = value
    return value


def summarize_case_rows(method: str, rows, domain_names):
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["domain"])].append(row)
    summary = []
    for domain in domain_names:
        values = grouped[domain]
        hd = np.asarray([float(v["hd95"]) for v in values], dtype=np.float64)
        summary.append({
            "method": method,
            "domain": domain,
            "n": len(values),
            "dice": float(np.mean([float(v["dice"]) for v in values])),
            "iou": float(np.mean([float(v["iou"]) for v in values])),
            "hd95": float(hd.mean()),
            "hd95_median": float(np.median(hd)),
            "hd95_p90": float(np.percentile(hd, 90)),
        })
    summary.append({
        "method": method,
        "domain": "macro",
        "n": int(sum(v["n"] for v in summary)),
        "dice": float(np.mean([v["dice"] for v in summary])),
        "iou": float(np.mean([v["iou"] for v in summary])),
        "hd95": float(np.mean([v["hd95"] for v in summary])),
        "hd95_median": float(np.mean([v["hd95_median"] for v in summary])),
        "hd95_p90": float(np.mean([v["hd95_p90"] for v in summary])),
    })
    return summary


def update_summary_file(path: str | Path, method: str, new_rows) -> None:
    path = Path(path)
    existing = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
    existing = [row for row in existing if row.get("method") != method]
    write_csv(path, existing + list(new_rows))

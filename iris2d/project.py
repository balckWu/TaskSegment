"""Reuse TSV3 utilities directly, without importing its model/trainer package."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from .protocol import load_recipe, normalized_sha256

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("thyroid", "TN3K", "BUSI_WHU", "BUS-BRA", "OTU", "prostate")
MODALITY = {"thyroid": "thyroid", "TN3K": "thyroid", "BUSI_WHU": "BUSI", "BUS-BRA": "BUSI"}


def load_util(relative, name):
    file = ROOT / relative
    if not file.is_file():
        raise FileNotFoundError(f"Copy this add-on into TSV3 first; missing {file}")
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def project_utilities():
    return (
        load_util("tasksegment/data/datasets.py", "_iris2d_project_data"),
        load_util("tasksegment/training/augmentations.py", "_iris2d_project_augmentation"),
        load_util("tasksegment/training/metrics.py", "_iris2d_project_metrics"),
    )


def source_hashes():
    files = ["tasksegment/data/datasets.py", "tasksegment/training/augmentations.py",
             "tasksegment/training/metrics.py", "paperlab/progress.py"]
    files += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "iris2d").glob("*.py"))]
    files += ["configs/iris2d_baseline.json", "configs/iris2d_reference.json"]
    result = {rel: normalized_sha256(ROOT / rel) for rel in files}
    _, common = load_recipe(ROOT)
    result["configs/paper_3seeds.json::controlled_protocol"] = hashlib.sha256(
        json.dumps(common, sort_keys=True).encode()).hexdigest()
    return result

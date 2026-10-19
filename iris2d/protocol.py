"""Controlled Iris baseline settings, audited against local TSV3 baselines.

No Torch import: preflight and dry-run work without loading any model.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CONTROL_KEYS = ("epochs", "batch_size", "val_batch_size", "grad_accum_steps",
                "effective_batch_size", "episodes_per_epoch", "warmup_epochs",
                "patience", "threshold", "metric_size", "domain_sampling_alpha",
                "deterministic", "amp", "postprocess", "seeds")


def normalized_sha256(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def portable_ast_data(value):
    """Canonical AST fields shared by Python 3.10 and 3.12.

    Python 3.12 adds an empty FunctionDef/ClassDef.type_params field even to
    ordinary 3.10-compatible code. Its absence and an empty list are equivalent.
    Keep nonempty type parameters, annotations, literals and function bodies.
    """
    if isinstance(value, ast.AST):
        return [type(value).__name__, [
            [name, portable_ast_data(child)] for name, child in ast.iter_fields(value)
            if not (name == "type_params" and not child)
        ]]
    if isinstance(value, list):
        return [portable_ast_data(child) for child in value]
    if value is Ellipsis:
        return ["constant", "ellipsis"]
    if isinstance(value, bytes):
        return ["constant", "bytes", value.hex()]
    if isinstance(value, complex):
        return ["constant", "complex", repr(value)]
    return value


def portable_ast_sha256(node):
    payload = json.dumps(portable_ast_data(node), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_ast_sha256(path):
    return portable_ast_sha256(ast.parse(Path(path).read_text(encoding="utf-8-sig")))


def function_sha256(path, name):
    tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return portable_ast_sha256(node)


def load_recipe(root=ROOT):
    root = Path(root)
    baseline = json.loads((root / "configs/iris2d_baseline.json").read_text(encoding="utf-8-sig"))
    reference = json.loads((root / baseline["reference_config"]).read_text(encoding="utf-8-sig"))["protocol"]
    common = {key: reference[key] for key in CONTROL_KEYS}
    if baseline["optimizer"] != "AdamW" or baseline["lr_schedule"] != "unet_family_linear_warmup_then_epoch_cosine":
        raise ValueError("Unsupported optimizer or learning rate schedule")
    if common["domain_sampling_alpha"] != 0. or not common["deterministic"] or not common["amp"]:
        raise ValueError("This controlled implementation requires balanced domains, determinism and AMP")
    if common["batch_size"] * common["grad_accum_steps"] != common["effective_batch_size"]:
        raise ValueError("Reference effective batch size is inconsistent")
    if common["metric_size"] != [512, 512] or common["threshold"] != .5 or common["postprocess"]:
        raise ValueError("Reference evaluation protocol changed; review before training")
    return baseline, common


def reference_report(root=ROOT):
    """Read-only compatibility report; retain all real code/protocol checks."""
    root = Path(root)
    expected = json.loads((root / "configs/iris2d_reference.json").read_text(encoding="utf-8"))
    if expected.get("ast_hash_format") != "portable_ast_json_v1":
        raise ValueError("Reference manifest needs the matching Python 3.10/3.12 compatibility update")
    mismatches = []
    formatting_only = []
    statuses = {}
    for relative, digest in expected["normalized_source_sha256"].items():
        file = root / relative
        actual = normalized_sha256(file) if file.is_file() else None
        semantic = None
        if actual != digest and file.is_file():
            try:
                semantic = source_ast_sha256(file)
            except (OSError, SyntaxError):
                pass
        semantic_match = semantic is not None and semantic == expected.get("source_ast_sha256", {}).get(relative)
        if actual == digest:
            status = "exact"
        elif semantic_match:
            status = "comments_or_formatting_only"
            formatting_only.append(relative)
        else:
            status = "different"
            mismatches.append(relative)
        statuses[relative] = {"status": status, "normalized_sha256": actual,
                              "source_ast_sha256": semantic}
    for relative, digest in expected["learning_rate_function_sha256"].items():
        try:
            actual = function_sha256(root / relative, "set_cosine_lr")
        except (OSError, StopIteration, SyntaxError):
            actual = None
        if actual != digest:
            mismatches.append(relative + "::set_cosine_lr")
        statuses[relative + "::set_cosine_lr"] = {
            "status": "exact" if actual == digest else "different", "function_sha256": actual}
    _, common = load_recipe(root)
    if common != expected["controlled_protocol"]:
        mismatches.append("configs/paper_3seeds.json (controlled protocol)")
    return {"ok": not mismatches, "python": sys.version, "mismatches": mismatches,
            "formatting_only": formatting_only, "files": statuses,
            "controlled_protocol": common}


def verify_reference(root=ROOT):
    """Detect substantive server source changes before spending GPU time."""
    report = reference_report(root)
    if report["mismatches"]:
        raise ValueError("Source/protocol differs from the reviewed local TSV3: "
                         + ", ".join(report["mismatches"])
                         + ". Run python scripts/diagnose_iris2d.py to collect the source differences.")
    expected = json.loads((Path(root) / "configs/iris2d_reference.json").read_text(encoding="utf-8"))
    expected["formatting_only"] = report["formatting_only"]
    return expected


def default_data_root(root=ROOT):
    root = Path(root)
    # Local checkout is TSV3/TaskSegment, with data in the parent. The server
    # layout places code and data together. An explicit CLI path always wins.
    candidates = (root / "data", root.parent / "data")
    for candidate in candidates:
        if (candidate / "TN3K/train/images").is_dir():
            return candidate.resolve()
    return candidates[0].resolve()


def default_run_root(root=ROOT):
    return default_data_root(root).parent / "paper_artifacts/runs/iris2d"


def set_cosine_lr(optimizer, epoch, num_epochs, base_lr, min_lr, warmup_epochs):
    """Same formula as U-Net/UNet++/Swin-Unet/TransUNet set_cosine_lr."""
    if warmup_epochs > 0 and epoch < warmup_epochs:
        current = base_lr * float(epoch + 1) / float(warmup_epochs)
    else:
        remaining = max(1, num_epochs - warmup_epochs - 1)
        progress = min(max((epoch - warmup_epochs) / remaining, 0.0), 1.0)
        current = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = float(current)
    return float(current)

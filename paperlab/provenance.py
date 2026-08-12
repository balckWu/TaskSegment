from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            command,
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=20,
        ).strip()
    except Exception:
        return None


def git_info(root: Path) -> dict[str, Any]:
    commit = _run(["git", "rev-parse", "HEAD"], cwd=root)
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root)
    status = _run(["git", "status", "--porcelain"], cwd=root)
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status) if status is not None else None,
    }


def code_manifest(root: Path, paths: Iterable[Path]) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for base in paths:
        target = base if base.is_absolute() else root / base
        if target.is_file():
            manifest[str(target.relative_to(root))] = sha256_file(target)
            continue
        if not target.exists():
            continue
        for path in sorted(target.rglob("*")):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            manifest[str(path.relative_to(root))] = sha256_file(path)
    return manifest


def system_manifest(root: Path, config_path: Path) -> dict[str, Any]:
    nvidia_smi = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    return {
        "created_at": utc_now(),
        "project_root": str(root),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "git": git_info(root),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": os.sys.executable,
        },
        "system": {
            "platform": platform.platform(),
            "hostname": platform.node(),
        },
        "gpus": nvidia_smi.splitlines() if nvidia_smi else [],
        "code_sha256": code_manifest(
            root,
            [
                Path("paperlab"),
                Path("tasksegment"),
                Path("scripts"),
                Path("configs"),
                Path("third_party"),
            ],
        ),
    }


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)

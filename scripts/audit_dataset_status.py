#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, UnidentifiedImageError

IMAGE_DIR_NAMES = ("images", "image", "imgs", "img")
MASK_DIR_NAMES = ("masks", "mask", "labels", "label", "annotations")
SPLIT_NAMES = ("train", "val", "valid", "validation", "test")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class SampleRecord:
    domain: str
    split: str
    sample_id: str
    image_path: str
    mask_path: str
    image_exists: bool
    mask_exists: bool
    image_width: int | None = None
    image_height: int | None = None
    mask_width: int | None = None
    mask_height: int | None = None
    size_match: bool | None = None
    mask_empty: bool | None = None
    mask_foreground_pixels: int | None = None
    mask_total_pixels: int | None = None
    foreground_ratio: float | None = None
    image_sha256: str | None = None
    image_error: str = ""
    mask_error: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit a multi-domain medical segmentation dataset")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--hash-chunk-size", type=int, default=1024 * 1024)
    p.add_argument("--no-mask-content-check", action="store_true")
    p.add_argument("--fail-on-leakage", action="store_true")
    return p.parse_args()


def normalize_split(name: str) -> str:
    value = name.lower()
    return "val" if value in {"valid", "validation"} else value


def find_named_subdir(parent: Path, candidates: Iterable[str]) -> Path | None:
    children = {c.name.lower(): c for c in parent.iterdir() if c.is_dir()}
    for name in candidates:
        if name.lower() in children:
            return children[name.lower()]
    return None


def list_images(folder: Path | None) -> list[Path]:
    if folder is None or not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def sha256_file(path: Path, chunk_size: int) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def safe_image_size(path: Path) -> tuple[int | None, int | None, str]:
    try:
        with Image.open(path) as im:
            w, h = im.size
        return w, h, ""
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def inspect_mask(path: Path):
    try:
        with Image.open(path) as m:
            w, h = m.size
            arr = np.asarray(m)
        if arr.ndim == 3:
            arr = np.max(arr, axis=2)
        fg = arr > 0
        fg_pixels = int(fg.sum())
        total = int(fg.size)
        ratio = float(fg_pixels / total) if total else math.nan
        return w, h, fg_pixels == 0, fg_pixels, total, ratio, ""
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        return None, None, None, None, None, None, f"{type(exc).__name__}: {exc}"


def build_mask_index(mask_paths: list[Path]) -> dict[str, list[Path]]:
    d: dict[str, list[Path]] = defaultdict(list)
    for p in mask_paths:
        d[p.stem].append(p)
    return d


def choose_mask(image_path: Path, mask_index: dict[str, list[Path]]) -> Path | None:
    matches = mask_index.get(image_path.stem, [])
    if not matches:
        return None
    same = [p for p in matches if p.suffix.lower() == image_path.suffix.lower()]
    return sorted(same or matches)[0]


def discover_domains(data_root: Path) -> list[Path]:
    return sorted(p for p in data_root.iterdir() if p.is_dir() and not p.name.startswith("."))


def discover_splits(domain_dir: Path) -> list[tuple[str, Path]]:
    out = []
    for child in sorted(domain_dir.iterdir()):
        if child.is_dir():
            s = normalize_split(child.name)
            if child.name.lower() in SPLIT_NAMES or s in {"train", "val", "test"}:
                out.append((s, child))
    return out


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else None


def summarize_records(records: list[SampleRecord]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[SampleRecord]] = defaultdict(list)
    for r in records:
        grouped[(r.domain, r.split)].append(r)
    rows = []
    for (domain, split), group in sorted(grouped.items()):
        widths = [r.image_width for r in group if r.image_width is not None]
        heights = [r.image_height for r in group if r.image_height is not None]
        ratios = [r.foreground_ratio for r in group if r.foreground_ratio is not None and not math.isnan(r.foreground_ratio)]
        rows.append({
            "domain": domain,
            "split": split,
            "image_count": sum(r.image_exists for r in group),
            "matched_mask_count": sum(r.mask_exists for r in group),
            "missing_mask_count": sum(not r.mask_exists for r in group),
            "empty_mask_count": sum(r.mask_empty is True for r in group),
            "size_mismatch_count": sum(r.size_match is False for r in group),
            "corrupt_image_count": sum(bool(r.image_error) for r in group),
            "corrupt_mask_count": sum(bool(r.mask_error) for r in group),
            "image_width_min": min(widths) if widths else None,
            "image_width_max": max(widths) if widths else None,
            "image_height_min": min(heights) if heights else None,
            "image_height_max": max(heights) if heights else None,
            "foreground_ratio_mean": float(np.mean(ratios)) if ratios else None,
            "foreground_ratio_median": float(np.median(ratios)) if ratios else None,
            "foreground_ratio_p05": percentile(ratios, 5),
            "foreground_ratio_p95": percentile(ratios, 95),
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields, seen = [], set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_dir = (args.output_dir or Path.cwd()).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not data_root.is_dir():
        print(f"[error] invalid data root: {data_root}", file=sys.stderr)
        return 1

    records: list[SampleRecord] = []
    orphan_masks: list[dict[str, str]] = []
    filename_locations: dict[str, list[dict[str, str]]] = defaultdict(list)
    domains = discover_domains(data_root)
    print(f"Data root: {data_root}")
    print(f"Domains: {[p.name for p in domains]}\n")

    for domain_dir in domains:
        for split, split_dir in discover_splits(domain_dir):
            image_dir = find_named_subdir(split_dir, IMAGE_DIR_NAMES)
            mask_dir = find_named_subdir(split_dir, MASK_DIR_NAMES)
            if image_dir is None:
                print(f"[warning] missing image dir: {domain_dir.name}/{split}")
                continue
            image_paths = list_images(image_dir)
            mask_paths = list_images(mask_dir)
            mask_index = build_mask_index(mask_paths)
            used_masks: set[Path] = set()
            print(f"Scanning {domain_dir.name}/{split}: images={len(image_paths)}, masks={len(mask_paths)}")

            for image_path in image_paths:
                mask_path = choose_mask(image_path, mask_index)
                if mask_path:
                    used_masks.add(mask_path)
                iw, ih, image_error = safe_image_size(image_path)
                mw = mh = None
                mask_empty = None
                fg_pixels = total_pixels = None
                fg_ratio = None
                mask_error = ""
                if mask_path and not args.no_mask_content_check:
                    mw, mh, mask_empty, fg_pixels, total_pixels, fg_ratio, mask_error = inspect_mask(mask_path)
                elif mask_path:
                    mw, mh, mask_error = safe_image_size(mask_path)
                size_match = None if None in (iw, ih, mw, mh) else (iw == mw and ih == mh)
                try:
                    digest = sha256_file(image_path, args.hash_chunk_size)
                except OSError as exc:
                    digest = None
                    image_error = image_error or f"{type(exc).__name__}: {exc}"

                r = SampleRecord(
                    domain=domain_dir.name,
                    split=split,
                    sample_id=image_path.stem,
                    image_path=str(image_path),
                    mask_path=str(mask_path) if mask_path else "",
                    image_exists=True,
                    mask_exists=mask_path is not None,
                    image_width=iw,
                    image_height=ih,
                    mask_width=mw,
                    mask_height=mh,
                    size_match=size_match,
                    mask_empty=mask_empty,
                    mask_foreground_pixels=fg_pixels,
                    mask_total_pixels=total_pixels,
                    foreground_ratio=fg_ratio,
                    image_sha256=digest,
                    image_error=image_error,
                    mask_error=mask_error,
                )
                records.append(r)
                filename_locations[image_path.name.lower()].append({
                    "domain": domain_dir.name,
                    "split": split,
                    "path": str(image_path),
                    "sha256": digest or "",
                })

            for mask_path in mask_paths:
                if mask_path not in used_masks:
                    orphan_masks.append({
                        "domain": domain_dir.name,
                        "split": split,
                        "sample_id": mask_path.stem,
                        "mask_path": str(mask_path),
                    })

    hash_groups: dict[str, list[SampleRecord]] = defaultdict(list)
    for r in records:
        if r.image_sha256:
            hash_groups[r.image_sha256].append(r)

    duplicate_rows = []
    pair_counter: Counter[tuple[str, str, str, str]] = Counter()
    cross_split_groups = cross_domain_groups = affected_files = 0
    for digest, group in sorted(hash_groups.items()):
        if len(group) < 2:
            continue
        locations = sorted({(r.domain, r.split) for r in group})
        splits = {r.split for r in group}
        ds = {r.domain for r in group}
        cross_split = len(splits) > 1
        cross_domain = len(ds) > 1
        if cross_split:
            cross_split_groups += 1
            affected_files += len(group)
        if cross_domain:
            cross_domain_groups += 1
        for r in group:
            duplicate_rows.append({
                "sha256": digest,
                "group_size": len(group),
                "cross_split": cross_split,
                "cross_domain": cross_domain,
                "domain": r.domain,
                "split": r.split,
                "sample_id": r.sample_id,
                "image_path": r.image_path,
            })
        for left, right in combinations(locations, 2):
            if left[1] != right[1]:
                pair_counter[(left[0], left[1], right[0], right[1])] += 1

    pair_rows = [{
        "left_domain": k[0], "left_split": k[1],
        "right_domain": k[2], "right_split": k[3],
        "duplicate_group_count": v,
    } for k, v in pair_counter.most_common()]

    filename_rows = []
    for filename, locs in sorted(filename_locations.items()):
        if len(locs) < 2:
            continue
        hashes = {x["sha256"] for x in locs if x["sha256"]}
        filename_rows.append({
            "filename": filename,
            "occurrence_count": len(locs),
            "same_content": len(hashes) == 1 and bool(hashes),
            "locations": " | ".join(f"{x['domain']}/{x['split']}:{x['path']}" for x in locs),
        })

    outputs = {
        "summary_csv": output_dir / "dataset_audit_summary.csv",
        "samples_csv": output_dir / "dataset_audit_samples.csv",
        "duplicates_csv": output_dir / "dataset_audit_duplicates.csv",
        "duplicate_pairs_csv": output_dir / "dataset_audit_duplicate_pairs.csv",
        "duplicate_filenames_csv": output_dir / "dataset_audit_duplicate_filenames.csv",
        "orphan_masks_csv": output_dir / "dataset_audit_orphan_masks.csv",
        "report_json": output_dir / "dataset_audit_report.json",
    }
    write_csv(outputs["summary_csv"], summarize_records(records))
    write_csv(outputs["samples_csv"], [asdict(r) for r in records])
    write_csv(outputs["duplicates_csv"], duplicate_rows)
    write_csv(outputs["duplicate_pairs_csv"], pair_rows)
    write_csv(outputs["duplicate_filenames_csv"], filename_rows)
    write_csv(outputs["orphan_masks_csv"], orphan_masks)

    report = {
        "data_root": str(data_root),
        "domains": [p.name for p in domains],
        "total_images": len(records),
        "missing_masks": sum(not r.mask_exists for r in records),
        "orphan_masks": len(orphan_masks),
        "empty_masks": sum(r.mask_empty is True for r in records),
        "size_mismatches": sum(r.size_match is False for r in records),
        "corrupt_images": sum(bool(r.image_error) for r in records),
        "corrupt_masks": sum(bool(r.mask_error) for r in records),
        "exact_duplicate_groups_total": sum(1 for g in hash_groups.values() if len(g) > 1),
        "cross_split_exact_duplicate_groups": cross_split_groups,
        "cross_split_affected_files": affected_files,
        "cross_domain_exact_duplicate_groups": cross_domain_groups,
        "duplicate_filename_groups": len(filename_rows),
        "outputs": {k: str(v) for k, v in outputs.items() if k != "report_json"},
    }
    outputs["report_json"].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("DATASET AUDIT SUMMARY")
    print("=" * 80)
    for k, v in report.items():
        if k not in {"outputs", "domains", "data_root"}:
            print(f"{k}: {v}")
    if pair_rows:
        print("\nTop cross-split duplicate pairs:")
        for row in pair_rows[:20]:
            print(f"  {row['left_domain']}/{row['left_split']} <-> {row['right_domain']}/{row['right_split']}: {row['duplicate_group_count']}")
    print("\nReports:")
    for path in outputs.values():
        print(f"  {path}")

    if cross_split_groups:
        print("\n[warning] Cross-split exact duplicates found; do not use this split for final paper results.")
        return 2 if args.fail_on_leakage else 0
    print("\n[ok] No cross-split exact duplicates found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

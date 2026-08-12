from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description="Detect exact image leakage across train/val/test splits")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--output", default="dataset_split_audit.csv")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.data_root)
    records = []
    for domain_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for split in ("train", "val", "test"):
            image_dir = domain_dir / split / "images"
            if not image_dir.is_dir():
                continue
            for path in sorted(image_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                    records.append({
                        "domain": domain_dir.name,
                        "split": split,
                        "sample_id": path.stem,
                        "path": str(path.resolve()),
                        "sha256": sha256(path),
                    })

    by_hash = defaultdict(list)
    for row in records:
        by_hash[row["sha256"]].append(row)
    leaks = []
    for digest, group in by_hash.items():
        splits = {row["split"] for row in group}
        if len(splits) > 1:
            for row in group:
                leaks.append({**row, "duplicate_group_size": len(group), "splits_in_group": ",".join(sorted(splits))})

    with Path(args.output).open("w", newline="", encoding="utf-8") as f:
        fields = ["domain", "split", "sample_id", "path", "sha256", "duplicate_group_size", "splits_in_group"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(leaks)
    print(f"scanned={len(records)} cross-split exact duplicates={len({row['sha256'] for row in leaks})}")
    print(f"audit report: {Path(args.output).resolve()}")
    if leaks:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Select representative ultrasound image/mask pairs for a TaskSegment figure.

The script scans six domains, pairs images with masks, scores representative
training samples, and exports publication-ready PNGs plus contact sheets.
It only reads the source dataset and never modifies it.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import math
import re
import sys
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"}
MASK_WORDS = {
    "mask", "masks", "label", "labels", "annotation", "annotations",
    "groundtruth", "groundtruths", "gt", "gts", "segmentation",
    "segmentations", "manual", "manuals",
}
SPLIT_ALIASES = {
    "train": {"train", "training"},
    "val": {"val", "valid", "validation"},
    "test": {"test", "testing"},
}
MASK_SUFFIX = re.compile(
    r"(?i)(?:[_\-\s]?(?:mask|masks|label|labels|seg|segmentation|gt|"
    r"groundtruth|annotation|manual))+$"
)
IMAGE_SUFFIX = re.compile(r"(?i)(?:[_\-\s]?(?:img|image|scan|us))+$")


@dataclass
class Candidate:
    domain: str
    image_path: str
    mask_path: str
    score: float
    area_ratio: float
    center_score: float
    contrast_score: float
    border_score: float
    component_score: float
    num_components: int
    width: int
    height: int


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def canonical_stem(path: Path) -> str:
    stem = MASK_SUFFIX.sub("", path.stem)
    stem = IMAGE_SUFFIX.sub("", stem)
    return norm(stem)


def is_mask(path: Path) -> bool:
    parent_tokens = [norm(x) for x in path.parts[:-1]]
    return any(x in MASK_WORDS for x in parent_tokens) or bool(
        MASK_SUFFIX.search(path.stem.lower())
    )


def matches_domain(path: Path, domain: str) -> bool:
    target = norm(domain)
    return any(target == norm(x) or target in norm(x) for x in path.parts)


def matches_split(path: Path, split: Optional[str]) -> bool:
    if not split:
        return True
    wanted = SPLIT_ALIASES.get(split.lower(), {norm(split)})
    return bool({norm(x) for x in path.parts} & wanted)


def parent_signature(path: Path) -> str:
    ignored = MASK_WORDS | {"image", "images", "img", "imgs", "scan", "scans"}
    return "/".join(
        norm(x) for x in path.parent.parts if norm(x) and norm(x) not in ignored
    )


def best_mask(image_path: Path, masks: list[Path]) -> Path:
    image_sig = parent_signature(image_path)

    def rank(mask_path: Path):
        similarity = difflib.SequenceMatcher(
            None, image_sig, parent_signature(mask_path)
        ).ratio()
        same_ext = int(image_path.suffix.lower() == mask_path.suffix.lower())
        return similarity, same_ext, str(mask_path)

    return max(masks, key=rank)


def discover_pairs(
    data_root: Path,
    domain: str,
    split: Optional[str],
    strict_split: bool,
) -> list[tuple[Path, Path]]:
    files = [
        p for p in data_root.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTS and matches_domain(p, domain)
    ]
    if not files:
        return []

    split_files = [p for p in files if matches_split(p, split)]
    if split and split_files:
        files = split_files
    elif split and strict_split:
        return []
    elif split:
        print(
            f"[WARN] {domain}: path中没有明确的'{split}'目录，改为搜索该域全部文件。",
            file=sys.stderr,
        )

    masks = [p for p in files if is_mask(p)]
    images = [p for p in files if not is_mask(p)]

    index: dict[str, list[Path]] = defaultdict(list)
    for p in masks:
        key = canonical_stem(p)
        if key:
            index[key].append(p)

    pairs = []
    seen = set()
    for image_path in images:
        candidates = index.get(canonical_stem(image_path), [])
        if not candidates:
            continue
        mask_path = best_mask(image_path, candidates)
        key = (str(image_path.resolve()), str(mask_path.resolve()))
        if key not in seen:
            seen.add(key)
            pairs.append((image_path, mask_path))
    return pairs


def load_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    else:
        with Image.open(path) as im:
            arr = np.asarray(im)
    arr = np.squeeze(np.asarray(arr))
    if arr.ndim == 3:
        arr = arr[..., :3].astype(np.float32).mean(axis=2)
    if arr.ndim != 2:
        raise ValueError(f"unsupported shape {arr.shape}: {path}")
    return arr


def normalize_image(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    values = arr[finite]
    lo, hi = np.percentile(values, [1.0, 99.0])
    if hi <= lo:
        lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    out = np.clip((arr - lo) / (hi - lo), 0, 1)
    out[~finite] = 0
    return np.round(out * 255).astype(np.uint8)


def binarize_mask(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=bool)
    values = arr[finite]
    vmax = float(values.max())
    if vmax <= 0:
        return np.zeros(arr.shape, dtype=bool)
    if np.unique(values).size <= 16:
        return arr > 0
    threshold = 0.5 if vmax <= 1 else 0.5 * vmax
    return arr > threshold


def resize_mask(mask: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    pil = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    return np.asarray(pil.resize(size_wh, Image.Resampling.NEAREST)) > 0


def count_components(mask: np.ndarray) -> int:
    try:
        from scipy import ndimage  # type: ignore
        _, n = ndimage.label(mask)
        return int(n)
    except Exception:
        pass

    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))
    count = 0
    for y, x in zip(*np.nonzero(mask)):
        if visited[y, x]:
            continue
        count += 1
        queue = deque([(int(y), int(x))])
        visited[y, x] = True
        while queue:
            cy, cx = queue.popleft()
            for dy, dx in neighbors:
                ny, nx = cy + dy, cx + dx
                if (
                    0 <= ny < h and 0 <= nx < w and mask[ny, nx]
                    and not visited[ny, nx]
                ):
                    visited[ny, nx] = True
                    queue.append((ny, nx))
    return count


def evaluate(
    domain: str,
    image_path: Path,
    mask_path: Path,
    target_area: float,
) -> Optional[Candidate]:
    try:
        image = normalize_image(load_array(image_path))
        mask = binarize_mask(load_array(mask_path))
    except Exception as exc:
        print(f"[WARN] 读取失败: {image_path} / {mask_path}: {exc}", file=sys.stderr)
        return None

    h, w = image.shape
    if mask.shape != image.shape:
        mask = resize_mask(mask, (w, h))

    area = float(mask.mean())
    if area < 0.002 or area > 0.75:
        return None

    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None

    cx = float(xs.mean()) / max(w - 1, 1)
    cy = float(ys.mean()) / max(h - 1, 1)
    distance = math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2) / math.sqrt(0.5)
    center_score = float(np.clip(1 - distance, 0, 1))

    border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    border_score = float(np.clip(1 - 5 * float(border.mean()), 0, 1))

    fg = image[mask].astype(np.float32)
    bg = image[~mask].astype(np.float32)
    std = float(np.std(image)) + 1e-6
    contrast = abs(float(fg.mean()) - float(bg.mean())) / std
    contrast_score = float(np.clip(contrast / 1.5, 0, 1))

    n_components = count_components(mask)
    component_score = 1 / (1 + 0.35 * max(n_components - 1, 0))
    size_score = math.exp(
        -0.5 * (abs(math.log(area + 1e-8) - math.log(target_area)) / 0.85) ** 2
    )

    score = (
        0.35 * size_score
        + 0.20 * center_score
        + 0.20 * contrast_score
        + 0.15 * border_score
        + 0.10 * component_score
    )

    return Candidate(
        domain=domain,
        image_path=str(image_path),
        mask_path=str(mask_path),
        score=float(score),
        area_ratio=area,
        center_score=center_score,
        contrast_score=contrast_score,
        border_score=border_score,
        component_score=float(component_score),
        num_components=n_components,
        width=w,
        height=h,
    )


def overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> Image.Image:
    base = np.stack([image, image, image], axis=-1).astype(np.float32)
    red = np.zeros_like(base)
    red[..., 0] = 255
    m = mask[..., None].astype(np.float32)
    result = base * (1 - alpha * m) + red * (alpha * m)
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), mode="RGB")


def save_candidate(candidate: Candidate, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    image = normalize_image(load_array(Path(candidate.image_path)))
    mask = binarize_mask(load_array(Path(candidate.mask_path)))
    h, w = image.shape
    if mask.shape != image.shape:
        mask = resize_mask(mask, (w, h))

    Image.fromarray(image, mode="L").save(out_dir / "image.png")
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(out_dir / "mask.png")
    overlay(image, mask).save(out_dir / "overlay.png")
    (out_dir / "metadata.json").write_text(
        json.dumps(asdict(candidate), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def contact_sheet(
    items: list[tuple[str, Path]],
    output: Path,
    columns: int,
    thumb: tuple[int, int] = (260, 200),
) -> None:
    if not items:
        return
    rows = math.ceil(len(items) / columns)
    pad, label_h = 16, 34
    cell_w, cell_h = thumb[0] + 2 * pad, thumb[1] + label_h + 2 * pad
    canvas = Image.new("RGB", (columns * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for i, (label, path) in enumerate(items):
        row, col = divmod(i, columns)
        x0, y0 = col * cell_w + pad, row * cell_h + pad
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail(thumb, Image.Resampling.LANCZOS)
            x = x0 + (thumb[0] - im.width) // 2
            y = y0 + label_h + (thumb[1] - im.height) // 2
            canvas.paste(im, (x, y))
        draw.text((x0, y0 + 8), label, fill="black", font=font)

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("figure_samples"))
    parser.add_argument(
        "--domains", nargs="+",
        default=["thyroid", "TN3K", "BUSI_WHU", "BUS-BRA", "OTU", "prostate"],
    )
    parser.add_argument("--episode-domain", default="TN3K")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--strict-split", action="store_true")
    parser.add_argument("--target-area", type=float, default=0.12)
    parser.add_argument("--max-evaluate", type=int, default=1000)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    if not data_root.is_dir():
        print(f"ERROR: data root不存在: {data_root}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    all_candidates: dict[str, list[Candidate]] = {}
    selected: dict[str, Candidate] = {}

    for domain in args.domains:
        pairs = discover_pairs(data_root, domain, args.split, args.strict_split)
        print(f"[{domain}] 配对数量: {len(pairs)}")
        scored = []
        for image_path, mask_path in pairs[: args.max_evaluate]:
            item = evaluate(domain, image_path, mask_path, args.target_area)
            if item is not None:
                scored.append(item)
        scored.sort(key=lambda x: (-x.score, x.image_path))
        all_candidates[domain] = scored

        if not scored:
            print(f"[WARN] {domain}: 没有找到有效的非空mask样本", file=sys.stderr)
            continue
        selected[domain] = scored[0]
        save_candidate(scored[0], output_dir / "domains" / domain)
        print(
            f"[{domain}] selected score={scored[0].score:.4f}, "
            f"area={scored[0].area_ratio:.4f}\n"
            f"  image={scored[0].image_path}\n"
            f"  mask ={scored[0].mask_path}"
        )

    episode_items = all_candidates.get(args.episode_domain, [])
    if len(episode_items) < 2:
        print(
            f"ERROR: episode-domain={args.episode_domain} 少于两个有效样本",
            file=sys.stderr,
        )
        return 3

    support = episode_items[0]
    query = next((x for x in episode_items[1:] if x.image_path != support.image_path), None)
    if query is None:
        print("ERROR: 找不到与support不同的query样本", file=sys.stderr)
        return 4

    episode_dir = output_dir / "episode_example"
    support_dir, query_dir = episode_dir / "_support", episode_dir / "_query"
    save_candidate(support, support_dir)
    save_candidate(query, query_dir)
    episode_dir.mkdir(parents=True, exist_ok=True)

    for prefix, src in (("support", support_dir), ("query", query_dir)):
        for name in ("image.png", "mask.png", "overlay.png"):
            (episode_dir / f"{prefix}_{name}").write_bytes((src / name).read_bytes())
        for p in src.iterdir():
            p.unlink()
        src.rmdir()

    episode_meta = {
        "domain": args.episode_domain,
        "support": asdict(support),
        "query": asdict(query),
        "note": "Support和Query是同一域内的两个不同训练样本；脚本不推断患者ID。",
    }
    (episode_dir / "metadata.json").write_text(
        json.dumps(episode_meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    rows = [asdict(selected[d]) for d in args.domains if d in selected]
    if rows:
        with (output_dir / "selection_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    manifest = {
        "data_root": str(data_root),
        "split": args.split,
        "selected": {d: asdict(x) for d, x in selected.items()},
        "episode_example": episode_meta,
    }
    (output_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    contact_sheet(
        [
            (d, output_dir / "domains" / d / "overlay.png")
            for d in args.domains if d in selected
        ],
        output_dir / "domain_contact_sheet.png",
        columns=3,
    )
    contact_sheet(
        [
            (f"Support ({args.episode_domain})", episode_dir / "support_overlay.png"),
            (f"Query ({args.episode_domain})", episode_dir / "query_overlay.png"),
            ("Support mask", episode_dir / "support_mask.png"),
            ("Query mask", episode_dir / "query_mask.png"),
        ],
        output_dir / "episode_contact_sheet.png",
        columns=2,
    )

    print("\n完成。")
    print(f"输出目录: {output_dir}")
    print(f"六域预览: {output_dir / 'domain_contact_sheet.png'}")
    print(f"Support/Query预览: {output_dir / 'episode_contact_sheet.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

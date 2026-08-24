from __future__ import annotations
#
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from paperlab.progress import paper_tqdm as tqdm

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _resize_pil(img: Image.Image, image_size: Tuple[int, int], interpolation=Image.BILINEAR) -> Image.Image:
    h, w = int(image_size[0]), int(image_size[1])
    return img.resize((w, h), interpolation)


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[None]
    else:
        arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(arr))


class PerImageZScoreNormalize:
    def __init__(self, clamp_value: float = 5.0, eps: float = 1e-6):
        self.clamp_value = float(clamp_value)
        self.eps = float(eps)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = tensor.mean()
        std = tensor.std().clamp_min(self.eps)
        return ((tensor - mean) / std).clamp(-self.clamp_value, self.clamp_value) / self.clamp_value


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        torch.use_deterministic_algorithms(False)


def _apply_clahe_pil(img: Image.Image, clip_limit=2.0, tile_grid_size=(8, 8)) -> Image.Image:
    arr = np.asarray(img, dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid_size))
    return Image.fromarray(clahe.apply(arr))


def _default_clahe_cfg_by_domain() -> Dict[str, Dict[str, float]]:
    return {
        "thyroid": {"prob": 0.18, "clip_limit": 1.4, "tile_grid_size": 8},
        "TN3K": {"prob": 0.18, "clip_limit": 1.4, "tile_grid_size": 8},
        "BUSI_WHU": {"prob": 0.30, "clip_limit": 1.8, "tile_grid_size": 8},
        "BUS-BRA": {"prob": 0.30, "clip_limit": 1.8, "tile_grid_size": 8},
        "OTU": {"prob": 0.30, "clip_limit": 1.8, "tile_grid_size": 8},
        "prostate": {"prob": 0.30, "clip_limit": 1.8, "tile_grid_size": 8},
    }


class MultiOrganDataset(Dataset):
    """Multi-domain binary segmentation dataset with stable sample metadata.

    When ``cache_in_memory`` is enabled, the cache stores resized uint8 image and
    mask arrays, not normalized float tensors. This reduces RAM usage by roughly
    6x compared with caching float32 images plus int64 masks and also allows
    stochastic CPU CLAHE to remain active on every access.
    """

    def __init__(
        self,
        data_roots: Dict[str, str],
        image_transform=None,
        split: Optional[str] = None,
        image_size: Tuple[int, int] = (512, 512),
        seed: int = 42,
        query_positive_ratio: float = 0.7,
        prefer_empty_queries: bool = True,
        cache_in_memory: bool = False,
        use_cpu_clahe: bool = False,
        cpu_clahe_cfg_by_domain: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        self.image_transform = image_transform
        self.samples: List[dict] = []
        self.organ_to_indices: Dict[str, List[int]] = {}
        self.organ_to_positive_indices: Dict[str, List[int]] = {}
        self.organ_to_empty_indices: Dict[str, List[int]] = {}
        self.split = split
        self.image_size = tuple(int(x) for x in image_size)
        self.seed = int(seed)
        self.query_positive_ratio = float(np.clip(query_positive_ratio, 0.0, 1.0))
        self.prefer_empty_queries = bool(prefer_empty_queries)
        self.cache_in_memory = bool(cache_in_memory)
        self.use_cpu_clahe = bool(use_cpu_clahe)
        self.cpu_clahe_cfg_by_domain = cpu_clahe_cfg_by_domain or _default_clahe_cfg_by_domain()
        self._cache: Dict[int, Tuple[np.ndarray, np.ndarray, str]] = {}

        print(f"正在初始化多器官数据集 (split: {split})...")
        for organ_name, root_path in data_roots.items():
            split_names = [split] if split is not None else ["train", "val", "test"]
            for split_name in split_names:
                split_path = Path(root_path) / str(split_name)
                img_dir, mask_dir = split_path / "images", split_path / "masks"
                if not img_dir.is_dir() or not mask_dir.is_dir():
                    continue

                img_files = sorted(
                    p for p in img_dir.iterdir()
                    if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in _IMAGE_EXTENSIONS
                )
                valid_count = positive_count = missing_mask_count = 0
                for img_path in img_files:
                    candidates = [
                        mask_dir / f"{img_path.stem}.png",
                        mask_dir / f"{img_path.stem}.PNG",
                        mask_dir / img_path.name,
                    ]
                    mask_path = next((p for p in candidates if p.is_file()), None)
                    if mask_path is None:
                        missing_mask_count += 1
                        continue

                    with Image.open(mask_path) as label_pil:
                        label_np = np.asarray(label_pil.convert("L"))
                    has_foreground = bool((label_np > 0).any())
                    original_shape = tuple(int(x) for x in label_np.shape[-2:])
                    sample = {
                        "image": str(img_path),
                        "mask": str(mask_path),
                        "organ": str(organ_name),
                        "split": str(split_name),
                        "sample_id": img_path.stem,
                        "has_foreground": has_foreground,
                        "original_shape": original_shape,
                    }
                    idx = len(self.samples)
                    self.samples.append(sample)
                    self.organ_to_indices.setdefault(organ_name, []).append(idx)
                    target = self.organ_to_positive_indices if has_foreground else self.organ_to_empty_indices
                    target.setdefault(organ_name, []).append(idx)
                    positive_count += int(has_foreground)
                    valid_count += 1

                self.organ_to_positive_indices.setdefault(organ_name, [])
                self.organ_to_empty_indices.setdefault(organ_name, [])
                print(
                    f" - 加载 {organ_name}/{split_name}: 总样本 {valid_count} | "
                    f"前景样本 {positive_count} | 空掩码 {valid_count-positive_count} | 缺失掩码 {missing_mask_count}"
                )

        if self.cache_in_memory:
            estimate_gib = self.estimated_full_cache_bytes() / (1024 ** 3)
            print(
                f"[cache] split={self.split} CPU RAM 缓存已启用（uint8 resized image+mask，惰性加载），"
                f"预计全部缓存约 {estimate_gib:.2f} GiB"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def get_sample_info(self, idx: int) -> dict:
        return dict(self.samples[int(idx)])

    def get_organ_indices(self, organ_name: str, positive_only: bool = False) -> List[int]:
        source = self.organ_to_positive_indices if positive_only else self.organ_to_indices
        return list(source.get(organ_name, []))

    def load_original_mask(self, idx: int) -> np.ndarray:
        """Compatibility helper. Paper metrics should use ``load_resized_mask``."""
        with Image.open(self.samples[int(idx)]["mask"]) as label:
            return (np.asarray(label.convert("L")) > 0).astype(np.uint8)

    def load_resized_mask(self, idx: int) -> np.ndarray:
        """Return the binary target on the unified dataset grid (normally 512x512)."""
        _, label, _ = self._get_raw_item(int(idx))
        return (label > 0).astype(np.uint8, copy=True)

    def _sample_indices(self, pool: List[int], num_samples: int, replace: Optional[bool] = None) -> List[int]:
        if num_samples <= 0 or not pool:
            return []
        if replace is None:
            replace = len(pool) < num_samples
        return np.random.choice(pool, num_samples, replace=replace).tolist()

    def get_organ_samples(self, organ_name: str, num_samples: int, positive_only: bool = False) -> List[int]:
        indices = self.get_organ_indices(organ_name, positive_only)
        if not indices:
            raise ValueError(f"器官/域 {organ_name} 没有可用样本")
        if num_samples >= len(indices):
            return indices  # stable evaluation order
        return self._sample_indices(indices, num_samples, replace=False)

    def sample_episode_indices(self, organ_name: str, num_support: int, num_query: int) -> Tuple[List[int], List[int]]:
        pos_pool = self.get_organ_indices(organ_name, positive_only=True)
        all_pool = self.get_organ_indices(organ_name, positive_only=False)
        empty_pool = list(self.organ_to_empty_indices.get(organ_name, []))
        if not pos_pool or not all_pool:
            raise ValueError(f"器官/域 {organ_name} 没有足够样本")

        support = self._sample_indices(pos_pool, num_support)
        used = set(support)
        n_pos = min(num_query, max(1, int(round(num_query * self.query_positive_ratio)))) if num_query else 0
        pos_candidates = [i for i in pos_pool if i not in used] or pos_pool
        query = self._sample_indices(pos_candidates, n_pos, replace=len(pos_candidates) < n_pos)
        remaining = num_query - len(query)
        if remaining > 0:
            candidates = []
            if self.prefer_empty_queries:
                candidates.extend(i for i in empty_pool if i not in used)
            candidates.extend(i for i in all_pool if i not in used and i not in candidates)
            candidates = candidates or all_pool
            query.extend(self._sample_indices(candidates, remaining, replace=len(candidates) < remaining))
        return support, query

    def _maybe_apply_cpu_clahe(self, img: Image.Image, organ_name: str) -> Image.Image:
        if not self.use_cpu_clahe:
            return img
        cfg = self.cpu_clahe_cfg_by_domain.get(organ_name)
        if not cfg or random.random() >= float(cfg.get("prob", 0.0)):
            return img
        grid = int(cfg.get("tile_grid_size", 8))
        return _apply_clahe_pil(img, float(cfg.get("clip_limit", 2.0)), (grid, grid))

    def _load_raw_item_uncached(self, idx: int) -> Tuple[np.ndarray, np.ndarray, str]:
        sample = self.samples[int(idx)]
        with Image.open(sample["image"]) as p:
            img = _resize_pil(p.convert("L"), self.image_size, Image.BILINEAR)
        with Image.open(sample["mask"]) as p:
            label = _resize_pil(p.convert("L"), self.image_size, Image.NEAREST)
        img_np = np.ascontiguousarray(np.asarray(img, dtype=np.uint8))
        label_np = np.ascontiguousarray((np.asarray(label) > 0).astype(np.uint8))
        return img_np, label_np, sample["organ"]

    def _get_raw_item(self, idx: int) -> Tuple[np.ndarray, np.ndarray, str]:
        idx = int(idx)
        if self.cache_in_memory and idx in self._cache:
            return self._cache[idx]
        item = self._load_raw_item_uncached(idx)
        if self.cache_in_memory:
            self._cache[idx] = item
        return item

    def __getitem__(self, idx: int):
        img_np, label_np, organ = self._get_raw_item(int(idx))
        # Copy before stochastic transforms so cached arrays remain immutable.
        img = Image.fromarray(img_np.copy(), mode="L")
        img = self._maybe_apply_cpu_clahe(img, organ)
        img_tensor = _pil_to_tensor(img)
        if self.image_transform is not None:
            img_tensor = self.image_transform(img_tensor)
        label_tensor = torch.from_numpy(label_np.astype(np.int64, copy=True))
        return img_tensor.contiguous(), label_tensor.contiguous(), organ

    def warm_cache(self, indices: Optional[Iterable[int]] = None, show_progress: bool = True) -> Dict[str, float]:
        """Eagerly decode and resize samples into the CPU RAM cache.

        This is useful for validation and test, where deterministic evaluation is
        preferred. Training can use the same cache lazily to avoid a long startup.
        """
        if not self.cache_in_memory:
            return self.cache_info()
        seq = list(range(len(self))) if indices is None else [int(i) for i in indices]
        iterator = tqdm(seq, desc=f"预加载 {self.split or 'dataset'} CPU cache", leave=False) if show_progress else seq
        for idx in iterator:
            self._get_raw_item(idx)
        info = self.cache_info()
        print(
            f"[cache] split={self.split} cached={int(info['cached_items'])}/{int(info['total_items'])} "
            f"RAM={info['cached_gib']:.2f} GiB"
        )
        return info

    def clear_cache(self) -> None:
        self._cache.clear()

    def estimated_full_cache_bytes(self) -> int:
        h, w = self.image_size
        # one uint8 image and one uint8 binary mask per sample
        return int(len(self.samples) * h * w * 2)

    def cache_info(self) -> Dict[str, float]:
        cached_bytes = int(sum(img.nbytes + mask.nbytes for img, mask, _ in self._cache.values()))
        return {
            "enabled": float(self.cache_in_memory),
            "cached_items": float(len(self._cache)),
            "total_items": float(len(self.samples)),
            "cached_gib": float(cached_bytes / (1024 ** 3)),
            "estimated_full_gib": float(self.estimated_full_cache_bytes() / (1024 ** 3)),
        }

    @property
    def label_resize(self):
        return lambda label: _resize_pil(label, self.image_size, Image.NEAREST)


def get_image_transform():
    return PerImageZScoreNormalize(clamp_value=5.0)


def get_datasets(
    data_config: Dict[str, str],
    image_size: Tuple[int, int] = (512, 512),
    seed: int = 42,
    train_query_positive_ratio: float = 0.7,
    prefer_empty_train_queries: bool = True,
    cache_in_memory: bool = True,
    use_cpu_clahe: bool = False,
):
    transform = get_image_transform()
    train = MultiOrganDataset(
        data_config, transform, "train", image_size, seed,
        train_query_positive_ratio, prefer_empty_train_queries,
        cache_in_memory, use_cpu_clahe,
    )
    val = MultiOrganDataset(data_config, transform, "val", image_size, seed, cache_in_memory=cache_in_memory)
    test = MultiOrganDataset(data_config, transform, "test", image_size, seed, cache_in_memory=cache_in_memory)
    return train, val, test

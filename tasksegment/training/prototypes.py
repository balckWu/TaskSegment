from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from ..data.datasets import MultiOrganDataset
from .memory import EMATaskMemoryBank


@torch.no_grad()
def rebuild_visual_prototype_bank(
    model: torch.nn.Module,
    train_dataset: MultiOrganDataset,
    device: torch.device,
    samples_per_domain: int = 64,
    batch_size: int = 4,
    seed: int = 42,
) -> Tuple[EMATaskMemoryBank, Dict[str, List[int]]]:
    """Recompute deterministic support-only prototypes with the current model.

    Only training images and training masks are used. No validation/test mask is
    consulted. The sampled index manifest is returned for exact reproducibility.
    """
    was_training = model.training
    model.eval()
    bank = EMATaskMemoryBank(momentum=0.0, token_scope="visual")
    manifest: Dict[str, List[int]] = {}
    rng = np.random.default_rng(int(seed))

    for domain in sorted(train_dataset.organ_to_positive_indices):
        pool = np.asarray(train_dataset.organ_to_positive_indices[domain], dtype=np.int64)
        if pool.size == 0:
            continue
        count = pool.size if samples_per_domain <= 0 else min(int(samples_per_domain), pool.size)
        chosen = np.sort(rng.choice(pool, size=count, replace=False)).tolist()
        manifest[domain] = [int(x) for x in chosen]

        token_sum = None
        token_count = 0
        for start in range(0, len(chosen), max(1, int(batch_size))):
            indices = chosen[start : start + max(1, int(batch_size))]
            xs = torch.stack([train_dataset[i][0] for i in indices]).to(device)
            ys = torch.stack([train_dataset[i][1] for i in indices]).unsqueeze(1).float().to(device)
            pyramid = model.encoder.forward_features(xs)
            visual = model.task_encoder.forward_visual(pyramid["bottleneck"], ys)
            batch_sum = visual.detach().sum(dim=0, keepdim=True)
            token_sum = batch_sum if token_sum is None else token_sum + batch_sum
            token_count += visual.shape[0]

        if token_sum is None or token_count == 0:
            continue
        bank.set(domain, token_sum / float(token_count))

    model.train(was_training)
    return bank, manifest

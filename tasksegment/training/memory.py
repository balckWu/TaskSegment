from __future__ import annotations

import random
from typing import Dict, List, Mapping, Optional

import numpy as np
import torch


def make_domain_schedule(
    domains: List[str],
    total_iters: int,
    domain_sizes: Mapping[str, int] | None = None,
    alpha: float = 0.0,
) -> List[str]:
    """Create a shuffled domain schedule.

    ``alpha=0`` is exactly domain-balanced. ``alpha=1`` samples in proportion
    to dataset size. Intermediate values provide temperature sampling.
    """
    if not domains:
        raise ValueError("domains cannot be empty")
    total_iters = int(total_iters)
    alpha = float(alpha)
    if total_iters <= 0:
        return []

    if domain_sizes is None or alpha <= 0.0:
        reps, rem = divmod(total_iters, len(domains))
        schedule: List[str] = []
        for _ in range(reps):
            chunk = list(domains)
            random.shuffle(chunk)
            schedule.extend(chunk)
        if rem:
            tail = list(domains)
            random.shuffle(tail)
            schedule.extend(tail[:rem])
        return schedule

    sizes = np.asarray([max(1, int(domain_sizes.get(d, 1))) for d in domains], dtype=np.float64)
    probs = np.power(sizes, alpha)
    probs /= probs.sum()
    sampled = np.random.choice(np.asarray(domains, dtype=object), size=total_iters, replace=True, p=probs)
    return sampled.tolist()


# Backward-compatible name.
def make_balanced_domain_schedule(domains: List[str], total_iters: int) -> List[str]:
    return make_domain_schedule(domains, total_iters, domain_sizes=None, alpha=0.0)


class TaskPrototypeBank:
    """Deterministic per-domain visual task prototypes.

    DiceV2 deliberately removes the unused online EMA path. The bank is rebuilt
    from a fixed manifest of training supports with the current model, so the
    stored representation exactly matches validation and inference.
    """

    def __init__(self, momentum: float = 0.0, token_scope: str = "visual") -> None:
        # ``momentum`` is accepted only for old call sites/checkpoints.
        self.momentum = float(momentum)
        self.token_scope = str(token_scope)
        self.bank: Dict[str, torch.Tensor] = {}

    def has(self, class_id: str) -> bool:
        return str(class_id) in self.bank

    def get(self, class_id: str, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
        value = self.bank.get(str(class_id))
        if value is None:
            return None
        return value.to(device) if device is not None else value

    @torch.no_grad()
    def set(self, class_id: str, task_tokens: torch.Tensor) -> None:
        if task_tokens.ndim != 3 or task_tokens.shape[0] != 1:
            raise ValueError("prototype must have shape [1,T,C]")
        self.bank[str(class_id)] = task_tokens.detach().cpu().clone()

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {key: value.clone().cpu() for key, value in self.bank.items()}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.bank = {str(key): value.clone().cpu() for key, value in state_dict.items()}

    def summary(self) -> str:
        if not self.bank:
            return "empty"
        return ", ".join(f"{key}: {tuple(value.shape)}" for key, value in self.bank.items())


# Existing inference/checkpoint modules import the old class name.
EMATaskMemoryBank = TaskPrototypeBank

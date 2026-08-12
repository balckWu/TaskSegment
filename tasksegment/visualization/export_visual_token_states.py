from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import matplotlib

# Prediction is normally executed on a headless server.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


_EXPORTED = False


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _normalise_01(x: torch.Tensor) -> np.ndarray:
    x = x.detach().float().cpu()
    finite = torch.isfinite(x)
    if not bool(finite.any()):
        return torch.zeros_like(x).numpy()

    valid = x[finite]
    lo = valid.min()
    hi = valid.max()
    x = torch.nan_to_num(x, nan=float(lo), posinf=float(hi), neginf=float(lo))
    if float(hi - lo) < 1e-8:
        return torch.zeros_like(x).numpy()
    return ((x - lo) / (hi - lo)).clamp(0.0, 1.0).numpy()


def _prepare_image(x: torch.Tensor) -> np.ndarray:
    """Convert [H,W] or [C,H,W] support input to a displayable 2-D map."""
    x = x.detach().float().cpu()
    if x.ndim == 3:
        if x.shape[0] == 1:
            x = x[0]
        else:
            x = x.mean(dim=0)
    if x.ndim != 2:
        raise ValueError(f"Expected a 2-D image after channel reduction, got {tuple(x.shape)}")
    return _normalise_01(x)


def _save_map(
    image: np.ndarray,
    path: Path,
    *,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(3.0, 3.0), dpi=300)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(
        image,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.axis("off")
    fig.savefig(path, dpi=300, bbox_inches=None, pad_inches=0, facecolor="white")
    plt.close(fig)


def _save_token_matrix(tokens: torch.Tensor, path: Path, title: str) -> None:
    """Save a token matrix, normally [5, 1024], as a heatmap."""
    matrix = tokens.detach().float().cpu()
    if matrix.ndim != 2:
        raise ValueError(f"Token matrix must be 2-D, got {tuple(matrix.shape)}")

    matrix_np = matrix.numpy()
    rows, width = matrix_np.shape
    figure_width = 9.0 if width >= 512 else 6.0
    figure_height = max(2.3, 0.45 * rows + 1.6)

    fig = plt.figure(figsize=(figure_width, figure_height), dpi=240)
    ax = fig.add_axes([0.10, 0.28, 0.84, 0.58])
    im = ax.imshow(matrix_np, aspect="auto", interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Embedding dimension", fontsize=9)
    ax.set_ylabel("Token", fontsize=9)

    if rows == 5:
        ax.set_yticks(range(5))
        ax.set_yticklabels(
            ["foreground", "context 1", "context 2", "context 3", "context 4"],
            fontsize=8,
        )
    else:
        ax.set_yticks(range(rows))
        ax.set_yticklabels([str(i + 1) for i in range(rows)], fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _raw_visual_sequence(raw_visual_tokens: torch.Tensor) -> torch.Tensor | None:
    """Best-effort conversion to [K, 5, D]."""
    raw = raw_visual_tokens.detach()

    # Remove singleton wrapper dimensions sometimes introduced by batching.
    while raw.ndim > 3 and raw.shape[0] == 1:
        raw = raw[0]

    if raw.ndim == 2:
        # A single already-aggregated five-token sequence.
        return raw.unsqueeze(0)

    if raw.ndim == 3:
        return raw

    return None


def _episode_visual_sequence(visual_tokens: torch.Tensor) -> torch.Tensor | None:
    """Best-effort conversion to one [5, D] episode prototype."""
    visual = visual_tokens.detach()

    while visual.ndim > 3 and visual.shape[0] == 1:
        visual = visual[0]

    if visual.ndim == 2:
        return visual

    if visual.ndim == 3:
        # encode_task may expand the same episode prototype to the query batch.
        return visual[0]

    return None


def _resolve_domain(
    domain_ids: torch.Tensor | Sequence[int] | int | None,
    domain_names: Sequence[str] | None,
) -> tuple[int | None, str]:
    domain_id: int | None = None

    if domain_ids is not None:
        ids = torch.as_tensor(domain_ids).detach().cpu().reshape(-1)
        if ids.numel() > 0:
            domain_id = int(ids[0].item())

    if domain_id is not None and domain_names is not None:
        names = tuple(str(name) for name in domain_names)
        if 0 <= domain_id < len(names):
            return domain_id, names[domain_id]

    if domain_id is not None:
        return domain_id, f"domain_{domain_id}"

    return None, "unknown_domain"


def maybe_export_visual_token_states(
    *,
    xs: torch.Tensor | None,
    ys: torch.Tensor | None,
    support_pyramid: dict[str, torch.Tensor] | None,
    raw_visual_tokens: torch.Tensor | None,
    visual_tokens: torch.Tensor | None,
    domain_ids: torch.Tensor | Sequence[int] | int | None = None,
    domain_names: Sequence[str] | None = None,
) -> None:
    """Export the real TSV4 support bottleneck, resized mask and visual tokens.

    The function does nothing unless TSV4_EXPORT_VISUAL_STATES=1 is present in
    the environment. It exports once per prediction process. Set
    TSV4_EXPORT_DOMAIN to a domain name such as ``thyroid`` to wait until that
    domain is encountered.
    """
    global _EXPORTED

    if _EXPORTED or not _env_flag("TSV4_EXPORT_VISUAL_STATES"):
        return

    if xs is None or ys is None or support_pyramid is None:
        return

    if raw_visual_tokens is None or visual_tokens is None:
        return

    if "bottleneck" not in support_pyramid:
        raise KeyError(
            "The TSV4 support pyramid has no 'bottleneck' key. "
            f"Available keys: {list(support_pyramid.keys())}"
        )

    domain_id, domain_name = _resolve_domain(domain_ids, domain_names)
    requested_domain = os.environ.get("TSV4_EXPORT_DOMAIN", "").strip()
    if requested_domain and requested_domain.lower() != domain_name.lower():
        return

    output_root = Path(
        os.environ.get(
            "TSV4_EXPORT_VISUAL_DIR",
            "/tank/wk/TSV4/paper_artifacts/figure_exports/visual_task_token_states",
        )
    ).expanduser()
    output_dir = output_root / domain_name
    output_dir.mkdir(parents=True, exist_ok=True)

    bottleneck = support_pyramid["bottleneck"]

    if xs.ndim != 4:
        raise ValueError(f"xs must have shape [K,C,H,W], got {tuple(xs.shape)}")

    if ys.ndim == 3:
        ys_4d = ys.unsqueeze(1)
    elif ys.ndim == 4:
        ys_4d = ys
    else:
        raise ValueError(f"ys must have shape [K,H,W] or [K,1,H,W], got {tuple(ys.shape)}")

    if bottleneck.ndim != 4:
        raise ValueError(
            "support_pyramid['bottleneck'] must be [K,C,Hb,Wb], "
            f"got {tuple(bottleneck.shape)}"
        )

    if not (xs.shape[0] == ys_4d.shape[0] == bottleneck.shape[0]):
        raise ValueError(
            "Support batch mismatch: "
            f"xs={tuple(xs.shape)}, ys={tuple(ys_4d.shape)}, "
            f"bottleneck={tuple(bottleneck.shape)}"
        )

    resized_masks = F.interpolate(
        ys_4d.float(),
        size=bottleneck.shape[-2:],
        mode="nearest",
    )
    masked_bottleneck = bottleneck * resized_masks

    max_supports = max(1, int(os.environ.get("TSV4_EXPORT_MAX_SUPPORTS", "4")))
    support_count = min(int(xs.shape[0]), max_supports)

    # Keep raw tensors for later analysis or alternative visualisation.
    tensor_payload = {
        "xs": xs[:support_count].detach().cpu(),
        "ys": ys_4d[:support_count].detach().cpu(),
        "support_bottleneck": bottleneck[:support_count].detach().cpu(),
        "resized_masks": resized_masks[:support_count].detach().cpu(),
        "masked_bottleneck": masked_bottleneck[:support_count].detach().cpu(),
        "raw_visual_tokens": raw_visual_tokens.detach().cpu(),
        "visual_tokens": visual_tokens.detach().cpu(),
        "domain_id": domain_id,
        "domain_name": domain_name,
    }
    torch.save(tensor_payload, output_dir / "visual_task_token_states.pt")

    for index in range(support_count):
        number = index + 1

        support_image = _prepare_image(xs[index])
        support_mask = ys_4d[index, 0].detach().float().cpu().numpy()
        resized_mask = resized_masks[index, 0].detach().float().cpu().numpy()

        bottleneck_energy = torch.linalg.vector_norm(
            bottleneck[index].detach().float(), dim=0
        )
        masked_energy = torch.linalg.vector_norm(
            masked_bottleneck[index].detach().float(), dim=0
        )

        _save_map(
            support_image,
            output_dir / f"support_{number:02d}_image.png",
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        _save_map(
            support_mask,
            output_dir / f"support_{number:02d}_mask_original.png",
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        _save_map(
            _normalise_01(bottleneck_energy),
            output_dir / f"support_{number:02d}_bottleneck_energy.png",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        _save_map(
            resized_mask,
            output_dir / f"support_{number:02d}_mask_resized.png",
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        _save_map(
            _normalise_01(masked_energy),
            output_dir / f"support_{number:02d}_masked_bottleneck_energy.png",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )

    raw_sequence = _raw_visual_sequence(raw_visual_tokens)
    if raw_sequence is not None:
        for index in range(min(support_count, int(raw_sequence.shape[0]))):
            sample_tokens = raw_sequence[index]
            _save_token_matrix(
                sample_tokens,
                output_dir / f"support_{index + 1:02d}_five_visual_tokens.png",
                title=(
                    f"{domain_name}: support {index + 1} visual tokens "
                    "(1 foreground + 4 context)"
                ),
            )
            if sample_tokens.shape[0] >= 1:
                _save_token_matrix(
                    sample_tokens[0:1],
                    output_dir / f"support_{index + 1:02d}_foreground_token.png",
                    title=f"{domain_name}: support {index + 1} foreground token",
                )
            if sample_tokens.shape[0] >= 5:
                _save_token_matrix(
                    sample_tokens[1:5],
                    output_dir / f"support_{index + 1:02d}_four_context_tokens.png",
                    title=f"{domain_name}: support {index + 1} context tokens",
                )

    episode_sequence = _episode_visual_sequence(visual_tokens)
    if episode_sequence is not None:
        _save_token_matrix(
            episode_sequence,
            output_dir / "episode_visual_prototype.png",
            title=f"{domain_name}: K-support episode visual prototype",
        )

    metadata = {
        "domain_id": domain_id,
        "domain_name": domain_name,
        "support_count_saved": support_count,
        "support_pyramid_keys": list(support_pyramid.keys()),
        "xs_shape": list(xs.shape),
        "ys_shape": list(ys_4d.shape),
        "bottleneck_shape": list(bottleneck.shape),
        "resized_masks_shape": list(resized_masks.shape),
        "raw_visual_tokens_shape": list(raw_visual_tokens.shape),
        "visual_tokens_shape": list(visual_tokens.shape),
        "output_directory": str(output_dir),
    }
    (output_dir / "tensor_shapes.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    readme = f"""TSV4 visual task-token state export
===================================
Domain: {domain_name}
Domain ID: {domain_id}

Use these images in the corrected Visual Task Token Branch figure:
- support_01_bottleneck_energy.png: real shared-encoder support bottleneck
- support_01_mask_resized.png: real nearest-neighbour mask at bottleneck size
- support_01_masked_bottleneck_energy.png: real masked support bottleneck
- support_01_five_visual_tokens.png: one foreground plus four context tokens
- episode_visual_prototype.png: support-averaged visual prototype

The .pt file contains the original tensors without display normalisation.
"""
    (output_dir / "README.txt").write_text(readme, encoding="utf-8")

    print("=" * 88, flush=True)
    print("[TSV4 Visual Export] completed", flush=True)
    print(f"[TSV4 Visual Export] domain: {domain_name} (id={domain_id})", flush=True)
    print(f"[TSV4 Visual Export] output: {output_dir}", flush=True)
    print(f"[TSV4 Visual Export] bottleneck: {tuple(bottleneck.shape)}", flush=True)
    print(
        f"[TSV4 Visual Export] raw visual tokens: {tuple(raw_visual_tokens.shape)}",
        flush=True,
    )
    print(
        f"[TSV4 Visual Export] episode visual tokens: {tuple(visual_tokens.shape)}",
        flush=True,
    )
    print("=" * 88, flush=True)

    _EXPORTED = True

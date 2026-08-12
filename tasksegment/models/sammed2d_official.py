from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_torch_load(path: str | Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_model_state(payload: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"Unsupported checkpoint payload: {type(payload)!r}")
    for key in ("model", "model_state_dict", "state_dict"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            payload = value
            break
    state: Dict[str, torch.Tensor] = {}
    for key, value in payload.items():
        if not torch.is_tensor(value):
            continue
        clean = str(key)
        for prefix in ("module.", "network.", "sam."):
            if clean.startswith(prefix):
                clean = clean[len(prefix):]
        state[clean] = value
    if not state:
        raise ValueError("No tensor model state found in checkpoint")
    return state


def _resize_like(source: torch.Tensor, target: torch.Tensor, key: str) -> torch.Tensor | None:
    if source.shape == target.shape:
        return source
    if key == "image_encoder.pos_embed" and source.ndim == target.ndim == 4:
        value = source.permute(0, 3, 1, 2)
        value = F.interpolate(value.float(), size=target.shape[1:3], mode="bilinear", align_corners=False)
        return value.permute(0, 2, 3, 1).to(dtype=target.dtype)
    if "rel_pos" in key and source.ndim == target.ndim == 2:
        value = source[None, None].float()
        value = F.interpolate(value, size=target.shape, mode="bilinear", align_corners=False)
        return value[0, 0].to(dtype=target.dtype)
    return None


class OfficialSAMMed2DAdapter(nn.Module):
    """Adapter around the official OpenGVLab SAM-Med2D ViT-B implementation.

    Two protocols are deliberately separated:
    * prompt-free downstream adaptation for the automatic-segmentation main table;
    * deterministic ground-truth-box prompting for an oracle/interactive table.

    Grayscale ultrasound is repeated to RGB. All metrics remain on the project's
    unified 512x512 grid.
    """

    def __init__(
        self,
        official_repository_path: str,
        base_checkpoint: str,
        model_resolution: int = 512,
        output_resolution: Tuple[int, int] = (512, 512),
        minimum_loaded_fraction: float = 0.98,
    ) -> None:
        super().__init__()
        self.official_repository_path = str(Path(official_repository_path).resolve())
        self.base_checkpoint = str(Path(base_checkpoint).resolve())
        self.model_resolution = int(model_resolution)
        self.output_resolution = tuple(int(v) for v in output_resolution)
        if self.model_resolution not in (256, 512, 1024):
            raise ValueError("model_resolution must be 256, 512, or 1024")
        if len(self.output_resolution) != 2:
            raise ValueError("output_resolution must be (height, width)")

        repo = Path(self.official_repository_path)
        if not (repo / "segment_anything" / "build_sam.py").is_file():
            raise FileNotFoundError(
                f"Official SAM-Med2D repository not found at {repo}. "
                "Run scripts/install_official_sammed2d.sh first."
            )
        checkpoint = Path(self.base_checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        segment_anything = importlib.import_module("segment_anything")
        args = SimpleNamespace(
            image_size=self.model_resolution,
            sam_checkpoint=None,
            encoder_adapter=True,
        )
        self.network = segment_anything.sam_model_registry["vit_b"](args)
        self.load_report = self._load_verified_base(checkpoint, minimum_loaded_fraction)
        self.configure_prompt_free_trainables()

    def _load_verified_base(self, checkpoint: Path, minimum_fraction: float) -> Dict[str, Any]:
        incoming = _extract_model_state(_safe_torch_load(checkpoint))
        target = self.network.state_dict()
        loadable: Dict[str, torch.Tensor] = {}
        interpolated = []
        incompatible = []
        unexpected = []

        for key, value in incoming.items():
            if key not in target:
                unexpected.append(key)
                continue
            adjusted = _resize_like(value, target[key], key)
            if adjusted is None:
                incompatible.append(
                    {"key": key, "checkpoint_shape": list(value.shape), "model_shape": list(target[key].shape)}
                )
                continue
            if adjusted.shape != target[key].shape:
                incompatible.append(
                    {"key": key, "checkpoint_shape": list(value.shape), "model_shape": list(target[key].shape)}
                )
                continue
            if value.shape != adjusted.shape:
                interpolated.append(key)
            loadable[key] = adjusted

        merged = dict(target)
        merged.update(loadable)
        self.network.load_state_dict(merged, strict=True)

        total_elements = sum(int(v.numel()) for v in target.values())
        loaded_elements = sum(int(target[k].numel()) for k in loadable)
        loaded_fraction = loaded_elements / max(total_elements, 1)
        if loaded_fraction < float(minimum_fraction):
            raise RuntimeError(
                f"Only {loaded_fraction:.4%} of model elements were loaded from {checkpoint}; "
                f"required >= {minimum_fraction:.2%}. Incompatible examples: {incompatible[:5]}"
            )
        return {
            "checkpoint": str(checkpoint),
            "loaded_keys": len(loadable),
            "target_keys": len(target),
            "loaded_fraction": loaded_fraction,
            "interpolated_keys": interpolated,
            "incompatible_keys": incompatible,
            "unexpected_key_count": len(unexpected),
            "unexpected_key_examples": unexpected[:20],
        }

    def configure_prompt_free_trainables(self) -> None:
        for parameter in self.network.parameters():
            parameter.requires_grad = False

        for name, parameter in self.network.image_encoder.named_parameters():
            if "adapter" in name.lower():
                parameter.requires_grad = True

        for name, parameter in self.network.prompt_encoder.named_parameters():
            if "no_mask_embed" in name:
                parameter.requires_grad = True

        for parameter in self.network.mask_decoder.parameters():
            parameter.requires_grad = True

    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        names = {name for name, p in self.network.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu() for k, v in self.network.state_dict().items() if k in names}

    def load_trainable_state_dict(self, delta: Mapping[str, torch.Tensor]) -> None:
        target = self.network.state_dict()
        unknown = [k for k in delta if k not in target]
        mismatch = [k for k, v in delta.items() if k in target and tuple(v.shape) != tuple(target[k].shape)]
        if unknown or mismatch:
            raise ValueError(f"Invalid trainable delta. unknown={unknown[:5]} mismatch={mismatch[:5]}")
        merged = dict(target)
        merged.update(delta)
        self.network.load_state_dict(merged, strict=True)

    def parameter_report(self) -> Dict[str, int]:
        return {
            "total_parameters": int(sum(p.numel() for p in self.network.parameters())),
            "trainable_parameters": int(sum(p.numel() for p in self.network.parameters() if p.requires_grad)),
            "image_encoder_adapter_parameters": int(
                sum(p.numel() for n, p in self.network.image_encoder.named_parameters() if p.requires_grad)
            ),
            "prompt_no_mask_parameters": int(
                sum(p.numel() for n, p in self.network.prompt_encoder.named_parameters() if p.requires_grad)
            ),
            "mask_decoder_parameters": int(
                sum(p.numel() for p in self.network.mask_decoder.parameters() if p.requires_grad)
            ),
        }

    def export_config(self) -> Dict[str, Any]:
        return {
            "official_repository_path": self.official_repository_path,
            "base_checkpoint": self.base_checkpoint,
            "model_resolution": self.model_resolution,
            "output_resolution": self.output_resolution,
        }

    def _encode_image(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError(f"Expected Bx1xHxW grayscale input, got {tuple(images.shape)}")
        if images.shape[0] != 1:
            raise ValueError("_encode_image expects one image; use forward_prompt_free for batches")
        x = images.float().clamp(0.0, 1.0)
        x = F.interpolate(
            x,
            size=(self.model_resolution, self.model_resolution),
            mode="bilinear",
            align_corners=False,
        )
        x = x.repeat(1, 3, 1, 1) * 255.0
        x = self.network.preprocess(x[0]).unsqueeze(0)
        return self.network.image_encoder(x)

    def _decode(
        self,
        image_embeddings: torch.Tensor,
        boxes: torch.Tensor | None,
        multimask_output: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sparse, dense = self.network.prompt_encoder(
            points=None,
            boxes=boxes,
            masks=None,
        )
        low_res, iou = self.network.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.network.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=multimask_output,
        )
        return low_res, iou

    def _to_output_grid(self, logits: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            logits,
            size=self.output_resolution,
            mode="bilinear",
            align_corners=False,
        )

    def forward_prompt_free(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError(f"Expected Bx1xHxW grayscale input, got {tuple(images.shape)}")
        outputs = []
        for index in range(images.shape[0]):
            embedding = self._encode_image(images[index:index + 1])
            low_res, _ = self._decode(
                embedding,
                boxes=None,
                multimask_output=False,
            )
            positive_logit = self._to_output_grid(low_res)[:, 0]
            outputs.append(
                torch.stack(
                    [torch.zeros_like(positive_logit), positive_logit],
                    dim=1,
                )
            )
        return torch.cat(outputs, dim=0)

    @staticmethod
    def tight_box_from_mask(mask: torch.Tensor) -> torch.Tensor:
        if mask.ndim == 3:
            mask = mask[0]
        coordinates = torch.nonzero(mask > 0, as_tuple=False)
        if coordinates.numel() == 0:
            raise ValueError("GT-box oracle cannot be evaluated on an empty target mask")
        y0 = coordinates[:, 0].min().item()
        y1 = coordinates[:, 0].max().item()
        x0 = coordinates[:, 1].min().item()
        x1 = coordinates[:, 1].max().item()
        return torch.tensor([x0, y0, x1, y1], device=mask.device, dtype=torch.float32)

    @torch.inference_mode()
    def forward_gt_box_oracle(self, images: torch.Tensor, target_masks: torch.Tensor):
        embedding = self._encode_image(images)
        box_512 = self.tight_box_from_mask(target_masks)
        scale_x = self.model_resolution / float(self.output_resolution[1])
        scale_y = self.model_resolution / float(self.output_resolution[0])
        box_model = box_512.clone()
        box_model[[0, 2]] *= scale_x
        box_model[[1, 3]] *= scale_y
        low_res, predicted_iou = self._decode(
            embedding,
            boxes=box_model.unsqueeze(0),
            multimask_output=True,
        )
        best_index = int(predicted_iou[0].argmax().item())
        positive_logit = self._to_output_grid(low_res[:, best_index:best_index + 1])[:, 0]
        logits = torch.stack([torch.zeros_like(positive_logit), positive_logit], dim=1)
        return logits, box_512.detach().cpu()

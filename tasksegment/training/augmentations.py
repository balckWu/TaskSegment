from __future__ import annotations

import random
from typing import Sequence, Tuple

import torch
import torch.nn.functional as F


def _add_speckle_noise(img_tensor: torch.Tensor, noise_level: float = 0.1) -> torch.Tensor:
    """乘性 speckle noise，适合模拟超声斑点噪声。"""
    noise = torch.randn_like(img_tensor) * noise_level
    return img_tensor + noise * img_tensor


def _random_hflip(
    imgs: torch.Tensor,
    masks: torch.Tensor,
    prob: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if random.random() < prob:
        imgs = torch.flip(imgs, dims=[-1])
        masks = torch.flip(masks, dims=[-1])
    return imgs, masks


def _random_vflip(
    imgs: torch.Tensor,
    masks: torch.Tensor,
    prob: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if random.random() < prob:
        imgs = torch.flip(imgs, dims=[-2])
        masks = torch.flip(masks, dims=[-2])
    return imgs, masks


def _random_rot90(
    imgs: torch.Tensor,
    masks: torch.Tensor,
    prob: float,
    choices: Sequence[int] = (1, 3),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """随机 90/270 度旋转。

    这里默认不用 k=2，即 180 度旋转。
    医学超声图像通常有固定探头方向和解剖方向，180 度旋转可能过强。
    """
    if random.random() < prob:
        k = random.choice(tuple(choices))
        imgs = torch.rot90(imgs, k, dims=[-2, -1])
        masks = torch.rot90(masks, k, dims=[-2, -1])
    return imgs, masks


def _apply_geometric_aug(
    imgs: torch.Tensor,
    masks: torch.Tensor,
    *,
    hflip_prob: float,
    vflip_prob: float,
    rot90_prob: float,
    rot90_choices: Sequence[int] = (1, 3),
) -> Tuple[torch.Tensor, torch.Tensor]:
    imgs, masks = _random_hflip(imgs, masks, hflip_prob)
    imgs, masks = _random_vflip(imgs, masks, vflip_prob)
    imgs, masks = _random_rot90(imgs, masks, rot90_prob, rot90_choices)
    return imgs, masks


def _apply_brightness_contrast(
    imgs: torch.Tensor,
    prob: float,
    scale_range: Tuple[float, float],
    shift_range: Tuple[float, float],
) -> torch.Tensor:
    """线性亮度/对比度扰动。

    imgs 默认已经在 [-1, 1] 范围内。
    """
    if random.random() < prob:
        scale = torch.empty(
            imgs.shape[0], 1, 1, 1,
            device=imgs.device,
            dtype=imgs.dtype,
        ).uniform_(*scale_range)

        shift = torch.empty(
            imgs.shape[0], 1, 1, 1,
            device=imgs.device,
            dtype=imgs.dtype,
        ).uniform_(*shift_range)

        imgs = imgs * scale + shift

    return imgs


def _apply_gamma(
    imgs: torch.Tensor,
    prob: float,
    gamma_range: Tuple[float, float],
) -> torch.Tensor:
    """Gamma 灰度扰动。

    先把 [-1, 1] 映射到 [0, 1]，做 gamma，再映射回 [-1, 1]。
    """
    if random.random() < prob:
        gamma = torch.empty(
            imgs.shape[0], 1, 1, 1,
            device=imgs.device,
            dtype=imgs.dtype,
        ).uniform_(*gamma_range)

        imgs01 = ((imgs + 1.0) * 0.5).clamp(0.0, 1.0)
        imgs01 = torch.pow(imgs01, gamma)
        imgs = imgs01 * 2.0 - 1.0

    return imgs


def _apply_blur(
    imgs: torch.Tensor,
    prob: float,
) -> torch.Tensor:
    """轻微均值模糊，模拟超声边界模糊或成像不清。"""
    if random.random() < prob:
        imgs = F.avg_pool2d(imgs, kernel_size=3, stride=1, padding=1)
    return imgs


def _apply_gaussian_noise(
    imgs: torch.Tensor,
    prob: float,
    std: float,
) -> torch.Tensor:
    """加性高斯噪声。"""
    if random.random() < prob:
        imgs = imgs + std * torch.randn_like(imgs)
    return imgs


def _apply_thyroid_photometric_aug(imgs: torch.Tensor) -> torch.Tensor:
    """thyroid / TN3K 的灰度增强。

    这部分基本保留原设置，只稍微控制整体强度。
    """
    if random.random() < 0.18:
        imgs = _add_speckle_noise(imgs, noise_level=0.03)

    imgs = _apply_brightness_contrast(
        imgs,
        prob=0.38,
        scale_range=(0.96, 1.04),
        shift_range=(-0.04, 0.04),
    )

    imgs = _apply_gamma(
        imgs,
        prob=0.15,
        gamma_range=(0.95, 1.05),
    )

    imgs = _apply_blur(
        imgs,
        prob=0.10,
    )

    imgs = _apply_gaussian_noise(
        imgs,
        prob=0.15,
        std=0.01,
    )

    return imgs


def _apply_busi_photometric_aug(imgs: torch.Tensor) -> torch.Tensor:
    """BUSI / BUS-BRA 的灰度增强，保守版。

    原设置相对偏强：
        speckle: 0.45, level=0.08
        brightness: 0.60
        gamma: 0.25, range=(0.90, 1.25)
        blur: 0.20
        gaussian: 0.40, std=0.02

    这里降弱为：
        speckle: 0.35, level=0.06
        brightness: 0.50
        gamma: 0.18, range=(0.95, 1.15)
        blur: 0.12
        gaussian: 0.25, std=0.015
    """
    if random.random() < 0.35:
        imgs = _add_speckle_noise(imgs, noise_level=0.06)

    imgs = _apply_brightness_contrast(
        imgs,
        prob=0.50,
        scale_range=(0.95, 1.06),
        shift_range=(-0.06, 0.06),
    )

    imgs = _apply_gamma(
        imgs,
        prob=0.18,
        gamma_range=(0.95, 1.15),
    )

    imgs = _apply_blur(
        imgs,
        prob=0.12,
    )

    imgs = _apply_gaussian_noise(
        imgs,
        prob=0.25,
        std=0.015,
    )

    return imgs



def augment_episode_medical(
    imgs: torch.Tensor,
    masks: torch.Tensor,
    modality: str = "generic",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GPU-safe episode augmentation.

    Args:
        imgs: image tensor [B, C, H, W], usually normalized to [-1, 1].
        masks: binary mask tensor [B, 1, H, W].

    Returns:
        Augmented images and masks. CPU CLAHE is handled in the dataset, not here.
    """

    modality = str(modality)

    if modality == "thyroid":
        # thyroid / TN3K: keep geometry conservative; rot90 can break ultrasound orientation priors.
        imgs, masks = _apply_geometric_aug(
            imgs,
            masks,
            hflip_prob=0.45,
            vflip_prob=0.03,
            rot90_prob=0.0,
            rot90_choices=(1, 3),
        )
        imgs = _apply_thyroid_photometric_aug(imgs)

    elif modality == "BUSI":
        # BUSI / BUS-BRA use conservative geometry and photometric perturbations.
        imgs, masks = _apply_geometric_aug(
            imgs,
            masks,
            hflip_prob=0.50,
            vflip_prob=0.10,
            rot90_prob=0.15,
            rot90_choices=(1, 3),
        )
        imgs = _apply_busi_photometric_aug(imgs)

    else:
        # Generic fallback: light geometric augmentation only.
        imgs, masks = _apply_geometric_aug(
            imgs,
            masks,
            hflip_prob=0.50,
            vflip_prob=0.10,
            rot90_prob=0.15,
            rot90_choices=(1, 3),
        )

    imgs = imgs.clamp(-1.0, 1.0)
    return imgs, masks
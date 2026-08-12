import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")

    # 转换为二值掩膜
    mask = (mask > 0).astype(np.uint8) * 255
    return mask


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)

    min_value = float(image.min())
    max_value = float(image.max())

    if max_value - min_value < 1e-8:
        return np.zeros_like(image, dtype=np.float32)

    return (image - min_value) / (max_value - min_value)


def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(
        image,
        (size, size),
        interpolation=cv2.INTER_LINEAR,
    )


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    # Mask 必须使用 nearest，避免边界产生新的灰度值
    resized = cv2.resize(
        mask,
        (size, size),
        interpolation=cv2.INTER_NEAREST,
    )
    return (resized > 0).astype(np.uint8) * 255


def save_image(array: np.ndarray, path: Path, is_mask: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(3, 3))

    if is_mask or array.ndim == 2:
        plt.imshow(array, cmap="gray", vmin=0, vmax=255 if is_mask else None)
    else:
        plt.imshow(array)

    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_grid(
    image_states: list[np.ndarray],
    mask_states: list[np.ndarray],
    sizes: list[int],
    output_path: Path,
) -> None:
    count = len(sizes)

    fig, axes = plt.subplots(
        2,
        count,
        figsize=(2.5 * count, 5.0),
    )

    for index, size in enumerate(sizes):
        image = image_states[index]
        mask = mask_states[index]

        if image.ndim == 2:
            axes[0, index].imshow(image, cmap="gray")
        else:
            axes[0, index].imshow(image)

        axes[0, index].set_title(
            rf"$I_s^{{{index + 1}}}$ ({size}$\times${size})",
            fontsize=11,
        )
        axes[0, index].axis("off")

        axes[1, index].imshow(mask, cmap="gray", vmin=0, vmax=255)
        axes[1, index].set_title(
            rf"$M_s^{{{index + 1}}}$ ({size}$\times${size})",
            fontsize=11,
        )
        axes[1, index].axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export real support-image and support-mask multi-scale states."
    )

    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to one real ultrasound image.",
    )
    parser.add_argument(
        "--mask",
        type=Path,
        required=True,
        help="Path to its segmentation mask.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/tank/wk/TSV6/paper_artifacts/vis_states/image_mask_pyramid"
        ),
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[512, 256, 128, 64, 32],
        help="Five output resolutions.",
    )

    args = parser.parse_args()

    if len(args.sizes) != 5:
        raise ValueError(
            f"Exactly five sizes are expected, but got {len(args.sizes)}."
        )

    image = read_image(args.image)
    mask = read_mask(args.mask)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    image_states = []
    mask_states = []

    for index, size in enumerate(args.sizes, start=1):
        resized_image = resize_image(image, size)
        resized_mask = resize_mask(mask, size)

        image_states.append(resized_image)
        mask_states.append(resized_mask)

        save_image(
            resized_image,
            output_dir / f"support_image_state_{index}_{size}.png",
            is_mask=False,
        )

        save_image(
            resized_mask,
            output_dir / f"support_mask_state_{index}_{size}.png",
            is_mask=True,
        )

        # 同时保存原始数组，后续做 PPT 时不会受到 matplotlib 裁切影响
        if resized_image.ndim == 2:
            cv2.imwrite(
                str(output_dir / f"support_image_raw_{index}_{size}.png"),
                resized_image,
            )
        else:
            cv2.imwrite(
                str(output_dir / f"support_image_raw_{index}_{size}.png"),
                cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR),
            )

        cv2.imwrite(
            str(output_dir / f"support_mask_raw_{index}_{size}.png"),
            resized_mask,
        )

    save_grid(
        image_states=image_states,
        mask_states=mask_states,
        sizes=args.sizes,
        output_path=output_dir / "image_mask_pyramid_grid.png",
    )

    print("=" * 80)
    print("Export completed")
    print(f"Image:  {args.image}")
    print(f"Mask:   {args.mask}")
    print(f"Output: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

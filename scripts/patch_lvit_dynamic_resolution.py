#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count == 0:
        if new in text:
            return text
        raise RuntimeError(f"Could not locate expected official source fragment: {label}")
    if count != 1:
        raise RuntimeError(f"Expected one occurrence for {label}, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply a minimal, auditable dynamic-resolution patch to official LViT."
    )
    parser.add_argument("--repo", default="third_party/LViT")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    lvit_path = repo / "nets" / "LViT.py"
    vit_path = repo / "nets" / "Vit.py"
    if not lvit_path.is_file() or not vit_path.is_file():
        raise FileNotFoundError(f"Official LViT source not found under {repo}")

    before = {
        "nets/LViT.py": sha256(lvit_path),
        "nets/Vit.py": sha256(vit_path),
    }

    lvit = lvit_path.read_text(encoding="utf-8")
    replacements = [
        (
            "self.downVit = VisionTransformer(config, vis, img_size=224, channel_num=64, patch_size=16, embed_dim=64)",
            "self.downVit = VisionTransformer(config, vis, img_size=img_size, channel_num=64, patch_size=16, embed_dim=64)",
            "downVit resolution",
        ),
        (
            "self.downVit1 = VisionTransformer(config, vis, img_size=112, channel_num=128, patch_size=8, embed_dim=128)",
            "self.downVit1 = VisionTransformer(config, vis, img_size=img_size // 2, channel_num=128, patch_size=8, embed_dim=128)",
            "downVit1 resolution",
        ),
        (
            "self.downVit2 = VisionTransformer(config, vis, img_size=56, channel_num=256, patch_size=4, embed_dim=256)",
            "self.downVit2 = VisionTransformer(config, vis, img_size=img_size // 4, channel_num=256, patch_size=4, embed_dim=256)",
            "downVit2 resolution",
        ),
        (
            "self.downVit3 = VisionTransformer(config, vis, img_size=28, channel_num=512, patch_size=2, embed_dim=512)",
            "self.downVit3 = VisionTransformer(config, vis, img_size=img_size // 8, channel_num=512, patch_size=2, embed_dim=512)",
            "downVit3 resolution",
        ),
        (
            "self.upVit = VisionTransformer(config, vis, img_size=224, channel_num=64, patch_size=16, embed_dim=64)",
            "self.upVit = VisionTransformer(config, vis, img_size=img_size, channel_num=64, patch_size=16, embed_dim=64)",
            "upVit resolution",
        ),
        (
            "self.upVit1 = VisionTransformer(config, vis, img_size=112, channel_num=128, patch_size=8, embed_dim=128)",
            "self.upVit1 = VisionTransformer(config, vis, img_size=img_size // 2, channel_num=128, patch_size=8, embed_dim=128)",
            "upVit1 resolution",
        ),
        (
            "self.upVit2 = VisionTransformer(config, vis, img_size=56, channel_num=256, patch_size=4, embed_dim=256)",
            "self.upVit2 = VisionTransformer(config, vis, img_size=img_size // 4, channel_num=256, patch_size=4, embed_dim=256)",
            "upVit2 resolution",
        ),
        (
            "self.upVit3 = VisionTransformer(config, vis, img_size=28, channel_num=512, patch_size=2, embed_dim=512)",
            "self.upVit3 = VisionTransformer(config, vis, img_size=img_size // 8, channel_num=512, patch_size=2, embed_dim=512)",
            "upVit3 resolution",
        ),
    ]
    for old, new, label in replacements:
        lvit = replace_once(lvit, old, new, label)
    lvit_path.write_text(lvit, encoding="utf-8")

    vit = vit_path.read_text(encoding="utf-8")
    old = "self.CTBN3 = ConvTransBN(in_channels=10, out_channels=196)"
    new = (
        "text_patch_count = (int(img_size) // int(patch_size)) ** 2\n"
        "        self.CTBN3 = ConvTransBN(in_channels=10, out_channels=text_patch_count)"
    )
    vit = replace_once(vit, old, new, "text-to-patch projection")
    vit_path.write_text(vit, encoding="utf-8")

    after = {
        "nets/LViT.py": sha256(lvit_path),
        "nets/Vit.py": sha256(vit_path),
    }
    marker = repo / "TASKSEGMENT_DYNAMIC_RESOLUTION_PATCH.txt"
    marker.write_text(
        "\n".join(
            [
                "Minimal TaskSegment LViT dynamic-resolution patch",
                "Purpose: allow the official architecture to use 512x512 while retaining 224x224 compatibility.",
                "Changes:",
                "- replace eight hard-coded feature-map sizes with img_size-derived values",
                "- replace the hard-coded 196 text-to-patch channels with (img_size/patch_size)^2",
                "",
                f"before nets/LViT.py {before['nets/LViT.py']}",
                f"after  nets/LViT.py {after['nets/LViT.py']}",
                f"before nets/Vit.py {before['nets/Vit.py']}",
                f"after  nets/Vit.py {after['nets/Vit.py']}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Patched official LViT source: {repo}")
    print(marker.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

REQUIRED_MEMBERS = {
    "embedding/kernel.npy",
    "embedding/bias.npy",
    "Transformer/encoder_norm/scale.npy",
    "Transformer/posembed_input/pos_embedding.npy",
    "conv_root/kernel.npy",
    "gn_root/scale.npy",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        default="pretrained/transunet/R50+ViT-B_16.npz",
    )
    args = parser.parse_args()
    path = Path(args.path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as archive:
        members = set(archive.namelist())
    missing = sorted(REQUIRED_MEMBERS - members)
    if missing:
        raise RuntimeError(f"Incomplete/incorrect TransUNet NPZ; missing: {missing}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    print(f"path={path}")
    print(f"size_bytes={path.stat().st_size}")
    print(f"sha256={digest.hexdigest()}")
    print("required_keys=OK")


if __name__ == "__main__":
    main()

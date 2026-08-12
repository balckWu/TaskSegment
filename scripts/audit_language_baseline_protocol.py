#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED = {
    "thyroid": "a thyroid nodule in an ultrasound image",
    "TN3K": "a thyroid nodule in an ultrasound image",
    "BUSI_WHU": "a breast lesion in an ultrasound image",
    "BUS-BRA": "a breast mass in an ultrasound image",
    "OTU": "an ovarian tumor in an ultrasound image",
    "prostate": "the prostate gland in an ultrasound image",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clipseg-prompts", default="clipseg_prompts.json")
    parser.add_argument("--lvit-prompts", default="lvit_prompts.json")
    args = parser.parse_args()

    clip_path = Path(args.clipseg_prompts)
    lvit_path = Path(args.lvit_prompts)
    for path in (clip_path, lvit_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    clip = load(clip_path)
    lvit = load(lvit_path)
    if clip != EXPECTED:
        raise SystemExit(f"CLIPSeg prompts differ from the preregistered protocol: {clip}")
    if lvit != EXPECTED:
        raise SystemExit(f"LViT prompts differ from the preregistered protocol: {lvit}")
    if clip != lvit:
        raise SystemExit("CLIPSeg and LViT prompt dictionaries do not match")

    print("Language-baseline prompt audit passed")
    print("clipseg_prompt_sha256:", digest(clip_path))
    print("lvit_prompt_sha256:", digest(lvit_path))
    print("domains:", ", ".join(EXPECTED))
    print("The two methods receive identical domain-level textual information.")


if __name__ == "__main__":
    main()

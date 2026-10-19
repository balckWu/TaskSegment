"""Read-only source, recipe and optional dataset layout verification."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from iris2d.protocol import default_data_root, load_recipe, set_cosine_lr, verify_reference

EXPECTED = {"thyroid": (855, 285, 286), "TN3K": (2447, 432, 614),
            "BUSI_WHU": (555, 186, 186), "BUS-BRA": (1312, 280, 283),
            "OTU": (656, 164, 382), "prostate": (67, 14, 16)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(default_data_root()))
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()
    reference = verify_reference()
    recipe, common = load_recipe()
    print("PASS: local/server reference utilities and baseline LR functions agree.")
    for relative in reference.get("formatting_only", []):
        print(f"PASS equivalent code (comments/formatting differ): {relative}")
    print(f"AdamW lr={recipe['lr']:g} min_lr={recipe['min_lr']:g} weight_decay={recipe['weight_decay']:g}")
    print(f"warmup={common['warmup_epochs']} epochs={common['epochs']} schedule={recipe['lr_schedule']}")
    class Optimizer:
        param_groups = [{}]
    print("LR epochs 1/2/5/6/7/150:", [set_cosine_lr(Optimizer(), e-1, common['epochs'], recipe['lr'], recipe['min_lr'], common['warmup_epochs']) for e in (1, 2, 5, 6, 7, 150)])
    if args.check_data:
        root = Path(args.data_root)
        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        for domain, counts in EXPECTED.items():
            for split, expected in zip(("train", "val", "test"), counts):
                folder = root / domain / split
                images, masks = folder / "images", folder / "masks"
                if not images.is_dir() or not masks.is_dir():
                    raise ValueError(f"Missing data folder: {folder}")
                names = [p for p in images.iterdir() if p.is_file() and not p.name.startswith('.') and p.suffix.lower() in extensions]
                valid = [p for p in names if any((masks / name).is_file() for name in (p.stem+'.png', p.stem+'.PNG', p.name))]
                if len(valid) != expected or len(valid) != len(names):
                    raise ValueError(f"{domain}/{split}: image/mask pairs {len(valid)}, images {len(names)}, expected {expected}")
            print(f"PASS data {domain}: {counts}")
        print("PASS: all 9020 image/mask pairs match the paper split counts.")
    print("No training or test evaluation was started.")


if __name__ == "__main__":
    main()

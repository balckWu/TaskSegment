#!/usr/bin/env python3
"""Run raw nnU-Net v2 inference on the held-out TaskSegment test set."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Predict TaskSegment test cases with nnU-Net.")
    p.add_argument("--work-dir", default="./comparisons/nnunet")
    p.add_argument("--dataset-id", type=int, default=501)
    p.add_argument("--dataset-name", default="TaskSegment6Domain")
    p.add_argument("--configuration", default="2d")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--trainer", default="nnUNetTrainer")
    p.add_argument("--plans", default="nnUNetPlans")
    p.add_argument("--gpu", default="5")
    p.add_argument("--disable-tta", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    work = Path(args.work_dir).resolve()
    dataset = f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    raw = work / "nnUNet_raw"
    pre = work / "nnUNet_preprocessed"
    results = work / "nnUNet_results"
    input_dir = raw / dataset / "imagesTs"
    output_dir = work / "predictions" / dataset / f"{args.configuration}_fold{args.fold}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if shutil.which("nnUNetv2_predict") is None:
        raise RuntimeError("nnUNetv2_predict not found. Install nnunetv2 first.")

    env = os.environ.copy()
    env["nnUNet_raw"] = str(raw)
    env["nnUNet_preprocessed"] = str(pre)
    env["nnUNet_results"] = str(results)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    command = [
        "nnUNetv2_predict",
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-d", str(args.dataset_id),
        "-c", args.configuration,
        "-f", str(args.fold),
        "-tr", args.trainer,
        "-p", args.plans,
        "-device", "cuda",
    ]
    if args.disable_tta:
        command.append("--disable_tta")

    print("[command]", " ".join(command), flush=True)
    subprocess.run(command, env=env, check=True)
    print(f"predictions={output_dir}")
    print(f"ground_truth={raw / dataset / 'labelsTs'}")


if __name__ == "__main__":
    main()

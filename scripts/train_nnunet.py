#!/usr/bin/env python3
"""Plan, preprocess, and train the nnU-Net v2 2D baseline.

This wrapper:
1. sets isolated nnU-Net paths under comparisons/nnunet;
2. runs official nnU-Net v2 planning/preprocessing;
3. installs the original TaskSegment train/validation partition as fold 0;
4. trains exactly fold 0 with the official nnUNetTrainer;
5. writes a timestamped command log.

It deliberately does not run five-fold ensembling or automatic postprocessing,
because the other architecture baselines use one fixed validation split and raw
predictions.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train nnU-Net v2 on TaskSegment.")
    parser.add_argument("--work-dir", default="./comparisons/nnunet")
    parser.add_argument("--dataset-id", type=int, default=501)
    parser.add_argument("--dataset-name", default="TaskSegment6Domain")
    parser.add_argument("--configuration", default="2d")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--trainer", default="nnUNetTrainer")
    parser.add_argument("--plans", default="nnUNetPlans")
    parser.add_argument("--gpu", default="5", help="Physical CUDA device id.")
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--skip-planning", action="store_true")
    parser.add_argument("--continue-training", action="store_true")
    parser.add_argument("--export-npz", action="store_true")
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu", "mps"),
        default="cuda",
    )
    return parser.parse_args()


def run(command: List[str], env: Dict[str, str], log_handle) -> None:
    text = " ".join(command)
    print(f"\n[command] {text}", flush=True)
    log_handle.write(f"\n[command] {text}\n")
    log_handle.flush()
    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        log_handle.write(line)
    return_code = process.wait()
    log_handle.flush()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir).resolve()
    raw_root = work_dir / "nnUNet_raw"
    preprocessed_root = work_dir / "nnUNet_preprocessed"
    results_root = work_dir / "nnUNet_results"
    dataset_name = f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    raw_dataset = raw_root / dataset_name

    required = [
        raw_dataset / "dataset.json",
        raw_dataset / "splits_final.json",
        raw_dataset / "imagesTr",
        raw_dataset / "labelsTr",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Run scripts/prepare_nnunet.py first. Missing:\n  " + "\n  ".join(missing)
        )

    for folder in (raw_root, preprocessed_root, results_root, work_dir / "logs"):
        folder.mkdir(parents=True, exist_ok=True)

    for executable in ("nnUNetv2_plan_and_preprocess", "nnUNetv2_train"):
        if shutil.which(executable) is None:
            raise RuntimeError(
                f"{executable} is not available. Install nnU-Net v2 first:\n"
                "  pip install nnunetv2"
            )

    env = os.environ.copy()
    env["nnUNet_raw"] = str(raw_root)
    env["nnUNet_preprocessed"] = str(preprocessed_root)
    env["nnUNet_results"] = str(results_root)
    if args.device == "cuda":
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = work_dir / "logs" / f"train_{dataset_name}_{args.configuration}_fold{args.fold}_{timestamp}.log"

    run_config = {
        "dataset_id": args.dataset_id,
        "dataset_name": dataset_name,
        "configuration": args.configuration,
        "fold": args.fold,
        "trainer": args.trainer,
        "plans": args.plans,
        "gpu": args.gpu,
        "device": args.device,
        "official_five_fold_ensemble": False,
        "automatic_postprocessing": False,
        "split_source": str(raw_dataset / "splits_final.json"),
        "paths": {
            "nnUNet_raw": str(raw_root),
            "nnUNet_preprocessed": str(preprocessed_root),
            "nnUNet_results": str(results_root),
        },
    }
    (work_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
    )

    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(json.dumps(run_config, indent=2) + "\n")

        if not args.skip_planning:
            plan_command = [
                "nnUNetv2_plan_and_preprocess",
                "-d",
                str(args.dataset_id),
                "-c",
                args.configuration,
                "--verify_dataset_integrity",
                "-np",
                str(args.num_processes),
            ]
            run(plan_command, env, log_handle)

        preprocessed_dataset = preprocessed_root / dataset_name
        if not preprocessed_dataset.exists():
            raise FileNotFoundError(
                f"Preprocessed dataset not found: {preprocessed_dataset}"
            )

        # nnU-Net reads this file in nnUNetTrainer.do_split().
        split_target = preprocessed_dataset / "splits_final.json"
        shutil.copy2(raw_dataset / "splits_final.json", split_target)
        print(f"[split] installed custom fold at {split_target}", flush=True)
        log_handle.write(f"[split] installed custom fold at {split_target}\n")

        train_command = [
            "nnUNetv2_train",
            str(args.dataset_id),
            args.configuration,
            str(args.fold),
            "-tr",
            args.trainer,
            "-p",
            args.plans,
            "-device",
            args.device,
        ]
        if args.continue_training:
            train_command.append("--c")
        if args.export_npz:
            train_command.append("--npz")
        run(train_command, env, log_handle)

    print("\nTraining command completed.")
    print(f"log={log_path}")
    print(f"results={results_root}")


if __name__ == "__main__":
    main()

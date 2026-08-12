#!/usr/bin/env bash
set -euo pipefail

cd /tank/wk/TSV3

source /root/miniconda3/etc/profile.d/conda.sh
conda activate TS

# 物理 GPU 4 映射为脚本内部的 cuda:0
export CUDA_VISIBLE_DEVICES=4

SEEDS=(42 3407 2026)

echo "===== U-Net GPU4 queue ====="
echo "started_at=$(date -Is)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

for SEED in "${SEEDS[@]}"; do
    RUN_DIR="/tank/wk/TSV3/paper_artifacts/runs/unet/seed_${SEED}"
    CHECKPOINT="${RUN_DIR}/checkpoint.pth"
    PRED_DIR="${RUN_DIR}/predictions"

    mkdir -p "${RUN_DIR}"

    echo
    echo "========================================"
    echo "U-Net seed=${SEED}"
    echo "started_at=$(date -Is)"
    echo "run_dir=${RUN_DIR}"
    echo "========================================"

    # 防止同一种子已经由其他进程启动
    if pgrep -af "[t]rain_unet.py.*--seed ${SEED}" >/dev/null; then
        echo "ERROR: U-Net seed ${SEED} is already running"
        exit 20
    fi

    python -u scripts/train_unet.py \
      --save-path "${CHECKPOINT}" \
      --data-root /tank/wk/TSV3/data \
      --device cuda:0 \
      --seed "${SEED}" \
      --epochs 150 \
      --batch-size 4 \
      --episodes-per-epoch 1500 \
      --grad-accum-steps 4 \
      --min-lr 1e-6 \
      --weight-decay 1e-4 \
      --warmup-epochs 5 \
      --patience 20 \
      --domain-sampling-alpha 0.0 \
      --threshold 0.5 \
      --cache-dataset \
      --preload-val-cache \
      --augmentation \
      --amp \
      --deterministic \
      --val-batch-size 4 \
      --lr 1e-4 \
      --max-grad-norm 5.0

    test -f "${CHECKPOINT}" || {
        echo "ERROR: checkpoint missing for seed ${SEED}"
        exit 21
    }

    python -u scripts/predict_baseline.py \
      --method unet \
      --checkpoint "${CHECKPOINT}" \
      --data-root /tank/wk/TSV3/data \
      --output-dir "${PRED_DIR}" \
      --device cuda:0 \
      --seed "${SEED}" \
      --batch-size 4 \
      --threshold 0.5 \
      --amp \
      --cache-dataset \
      --preload-cache

    echo "completed seed=${SEED} at $(date -Is)"
done

echo
echo "===== ALL U-NET RUNS COMPLETE ====="
echo "completed_at=$(date -Is)"

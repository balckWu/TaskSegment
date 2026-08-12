#!/usr/bin/env bash
set -euo pipefail

cd /tank/wk/TSV3

export CUDA_VISIBLE_DEVICES=1
PYTHON_BIN="/root/miniconda3/envs/TS/bin/python"

SEED=3407
RUN_DIR="/tank/wk/TSV3/paper_artifacts/runs/unetpp/seed_3407"
CHECKPOINT="${RUN_DIR}/checkpoint.pth"
PRED_DIR="${RUN_DIR}/predictions"

mkdir -p "${RUN_DIR}"

echo "=================================================="
echo "U-Net++ seed=${SEED}"
echo "physical_gpu=1"
echo "logical_device=cuda:0"
echo "python=${PYTHON_BIN}"
echo "started_at=$(date -Is)"
echo "=================================================="

if pgrep -af '[t]rain_unetpp.py.*--seed 3407' >/dev/null; then
    echo "ERROR: U-Net++ seed 3407 已经在运行"
    exit 20
fi

"${PYTHON_BIN}" -u scripts/train_unetpp.py   --save-path "${CHECKPOINT}"   --data-root /tank/wk/TSV3/data   --device cuda:0   --seed 3407   --epochs 150   --batch-size 4   --episodes-per-epoch 1500   --grad-accum-steps 4   --min-lr 1e-6   --weight-decay 1e-4   --warmup-epochs 5   --patience 20   --domain-sampling-alpha 0.0   --threshold 0.5   --cache-dataset   --preload-val-cache   --augmentation   --amp   --deterministic   --val-batch-size 4   --lr 1e-4   --max-grad-norm 5.0   --deep-supervision

test -s "${CHECKPOINT}" || {
    echo "ERROR: checkpoint 未生成"
    exit 21
}

echo "Training completed at $(date -Is)"
echo "Starting prediction..."

"${PYTHON_BIN}" -u scripts/predict_baseline.py   --method unetpp   --checkpoint "${CHECKPOINT}"   --data-root /tank/wk/TSV3/data   --output-dir "${PRED_DIR}"   --device cuda:0   --seed 3407   --batch-size 4   --threshold 0.5   --amp   --cache-dataset   --preload-cache

echo "=================================================="
echo "U-Net++ seed 3407 completed"
echo "completed_at=$(date -Is)"
echo "=================================================="

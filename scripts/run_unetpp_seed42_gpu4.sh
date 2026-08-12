#!/usr/bin/env bash
set -euo pipefail

cd /tank/wk/TSV3

PYTHON_BIN="/root/miniconda3/envs/TS/bin/python"
export CUDA_VISIBLE_DEVICES=4

RUN_DIR="/tank/wk/TSV3/paper_artifacts/runs/unetpp/seed_42"
CHECKPOINT="${RUN_DIR}/checkpoint.pth"
PRED_DIR="${RUN_DIR}/predictions"

mkdir -p "${RUN_DIR}"

echo "=================================================="
echo "U-Net++ seed 42"
echo "physical_gpu=4"
echo "logical_device=cuda:0"
echo "started_at=$(date -Is)"
echo "python=${PYTHON_BIN}"
echo "=================================================="

if pgrep -af '[t]rain_unetpp.py.*--seed 42' >/dev/null; then
    echo "ERROR: U-Net++ seed 42 已有训练进程"
    exit 20
fi

"${PYTHON_BIN}" -u scripts/train_unetpp.py   --save-path "${CHECKPOINT}"   --data-root /tank/wk/TSV3/data   --device cuda:0   --seed 42   --epochs 150   --batch-size 4   --episodes-per-epoch 1500   --grad-accum-steps 4   --min-lr 1e-6   --weight-decay 1e-4   --warmup-epochs 5   --patience 20   --domain-sampling-alpha 0.0   --threshold 0.5   --cache-dataset   --preload-val-cache   --augmentation   --amp   --deterministic   --val-batch-size 4   --lr 1e-4   --max-grad-norm 5.0   --deep-supervision

test -s "${CHECKPOINT}" || {
    echo "ERROR: checkpoint 没有生成"
    exit 21
}

echo "Training completed at $(date -Is)"
echo "Starting prediction..."

"${PYTHON_BIN}" -u scripts/predict_baseline.py   --method unetpp   --checkpoint "${CHECKPOINT}"   --data-root /tank/wk/TSV3/data   --output-dir "${PRED_DIR}"   --device cuda:0   --seed 42   --batch-size 4   --threshold 0.5   --amp   --cache-dataset   --preload-cache

echo "=================================================="
echo "U-Net++ seed 42 completed"
echo "completed_at=$(date -Is)"
echo "=================================================="

#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$ROOT/third_party/TransUNet"
mkdir -p "$ROOT/third_party"
if [[ ! -f "$TARGET/networks/vit_seg_modeling.py" ]]; then
  git clone https://github.com/Beckschen/TransUNet.git "$TARGET"
fi
if [[ -d "$TARGET/.git" ]]; then
  git -C "$TARGET" rev-parse HEAD | tee "$TARGET/OFFICIAL_COMMIT_USED.txt"
elif [[ -s "$TARGET/OFFICIAL_COMMIT_USED.txt" ]]; then
  echo "[code] using pinned TransUNet archive commit: $(cat "$TARGET/OFFICIAL_COMMIT_USED.txt")"
else
  echo "[fatal] TransUNet source has no commit manifest." >&2
  exit 2
fi
python -m pip install --upgrade ml-collections scipy numpy
echo "Official TransUNet available at: $TARGET"
echo "Prepare the main-table ImageNet-21k checkpoint with:"
echo "  bash scripts/prepare_transunet_pretrained.sh /path/to/R50+ViT-B_16.npz"

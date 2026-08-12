#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$ROOT/third_party/Swin-Unet"
mkdir -p "$ROOT/third_party"
if [[ ! -f "$TARGET/networks/vision_transformer.py" ]]; then
  git clone https://github.com/HuCaoFighting/Swin-Unet.git "$TARGET"
fi
if [[ -d "$TARGET/.git" ]]; then
  git -C "$TARGET" rev-parse HEAD | tee "$TARGET/OFFICIAL_COMMIT_USED.txt"
elif [[ -s "$TARGET/OFFICIAL_COMMIT_USED.txt" ]]; then
  echo "[code] using pinned Swin-Unet archive commit: $(cat "$TARGET/OFFICIAL_COMMIT_USED.txt")"
else
  echo "[fatal] Swin-Unet source has no commit manifest." >&2
  exit 2
fi
python -m pip install --upgrade "timm<1.0" einops scipy
echo "Official Swin-Unet available at: $TARGET"

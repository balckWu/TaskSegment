#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${TRANSUNET_PRETRAINED_TARGET:-$ROOT/pretrained/transunet/R50+ViT-B_16.npz}"
SOURCE="${1:-}"
OFFICIAL_URL="https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz"

mkdir -p "$(dirname "$TARGET")"

if [[ -f "$TARGET" ]]; then
  echo "[weights] already present: $TARGET"
elif [[ -n "$SOURCE" && -f "$SOURCE" ]]; then
  cp -f "$SOURCE" "$TARGET"
  echo "[weights] copied from: $SOURCE"
elif [[ -n "$SOURCE" ]]; then
  echo "[weights] downloading from supplied URL"
  if command -v wget >/dev/null 2>&1; then
    wget -c -O "$TARGET.part" "$SOURCE"
  else
    curl -fL --retry 5 --retry-delay 5 -C - -o "$TARGET.part" "$SOURCE"
  fi
  mv -f "$TARGET.part" "$TARGET"
else
  echo "[weights] trying the source documented by the official TransUNet repository"
  set +e
  if command -v wget >/dev/null 2>&1; then
    wget -c -O "$TARGET.part" "$OFFICIAL_URL"
    STATUS=$?
  else
    curl -fL --retry 3 --retry-delay 5 -C - -o "$TARGET.part" "$OFFICIAL_URL"
    STATUS=$?
  fi
  set -e
  if [[ $STATUS -ne 0 ]]; then
    rm -f "$TARGET.part"
    cat >&2 <<EOF
Automatic download failed.
Download R50+ViT-B_16.npz from the source referenced by the bundled official
third_party/TransUNet/README.md, then run either:

  bash scripts/prepare_transunet_pretrained.sh /absolute/path/R50+ViT-B_16.npz

or place it directly at:

  $TARGET
EOF
    exit 2
  fi
  mv -f "$TARGET.part" "$TARGET"
fi

python "$ROOT/scripts/verify_transunet_pretrained.py" "$TARGET"
echo "[weights] TransUNet pretrained checkpoint is ready."

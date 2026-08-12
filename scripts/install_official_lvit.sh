#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TARGET="$ROOT/third_party/LViT"
TMP="$ROOT/third_party/.lvit_install_tmp"
mkdir -p "$ROOT/third_party"
rm -rf "$TMP"

if [[ ! -f "$TARGET/nets/LViT.py" ]]; then
  echo "[install] downloading official HUANGLIZI/LViT repository"
  urls=(
    "https://github.com/HUANGLIZI/LViT.git"
    "https://gh-proxy.com/https://github.com/HUANGLIZI/LViT.git"
    "https://ghfast.top/https://github.com/HUANGLIZI/LViT.git"
    "https://ghproxy.net/https://github.com/HUANGLIZI/LViT.git"
  )
  cloned=0
  for url in "${urls[@]}"; do
    echo "[install] trying: $url"
    if git clone --depth 1 "$url" "$TMP"; then
      cloned=1
      break
    fi
    rm -rf "$TMP"
  done
  if [[ "$cloned" -ne 1 ]]; then
    echo "[fatal] Could not clone official LViT from GitHub or configured mirrors." >&2
    echo "Download https://github.com/HUANGLIZI/LViT and place it at $TARGET" >&2
    exit 2
  fi
  rm -rf "$TARGET"
  mv "$TMP" "$TARGET"
fi

if [[ -d "$TARGET/.git" ]]; then
  git -C "$TARGET" rev-parse HEAD | tee "$TARGET/OFFICIAL_COMMIT_USED.txt"
elif [[ -s "$TARGET/OFFICIAL_COMMIT_USED.txt" ]]; then
  echo "[code] using pinned LViT archive commit: $(cat "$TARGET/OFFICIAL_COMMIT_USED.txt")"
else
  echo "[fatal] LViT archive has no OFFICIAL_COMMIT_USED.txt" >&2
  exit 2
fi

python -m pip install -r requirements-lvit.txt
python scripts/patch_lvit_dynamic_resolution.py --repo "$TARGET"

BERT_DIR="${LVIT_BERT_DIR:-$ROOT/pretrained/bert-base-uncased}"
mkdir -p "$ROOT/pretrained"

if [[ ! -f "$BERT_DIR/config.json" ]]; then
  echo "[install] downloading frozen BERT text encoder through HF mirror"
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
  export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
  python -u - "$BERT_DIR" <<'PY'
import os
import sys
from huggingface_hub import snapshot_download

target = sys.argv[1]
snapshot_download(
    repo_id="google-bert/bert-base-uncased",
    local_dir=target,
)
print("BERT snapshot downloaded:", target)
PY
fi

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python - "$BERT_DIR" <<'PY'
import sys
from transformers import AutoModel, AutoTokenizer

path = sys.argv[1]
AutoTokenizer.from_pretrained(path, local_files_only=True)
model = AutoModel.from_pretrained(path, local_files_only=True)
assert int(model.config.hidden_size) == 768
print("Local BERT verification passed:", path)
PY

python - <<'PY'
from pathlib import Path
repo = Path("third_party/LViT")
required = [
    repo / "nets" / "LViT.py",
    repo / "nets" / "Vit.py",
    repo / "nets" / "pixlevel.py",
    repo / "TASKSEGMENT_DYNAMIC_RESOLUTION_PATCH.txt",
]
missing = [str(p) for p in required if not p.is_file()]
if missing:
    raise SystemExit("Missing official LViT files: " + ", ".join(missing))
print("Official LViT installation verified.")
PY

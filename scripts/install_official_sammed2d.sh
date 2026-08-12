#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OFFICIAL_REPO="$ROOT/third_party/SAM-Med2D"
OFFICIAL_COMMIT="bfd2b93b1158100c8abd81f61766a2de92c1c175"
WEIGHT_DIR="$ROOT/pretrained/sam-med2d"
WEIGHT_FILE="$WEIGHT_DIR/sam-med2d_b.pth"
WEIGHT_REPO="schengal1/SAM-Med2D_model"
WEIGHT_REVISION="ef05f45fff224f2b6b98fdb34a5c74a460b4b243"
EXPECTED_SHA256="921d8c130cc4bcc4b7220c064e60f207f15ad59f17ab36de340e41c67a490d18"

mkdir -p "$ROOT/third_party" "$WEIGHT_DIR"
python -m pip install -r requirements-sammed2d.txt

if [[ ! -f "$OFFICIAL_REPO/segment_anything/build_sam.py" ]]; then
  rm -rf "$OFFICIAL_REPO"
  urls=(
    "https://github.com/OpenGVLab/SAM-Med2D.git"
    "https://gh-proxy.com/https://github.com/OpenGVLab/SAM-Med2D.git"
    "https://ghfast.top/https://github.com/OpenGVLab/SAM-Med2D.git"
    "https://ghproxy.net/https://github.com/OpenGVLab/SAM-Med2D.git"
  )
  success=0
  for url in "${urls[@]}"; do
    echo "[code] trying $url"
    if git clone "$url" "$OFFICIAL_REPO"; then
      success=1
      break
    fi
    rm -rf "$OFFICIAL_REPO"
  done
  if [[ "$success" -ne 1 ]]; then
    echo "[fatal] Could not download the official OpenGVLab/SAM-Med2D code." >&2
    echo "Place the official repository at: $OFFICIAL_REPO" >&2
    exit 2
  fi
fi

if [[ -d "$OFFICIAL_REPO/.git" ]]; then
  git -C "$OFFICIAL_REPO" fetch origin "$OFFICIAL_COMMIT" --depth 1 || true
  git -C "$OFFICIAL_REPO" checkout --detach "$OFFICIAL_COMMIT"
  actual_commit="$(git -C "$OFFICIAL_REPO" rev-parse HEAD)"
  if [[ "$actual_commit" != "$OFFICIAL_COMMIT" ]]; then
    echo "[fatal] Official code commit mismatch: $actual_commit" >&2
    exit 2
  fi
  printf '%s\n' "$actual_commit" > "$OFFICIAL_REPO/OFFICIAL_COMMIT_USED.txt"
elif [[ -s "$OFFICIAL_REPO/OFFICIAL_COMMIT_USED.txt" ]]; then
  actual_commit="$(tr -d '[:space:]' < "$OFFICIAL_REPO/OFFICIAL_COMMIT_USED.txt")"
  if [[ "$actual_commit" != "$OFFICIAL_COMMIT" ]]; then
    echo "[fatal] Pinned SAM-Med2D archive commit mismatch: $actual_commit" >&2
    exit 2
  fi
  echo "[code] using pinned SAM-Med2D archive commit: $actual_commit"
else
  echo "[fatal] Official repository lacks git metadata and commit manifest." >&2
  exit 2
fi

if [[ ! -f "$WEIGHT_FILE" ]]; then
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
  export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
  python -u - "$WEIGHT_DIR" "$WEIGHT_REPO" "$WEIGHT_REVISION" <<'PY'
import sys
from huggingface_hub import snapshot_download

target, repo_id, revision = sys.argv[1:4]
path = snapshot_download(
    repo_id=repo_id,
    revision=revision,
    local_dir=target,
    allow_patterns=["sam-med2d_b.pth"],
)
print("Downloaded community transport mirror snapshot:", path)
PY
fi

actual_sha="$(sha256sum "$WEIGHT_FILE" | awk '{print $1}')"
if [[ "$actual_sha" != "$EXPECTED_SHA256" ]]; then
  echo "[fatal] SAM-Med2D checkpoint SHA-256 mismatch." >&2
  echo "expected=$EXPECTED_SHA256" >&2
  echo "actual=$actual_sha" >&2
  exit 2
fi

cat > "$WEIGHT_DIR/PROVENANCE.txt" <<EOF
Method and implementation:
  OpenGVLab/SAM-Med2D
Official code commit:
  $OFFICIAL_COMMIT

Checkpoint transport:
  Downloaded through HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
  Community mirror repository: $WEIGHT_REPO
  Mirror revision: $WEIGHT_REVISION
  File: sam-med2d_b.pth
  SHA-256: $EXPECTED_SHA256

Important:
  The Hugging Face repository above is a community transport mirror, not the
  official OpenGVLab checkpoint host. The method, code and checkpoint identity
  must be cited to the official SAM-Med2D paper/repository; the mirror is
  disclosed only for byte transport and reproducibility.
EOF

python -u - <<'PY'
from tasksegment.models.sammed2d_official import OfficialSAMMed2DAdapter

model = OfficialSAMMed2DAdapter(
    official_repository_path="./third_party/SAM-Med2D",
    base_checkpoint="./pretrained/sam-med2d/sam-med2d_b.pth",
    model_resolution=256,
    output_resolution=(512, 512),
)
print("Local SAM-Med2D verification passed")
print("load_report:", model.load_report)
print("parameter_report:", model.parameter_report())
PY

echo "[install] complete"

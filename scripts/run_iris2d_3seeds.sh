#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -u scripts/launch_iris2d.py --gpu "${1:-0}"

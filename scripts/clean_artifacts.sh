#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Cleaning local runtime artifacts in: $ROOT_DIR"

rm -rf \
  __pycache__ \
  .ipynb_checkpoints \
  logs \
  runs \
  model_tfg_serial_8mic

find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type d -name ".ipynb_checkpoints" -prune -exec rm -rf {} +
find . -type f \( -name "*.pyc" -o -name "*.pyo" -o -name ".DS_Store" \) -delete

echo "Done."

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Removing generated artifacts from the Git index only."
echo "Local files will be kept on disk."

git rm -r --cached --ignore-unmatch \
  __pycache__ \
  .ipynb_checkpoints \
  .vscode \
  logs \
  runs \
  model_tfg_serial_8mic \
  record

git rm --cached --ignore-unmatch \
  *.pyc \
  *.pyo \
  *.pth \
  *.pt \
  *.ckpt \
  *.npy \
  *.npz \
  *.mat \
  Network_loss.png

echo "Done. Review with: git status --short"

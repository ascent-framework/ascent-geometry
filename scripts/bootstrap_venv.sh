#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

echo "Repository root: ${ROOT_DIR}"
echo "Python binary:   ${PYTHON_BIN}"
echo "Venv path:       ${VENV_DIR}"

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${ROOT_DIR}/requirements.txt"

python "${ROOT_DIR}/scripts/verify_env.py"

echo
echo "Virtual environment is ready."
echo "Activate with: source .venv/bin/activate"

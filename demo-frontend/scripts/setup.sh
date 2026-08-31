#!/usr/bin/env bash
set -euo pipefail

DEMO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_DIR="$(cd "$DEMO_DIR/.." && pwd)"
UV_CACHE_DIR="${UV_CACHE_DIR:-/private/tmp/aigi-detect-uv-cache}"
UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/private/tmp/aigi-detect-uv-python}"
export UV_CACHE_DIR UV_PYTHON_INSTALL_DIR

cd "$DEMO_DIR"
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e "$REPO_DIR" -r server/requirements.txt pytest httpx
npm install --cache /private/tmp/aigi-detect-npm-cache
echo "AIGI Detect Demo dependencies are ready."

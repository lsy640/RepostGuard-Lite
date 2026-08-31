#!/usr/bin/env bash
set -euo pipefail

DEMO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DEMO_DIR"

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv. Run ./scripts/setup.sh first." >&2
  exit 1
fi

export PYTORCH_ENABLE_MPS_FALLBACK=1
npm run build
exec .venv/bin/python -m uvicorn server.app:app --host 127.0.0.1 --port 8000 --workers 1

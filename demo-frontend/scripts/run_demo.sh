#!/usr/bin/env bash
set -euo pipefail

DEMO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DEMO_DIR"

DEMO_HOST="${AIGI_DEMO_HOST:-127.0.0.1}"
DEMO_PORT="${AIGI_DEMO_PORT:-8000}"
DEMO_URL="http://${DEMO_HOST}:${DEMO_PORT}"

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv. Run ./scripts/setup.sh first." >&2
  exit 1
fi

export PYTORCH_ENABLE_MPS_FALLBACK=1
npm run build

listener_pid="$(lsof -nP -tiTCP:"$DEMO_PORT" -sTCP:LISTEN 2>/dev/null | head -n 1 || true)"
if [[ -n "$listener_pid" ]]; then
  listener_cwd="$(lsof -a -p "$listener_pid" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | head -n 1)"
  same_demo=false
  if [[ "$listener_cwd" == "$DEMO_DIR" ]] \
    && curl -fsS "$DEMO_URL/api/health" 2>/dev/null | grep -q '"models"'; then
    same_demo=true
  fi

  if [[ "$same_demo" == true ]]; then
    echo "Restarting existing AIGI Detect Demo on $DEMO_URL (PID $listener_pid)..."
    kill "$listener_pid"
    for _ in {1..80}; do
      if ! kill -0 "$listener_pid" 2>/dev/null; then
        break
      fi
      sleep 0.1
    done
    if kill -0 "$listener_pid" 2>/dev/null; then
      echo "Existing demo process $listener_pid did not stop. Stop it manually and rerun this script." >&2
      exit 1
    fi
  else
    echo "Port $DEMO_PORT is already used by another process (PID $listener_pid)." >&2
    echo "Stop that process or run: AIGI_DEMO_PORT=8001 ./scripts/run_demo.sh" >&2
    exit 1
  fi
fi

echo "Starting AIGI Detect Demo at $DEMO_URL"
exec .venv/bin/python -m uvicorn server.app:app --host "$DEMO_HOST" --port "$DEMO_PORT" --workers 1

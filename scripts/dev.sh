#!/usr/bin/env bash
# Run backend + frontend in dev mode in two background processes.
# Usage: ./scripts/dev.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VENV="$ROOT/backend/.venv"
if [[ ! -x "$VENV/bin/uvicorn" ]]; then
  echo "✗ backend/.venv is missing or incomplete." >&2
  echo "  Run ./scripts/setup.sh first."          >&2
  exit 1
fi

if [[ ! -d "$ROOT/frontend/node_modules" ]]; then
  echo "✗ frontend/node_modules is missing."     >&2
  echo "  Run ./scripts/setup.sh first."          >&2
  exit 1
fi

BACK_LOG="$ROOT/backend.dev.log"
FRONT_LOG="$ROOT/frontend.dev.log"

cleanup() {
  echo
  echo "Stopping dev servers…"
  kill 0 2>/dev/null || true
}
trap cleanup EXIT

echo "→ Starting backend  (logs: $BACK_LOG)"
( cd "$ROOT/backend" && "$VENV/bin/uvicorn" main:app --reload --port 8000 ) >"$BACK_LOG" 2>&1 &

echo "→ Starting frontend (logs: $FRONT_LOG)"
( cd "$ROOT/frontend" && npm run dev ) >"$FRONT_LOG" 2>&1 &

echo
echo "Backend  → http://localhost:8000  (POST /api/analyse, GET /api/result/{id})"
echo "Frontend → http://localhost:5173"
echo
echo "Tail logs with:  tail -f $BACK_LOG  $FRONT_LOG"
wait

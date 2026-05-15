#!/usr/bin/env bash
# One-shot setup: create venv with a supported Python (3.10 or 3.11),
# install Python deps, install npm deps.
#
# Usage:   ./scripts/setup.sh
#          PYTHON=/opt/homebrew/opt/python@3.11/bin/python3.11 ./scripts/setup.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ---------- find a supported interpreter ----------
# py-feat + mediapipe currently have wheels for 3.10 and 3.11 only.
# 3.12 has partial wheel coverage; 3.13/3.14 do not work yet.

find_python() {
  # 1. Honour an explicit override.
  if [[ -n "${PYTHON:-}" ]]; then
    echo "$PYTHON"; return
  fi
  # 2. Look for python3.11 / python3.10 on PATH.
  for cand in python3.11 python3.10; do
    if command -v "$cand" >/dev/null 2>&1; then
      echo "$cand"; return
    fi
  done
  # 3. Common Homebrew install paths.
  for cand in \
    /opt/homebrew/opt/python@3.11/bin/python3.11 \
    /opt/homebrew/opt/python@3.10/bin/python3.10 \
    /usr/local/opt/python@3.11/bin/python3.11 \
    /usr/local/opt/python@3.10/bin/python3.10; do
    if [[ -x "$cand" ]]; then echo "$cand"; return; fi
  done
  # 4. pyenv shims.
  if command -v pyenv >/dev/null 2>&1; then
    pyenv_path="$(pyenv which python3.11 2>/dev/null || pyenv which python3.10 2>/dev/null || true)"
    if [[ -n "$pyenv_path" ]]; then echo "$pyenv_path"; return; fi
  fi
  echo ""; return
}

PY="$(find_python)"

if [[ -z "$PY" ]]; then
  cat <<'EOF' >&2

✗ Could not find Python 3.10 or 3.11.

py-feat / mediapipe / numexpr do not yet have wheels for Python 3.12+,
so installing on 3.14 fails when pip tries to build numexpr from source.

Install Python 3.11 with one of:

    # Homebrew (recommended on macOS):
    brew install python@3.11

    # pyenv:
    brew install pyenv && pyenv install 3.11.9

Then re-run:
    ./scripts/setup.sh

Or, if you already have a 3.11 binary somewhere, point us at it:
    PYTHON=/path/to/python3.11 ./scripts/setup.sh
EOF
  exit 1
fi

# Verify it's actually 3.10/3.11.
VER="$("$PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
case "$VER" in
  3.10|3.11) ;;
  *)
    echo "✗ $PY is Python $VER — need 3.10 or 3.11. Override with PYTHON=… env var." >&2
    exit 1
    ;;
esac

echo "→ Using $PY  (Python $VER)"

# ---------- backend venv ----------
VENV="$ROOT/backend/.venv"
if [[ -d "$VENV" ]]; then
  EXISTING="$("$VENV/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "?")"
  if [[ "$EXISTING" != "$VER" ]]; then
    echo "→ Existing venv is Python $EXISTING — rebuilding for $VER"
    rm -rf "$VENV"
  fi
fi

if [[ ! -d "$VENV" ]]; then
  echo "→ Creating venv at backend/.venv"
  "$PY" -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip wheel setuptools

# Pre-install numpy so any source-build dep can see it via PEP 517 build envs.
# Pinned exactly because nltools (a py-feat dep) caps numpy at <1.24 while
# opencv 4.9 needs >=1.23.5 — 1.23.5 is the only valid intersection.
pip install --prefer-binary "numpy==1.23.5"

# Main install — every version above is pinned so pip can't backtrack into
# sdist-only territory. --prefer-binary keeps the resolver on the wheel path.
pip install --prefer-binary -r backend/requirements.txt
[ -f backend/.env ] || cp backend/.env.example backend/.env

# ---------- frontend ----------
echo "→ Installing npm deps"
( cd frontend && npm install )
[ -f frontend/.env ] || cp frontend/.env.example frontend/.env

cat <<'EOF'

✓ Setup complete.

Activate the backend env:
    source backend/.venv/bin/activate

Then start both dev servers (or use scripts/dev.sh):
    cd backend  && uvicorn main:app --reload --port 8000
    cd frontend && npm run dev
EOF

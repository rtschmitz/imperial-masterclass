#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_VERSION="3.2.8"
ROOT_FILE="${MASTERCLASS_ROOT:-$PROJECT_DIR/data/zz4l_masterclass.root}"
METADATA_FILE="${MASTERCLASS_METADATA:-$PROJECT_DIR/data/zz4l_masterclass.metadata.json}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"

echo "============================================================"
echo " Particle Physics Analysis Tool v$APP_VERSION"
echo "============================================================"
echo "Project:  $PROJECT_DIR"
echo "Backend:  $PROJECT_DIR/app.py"
echo "Frontend: $PROJECT_DIR/static"
echo "Data:     $ROOT_FILE"
echo "Metadata: $METADATA_FILE"
echo

if [[ ! -f "$ROOT_FILE" ]]; then
  echo "Missing reduced ROOT file: $ROOT_FILE"
  echo "Copy zz4l_masterclass.root into $PROJECT_DIR/data or set MASTERCLASS_ROOT."
  exit 1
fi

if [[ ! -f "$METADATA_FILE" ]]; then
  echo "Missing metadata file: $METADATA_FILE"
  echo "Copy zz4l_masterclass.metadata.json into $PROJECT_DIR/data or set MASTERCLASS_METADATA."
  exit 1
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --disable-pip-version-check -r "$PROJECT_DIR/requirements.txt"

export MASTERCLASS_ROOT="$ROOT_FILE"
export MASTERCLASS_METADATA="$METADATA_FILE"
export PYTHONDONTWRITEBYTECODE=1

cd "$PROJECT_DIR"
echo "Starting v$APP_VERSION at http://localhost:${PORT:-8000}/?v=$APP_VERSION"
exec python -m uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}"

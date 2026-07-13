#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8000}"

echo "Serving the static masterclass at http://localhost:$PORT/"
echo "No application backend is running; this is only a static file server."
exec python3 -m http.server "$PORT" --directory "$PROJECT_DIR/site"

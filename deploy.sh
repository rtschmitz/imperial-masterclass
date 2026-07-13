#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-$HOME/public_html/masterclass}"

mkdir -p "$TARGET"
rsync -av "$PROJECT_DIR/site/" "$TARGET/"
find "$TARGET" -type d -exec chmod 755 {} +
find "$TARGET" -type f -exec chmod 644 {} +

echo
echo "Static site copied to: $TARGET"
echo "No existing files were deleted."
echo "For the default target, test:"
echo "  https://homes.hep.ph.ic.ac.uk/~$(id -un)/masterclass/"

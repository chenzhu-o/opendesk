#!/usr/bin/env bash
# Build the OpenDesk MCP Bundle (.mcpb) for one-click install in Claude Desktop.
#
# The bundle uses the MCPB `uv` server type (manifest_version 0.4): it ships
# the opendesk Python source plus pyproject.toml, and the host installs the
# dependencies with uv on first launch.  No preinstalled `opendesk-mcp`
# binary or Python is required on the user's machine.
#
# Usage:
#   scripts/build_mcpb.sh            # → dist/opendesk-<version>.mcpb
#   scripts/build_mcpb.sh out.mcpb   # custom output path
#
# Requires Node.js (for `npx -y @anthropic-ai/mcpb`).  Set MCPB="mcpb" to use
# a globally installed CLI instead.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/python/pyproject.toml" | head -n1)"
OUT="${1:-$ROOT/dist/opendesk-$VERSION.mcpb}"
MCPB="${MCPB:-npx -y @anthropic-ai/mcpb@2}"

MANIFEST_VERSION="$(sed -n 's/^  "version": "\(.*\)",/\1/p' "$ROOT/manifest.json" | head -n1)"
if [[ "$MANIFEST_VERSION" != "$VERSION" ]]; then
  echo "ERROR: manifest.json version ($MANIFEST_VERSION) != python/pyproject.toml version ($VERSION)" >&2
  exit 1
fi

STAGE="$(mktemp -d "${TMPDIR:-/tmp}/opendesk-mcpb.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT

echo "Staging bundle in $STAGE"
cp "$ROOT/manifest.json" "$STAGE/manifest.json"
cp "$ROOT/.mcpbignore" "$STAGE/.mcpbignore"
# Drop [project.optional-dependencies]: `uv run` resolves every extra for
# every platform, and the bundle only needs the base dependencies (the MCP
# server and local computer use).
awk '/^\[/ { skip = ($0 == "[project.optional-dependencies]") } !skip { print }' \
  "$ROOT/python/pyproject.toml" > "$STAGE/pyproject.toml"
cp "$ROOT/python/README.md" "$STAGE/README.md"
cp "$ROOT/LICENSE" "$STAGE/LICENSE"
cp "$ROOT/PRIVACY.md" "$STAGE/PRIVACY.md"
# Copy the package source without caches.
(cd "$ROOT/python" && tar --exclude='__pycache__' --exclude='*.pyc' -cf - opendesk) | (cd "$STAGE" && tar -xf -)

# 512x512 icon from the docs logo (falls back to a plain copy without Pillow).
ICON_SRC="$ROOT/docs/_static/logo.png"
PY="${PYTHON:-python3}"
if ! "$PY" - "$ICON_SRC" "$STAGE/icon.png" <<'PYEOF' 2>/dev/null
import sys
from PIL import Image
img = Image.open(sys.argv[1]).convert("RGBA")
img.thumbnail((512, 512))
img.save(sys.argv[2], optimize=True)
PYEOF
then
  cp "$ICON_SRC" "$STAGE/icon.png"
fi

echo "Validating manifest"
$MCPB validate "$STAGE/manifest.json"

mkdir -p "$(dirname "$OUT")"
echo "Packing $OUT"
$MCPB pack "$STAGE" "$OUT"
echo "Built $OUT"

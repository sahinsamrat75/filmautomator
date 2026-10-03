#!/bin/sh
# Build the Claude Desktop extension (.mcpb) from packaging/mcpb/.
#
# Reproducible: the bundle contains only the manifest, the launcher and the
# README, because the server itself is the Filmautomator checkout the user
# points the extension at — not a snapshot that would drift from it.
#
# Uses the official MCPB toolchain (free, MIT) via npx. Nothing is installed
# permanently.
set -e
cd "$(dirname "$0")/.."

MCPB_VERSION="${MCPB_VERSION:-2.1.2}"
SRC="packaging/mcpb"
OUT="dist/filmautomator.mcpb"

echo "==> validating manifest"
npx --yes "@anthropic-ai/mcpb@${MCPB_VERSION}" validate "$SRC/manifest.json"

echo
echo "==> packing $SRC -> $OUT"
mkdir -p dist
npx --yes "@anthropic-ai/mcpb@${MCPB_VERSION}" pack "$SRC" "$OUT"

echo
echo "==> verifying the packed bundle by running its MCP handshake"
# Unpack and test the artifact that ships, not the source it came from.
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
npx --yes "@anthropic-ai/mcpb@${MCPB_VERSION}" unpack "$OUT" "$WORK" >/dev/null

FA_TEST_PROJECT_DIR="$(pwd)" \
FA_TEST_PYTHON="$(command -v python3)" \
python3 tests/verify_mcpb_bundle.py "$WORK/manifest.json"

echo
echo "Bundle: $(pwd)/${OUT}"

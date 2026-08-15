#!/bin/bash
# Build a distributable macOS package (DMG) of the Voicebox desktop app.
#
# Usage:
#   ./scripts/build-macos.sh
#
# Produces:
#   tauri/src-tauri/target/release/bundle/macos/Voicebox.app
#   tauri/src-tauri/target/release/bundle/dmg/Voicebox_<version>_<arch>.dmg
#
# Notes:
# - The DMG is ad-hoc signed and NOT notarized (no Developer ID on the build
#   machine), so recipients must right-click -> Open (or run
#   `xattr -cr /Applications/Voicebox.app`) once.
# - Updater artifacts are skipped locally because signing them requires the
#   TAURI_SIGNING_PRIVATE_KEY, which only exists in CI secrets. The release
#   pipeline (.github/workflows/release.yml) still produces them.
# - Only the host architecture is built (aarch64 on Apple Silicon).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAURI_DIR="$ROOT/tauri"
DMG_DIR="$TAURI_DIR/src-tauri/target/release/bundle/dmg"

echo "==> Checking prerequisites..."
command -v rustc >/dev/null 2>&1 || { echo "ERROR: rustc not found — install Rust (https://rustup.rs)" >&2; exit 1; }
command -v bun >/dev/null 2>&1 || { echo "ERROR: bun not found — install Bun (https://bun.sh)" >&2; exit 1; }
if [ ! -x "$ROOT/backend/venv/bin/python" ]; then
    echo "ERROR: backend/venv not found — run the project setup first (see docs/content/docs/developer/building.mdx)" >&2
    exit 1
fi

PLATFORM="$(rustc --print host-tuple)"
echo "==> Building sidecars for platform: $PLATFORM"
"$ROOT/scripts/build-server.sh"

run_tauri_build() {
    (cd "$TAURI_DIR" && bun run tauri build --config '{"bundle":{"createUpdaterArtifacts":false}}')
}

echo "==> Building Tauri app + DMG..."
if ! run_tauri_build; then
    echo "==> First tauri build failed; retrying once (DMG bundling can hit transient volume-busy errors)..."
    if ! run_tauri_build; then
        echo "ERROR: tauri build failed twice" >&2
        exit 1
    fi
fi

DMG="$(ls -1 "$DMG_DIR"/*.dmg 2>/dev/null | head -1 || true)"
if [ -z "$DMG" ]; then
    echo "ERROR: no DMG produced in $DMG_DIR" >&2
    exit 1
fi

# Keep `tauri dev` on the intended dev-server workflow: replace the packaged
# sidecars with the dev-mode placeholder scripts (the real sidecars stay in
# backend/dist and are re-copied by the next build).
PLACEHOLDER_SERVER="#!/bin/sh
echo \"[voicebox-server] Dev mode placeholder - start the real server with: bun run dev:server\"
exit 1"
PLACEHOLDER_MCP="#!/bin/sh
echo \"[voicebox-mcp] Dev mode placeholder - start the real server with: bun run dev:server\"
exit 1"
printf '%s\n' "$PLACEHOLDER_SERVER" > "$TAURI_DIR/src-tauri/binaries/voicebox-server-$PLATFORM"
printf '%s\n' "$PLACEHOLDER_MCP" > "$TAURI_DIR/src-tauri/binaries/voicebox-mcp-$PLATFORM"
chmod +x "$TAURI_DIR/src-tauri/binaries/voicebox-server-$PLATFORM" \
    "$TAURI_DIR/src-tauri/binaries/voicebox-mcp-$PLATFORM"

echo
echo "=================================================="
echo "macOS package ready:"
echo "  $DMG"
echo
echo "Distribute the .dmg. Recipients may need to right-click the app"
echo "-> Open once (or run: xattr -cr /Applications/Voicebox.app) because"
echo "this local build is ad-hoc signed and not notarized."
echo "=================================================="

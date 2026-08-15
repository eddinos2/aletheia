#!/usr/bin/env bash
# Build a unsigned Aletheia.app for local macOS use.
# Usage: ./scripts/macos-app.sh [--release]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROFILE=debug
CARGO_FLAGS=()
if [[ "${1:-}" == "--release" ]]; then
  PROFILE=release
  CARGO_FLAGS=(--release)
fi

cd "$ROOT"
cargo build -p aletheia-gui "${CARGO_FLAGS[@]}"

BIN="$ROOT/target/$PROFILE/aletheia-gui"
OUT="$ROOT/dist/Aletheia.app"
rm -rf "$OUT"
mkdir -p "$OUT/Contents/MacOS" "$OUT/Contents/Resources"
cp "$BIN" "$OUT/Contents/MacOS/aletheia-gui"
cp "$ROOT/crates/aletheia-gui/resources/Info.plist" "$OUT/Contents/Info.plist"
chmod +x "$OUT/Contents/MacOS/aletheia-gui"

# Ad-hoc sign so Gatekeeper is slightly happier on local machines.
if command -v codesign >/dev/null 2>&1; then
  codesign --force --deep --sign - "$OUT" 2>/dev/null || true
fi

echo "Built $OUT"
echo "Run: open \"$OUT\""

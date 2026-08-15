#!/usr/bin/env bash
# Build an unsigned Aletheia.app for local macOS use.
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
if ((${#CARGO_FLAGS[@]})); then
  cargo build -p aletheia-gui "${CARGO_FLAGS[@]}"
else
  cargo build -p aletheia-gui
fi

BIN="$ROOT/target/$PROFILE/aletheia-gui"
if [[ ! -x "$BIN" ]]; then
  echo "error: expected binary missing: $BIN" >&2
  exit 1
fi

OUT="$ROOT/dist/Aletheia.app"
mkdir -p "$ROOT/dist"
rm -rf "$OUT"
mkdir -p "$OUT/Contents/MacOS" "$OUT/Contents/Resources"
cp "$BIN" "$OUT/Contents/MacOS/aletheia-gui"
cp "$ROOT/crates/aletheia-gui/resources/Info.plist" "$OUT/Contents/Info.plist"
chmod +x "$OUT/Contents/MacOS/aletheia-gui"

# Ad-hoc sign so Gatekeeper is slightly happier on local machines.
# Unsigned path remains valid if codesign is absent or fails.
if command -v codesign >/dev/null 2>&1; then
  codesign --force --deep --sign - "$OUT" 2>/dev/null || \
    echo "note: ad-hoc codesign skipped; app is still runnable unsigned" >&2
fi

echo "Built $OUT"
echo "Run: open \"$OUT\""

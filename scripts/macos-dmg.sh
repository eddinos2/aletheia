#!/usr/bin/env bash
# Produce dist/Aletheia-<version>-unsigned.dmg from Aletheia.app.
# Prefer create-dmg when installed; otherwise fall back to hdiutil.
# Usage: ./scripts/macos-dmg.sh [--release]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Forward only a real flag (empty "${1:-}" confused older callers).
if [[ "${1:-}" == "--release" ]]; then
  "$ROOT/scripts/macos-app.sh" --release
else
  "$ROOT/scripts/macos-app.sh"
fi

VERSION="$(grep '^version' "$ROOT/crates/aletheia-gui/Cargo.toml" | head -1 | cut -d'"' -f2)"
if [[ -z "$VERSION" ]]; then
  echo "error: could not read version from crates/aletheia-gui/Cargo.toml" >&2
  exit 1
fi

APP="$ROOT/dist/Aletheia.app"
DMG="$ROOT/dist/Aletheia-${VERSION}-unsigned.dmg"
STAGE="$ROOT/dist/dmg-stage"

if [[ ! -d "$APP" ]]; then
  echo "error: missing $APP — macos-app.sh failed?" >&2
  exit 1
fi

cleanup() { rm -rf "$STAGE"; }
trap cleanup EXIT

rm -rf "$STAGE" "$DMG"
mkdir -p "$STAGE"
cp -R "$APP" "$STAGE/"
ln -sf /Applications "$STAGE/Applications"

if command -v create-dmg >/dev/null 2>&1; then
  # create-dmg exits non-zero if the target exists; we already removed it.
  create-dmg \
    --volname "Aletheia" \
    --window-pos 200 120 \
    --window-size 660 400 \
    --icon-size 100 \
    --icon "Aletheia.app" 160 180 \
    --app-drop-link 480 180 \
    "$DMG" \
    "$STAGE" || {
      echo "create-dmg failed; falling back to hdiutil" >&2
      hdiutil create \
        -volname "Aletheia" \
        -srcfolder "$STAGE" \
        -ov -format UDZO \
        "$DMG"
    }
else
  hdiutil create \
    -volname "Aletheia" \
    -srcfolder "$STAGE" \
    -ov -format UDZO \
    "$DMG"
fi

if [[ ! -f "$DMG" ]]; then
  echo "error: DMG was not created" >&2
  exit 1
fi

echo "DMG: $DMG"
echo "Note: unsigned local build. For distribution, notarize with an Apple Developer ID."
echo "See crates/aletheia-gui/README.md § Notarization."

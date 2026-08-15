#!/usr/bin/env bash
# Produce dist/Aletheia-<version>-unsigned.dmg from Aletheia.app.
# Prefer create-dmg when installed; otherwise fall back to hdiutil.
# Usage: ./scripts/macos-dmg.sh [--release]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
"$ROOT/scripts/macos-app.sh" "${1:-}"

VERSION="$(grep '^version' "$ROOT/crates/aletheia-gui/Cargo.toml" | head -1 | cut -d'"' -f2)"
APP="$ROOT/dist/Aletheia.app"
DMG="$ROOT/dist/Aletheia-${VERSION}-unsigned.dmg"
STAGE="$ROOT/dist/dmg-stage"
rm -rf "$STAGE" "$DMG"
mkdir -p "$STAGE"
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"

if command -v create-dmg >/dev/null 2>&1; then
  create-dmg \
    --volname "Aletheia" \
    --window-pos 200 120 \
    --window-size 660 400 \
    --icon-size 100 \
    --icon "Aletheia.app" 160 180 \
    --app-drop-link 480 180 \
    "$DMG" \
    "$STAGE"
else
  hdiutil create \
    -volname "Aletheia" \
    -srcfolder "$STAGE" \
    -ov -format UDZO \
    "$DMG"
fi

rm -rf "$STAGE"
echo "DMG: $DMG"
echo "Note: unsigned local build. For distribution, notarize with an Apple Developer ID."
echo "See crates/aletheia-gui/README.md § Notarization."

#!/bin/zsh
# Crea dist/opentraductor-<version>.dmg con la app y el atajo a /Applications. Uso: make_dmg.sh [version] [identidad para firmar el dmg]
set -euo pipefail
HERE="${0:A:h}"; ROOT="${HERE:h:h}"; VERSION="${1:-0.1.0}"; ID="${2:-}"
APP="$ROOT/dist/opentraductor.app"; DMG="$ROOT/dist/opentraductor-$VERSION.dmg"; STAGE="$(mktemp -d)"
[[ -d "$APP" ]] || { echo "No existe $APP" >&2; exit 1; }
ditto "$APP" "$STAGE/opentraductor.app"; ln -s /Applications "$STAGE/Applications"
rm -f "$DMG"
hdiutil create -volname "opentraductor" -srcfolder "$STAGE" -ov -format UDZO -imagekey zlib-level=6 "$DMG" >/dev/null
rm -rf "$STAGE"
[[ -n "$ID" && "$ID" != "-" ]] && codesign --force --sign "$ID" --timestamp "$DMG"
du -h "$DMG"; echo "listo: $DMG"

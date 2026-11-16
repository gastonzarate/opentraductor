#!/bin/zsh
# Firma dist/opentraductor.app de adentro hacia afuera (hardened runtime). Uso:
#   sign_app.sh "Developer ID Application: Nombre (TEAMID)"     firma real
#   sign_app.sh -                                                firma ad hoc (ensayo del recorrido, sin timestamp)
set -euo pipefail
HERE="${0:A:h}"; ROOT="${HERE:h:h}"; APP="${APP:-$ROOT/dist/opentraductor.app}"; ID="${1:?Falta la identidad de firma (o '-' para ensayo)}"
ENT="$HERE/entitlements.plist"
TS=(--timestamp); [[ "$ID" == "-" ]] && TS=(--timestamp=none)
[[ -d "$APP" ]] || { echo "No existe $APP. Ejecutá build_app.sh primero." >&2; exit 1; }
xattr -cr "$APP"
n=0
# 1) Todo Mach-O suelto (.so, .dylib, ejecutables) con los entitlements, el más profundo primero.
while IFS= read -r f; do
  if file -b "$f" | grep -q "Mach-O"; then
    codesign --force --sign "$ID" --options runtime "${TS[@]}" --entitlements "$ENT" "$f"; n=$((n+1))
  fi
done < <(find "$APP/Contents" -type f \( -perm -u+x -o -name '*.so' -o -name '*.dylib' -o -name '*.a' \) ! -path '*/Contents/MacOS/*' | awk '{print length, $0}' | sort -rn | cut -d' ' -f2-)
echo "firmados $n binarios internos"
# 2) El bundle completo (firma el ejecutable del shell y sella Resources).
codesign --force --sign "$ID" --options runtime "${TS[@]}" --entitlements "$ENT" "$APP"
codesign --verify --deep --strict --verbose=2 "$APP"
echo "firma OK: $APP"

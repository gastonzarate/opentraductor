#!/bin/zsh
# Firma una copia de la .app para la Mac App Store (App Sandbox) y arma el .pkg para subir con Transporter.
# Uso:
#   sign_store.sh -                                   ensayo local con firma ad hoc (no sirve para subir)
#   TEAMID=ABCDE12345 PROFILE=/ruta/OpenTraductor.provisionprofile \
#     sign_store.sh "Apple Distribution: Nombre (ABCDE12345)" "3rd Party Mac Developer Installer: Nombre (ABCDE12345)"
# Entrada: dist/opentraductor.app (build_app.sh). Salida: dist/store/OpenTraductor.app y dist/store/OpenTraductor-<version>.pkg
set -euo pipefail
HERE="${0:A:h}"; ROOT="${HERE:h:h}"; ID="${1:?Falta la identidad (o '-' para ensayo)}"; INSTALLER_ID="${2:-}"
SRC="${SRC:-$ROOT/dist/opentraductor.app}"; OUTD="${STORE_OUT:-$ROOT/dist/store}"; APP="$OUTD/OpenTraductor.app"
BUNDLE_ID="${BUNDLE_ID:-com.opentraductor.app}"
[[ -d "$SRC" ]] || { echo "No existe $SRC. Ejecutá build_app.sh primero." >&2; exit 1; }
if [[ "$ID" != "-" ]]; then
  : "${TEAMID:?Falta TEAMID}"; : "${PROFILE:?Falta PROFILE (perfil de aprovisionamiento de Mac App Store)}"
  [[ -f "$PROFILE" ]] || { echo "No existe $PROFILE" >&2; exit 1; }
  [[ -n "$INSTALLER_ID" ]] || { echo "Falta la identidad del instalador (3rd Party Mac Developer Installer)" >&2; exit 1; }
fi

echo "==> copia de trabajo en $APP"
rm -rf "$OUTD"; mkdir -p "$OUTD"; cp -cR "$SRC" "$APP"
PL="$APP/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleIdentifier $BUNDLE_ID" "$PL"
/usr/libexec/PlistBuddy -c "Delete :OTEdition" "$PL" 2>/dev/null || true
/usr/libexec/PlistBuddy -c "Add :OTEdition string store" "$PL"
/usr/libexec/PlistBuddy -c "Delete :ITSAppUsesNonExemptEncryption" "$PL" 2>/dev/null || true
/usr/libexec/PlistBuddy -c "Add :ITSAppUsesNonExemptEncryption bool false" "$PL"
rm -rf "$APP/Contents/_CodeSignature"
# Herramientas de compilación que no se usan en ejecución.
find "$APP/Contents/Resources/python" -type f \( -name 'protoc' -o -name 'protoc-*' \) -delete
# T-49: Moonshine es el ASR de la tienda (libmoonshine pide macOS 15, por eso LSMinimumSystemVersion=15.0). Ya no se quita.
# Guía 2.4.5(vi): sin código de licencia propio en la tienda. Desde T-70 el runtime no tiene license.py; se verifica.
RT="$APP/Contents/Resources/app/runtime"
if [[ -e "$RT/license.py" ]] || grep -rqE '^\s*(from|import)\s+(\.|runtime\.)license' --include='*.py' "$RT"; then
  echo "ERROR: el bundle lleva código de licencia; la tienda no puede llevarlo (2.4.5(vi))" >&2
  exit 1
fi
# Apple rechaza symlinks que apuntan fuera del bundle.
BAD="$(find "$APP" -type l -exec sh -c 'case "$(readlink "$1")" in /*) echo "$1";; esac' _ {} \;)"
[[ -z "$BAD" ]] || { echo "symlinks absolutos:\n$BAD" >&2; exit 1; }

APP_ENT="$OUTD/app.entitlements"; CHILD_ENT="$OUTD/child.entitlements"
cp "$HERE/store/app.entitlements" "$APP_ENT"; cp "$HERE/store/child.entitlements" "$CHILD_ENT"
if [[ "$ID" != "-" ]]; then
  cp "$PROFILE" "$APP/Contents/embedded.provisionprofile"
  for E in "$APP_ENT" "$CHILD_ENT"; do
    /usr/libexec/PlistBuddy -c "Add :com.apple.application-identifier string $TEAMID.$BUNDLE_ID" "$E"
    /usr/libexec/PlistBuddy -c "Add :com.apple.developer.team-identifier string $TEAMID" "$E"
  done
fi

echo "==> firmando de adentro hacia afuera"
n=0
while IFS=$'\t' read -r kind f; do
  [[ "$f" == "$APP/Contents/MacOS/"* ]] && continue
  if [[ "$kind" == exec ]]; then
    codesign --force --sign "$ID" --timestamp=none --entitlements "$CHILD_ENT" "$f"
  else
    codesign --force --sign "$ID" --timestamp=none "$f"
  fi
  n=$((n+1))
done < <(python3 "$HERE/store/classify_macho.py" "$APP/Contents")
echo "firmados $n binarios internos"
codesign --force --sign "$ID" --timestamp=none --entitlements "$APP_ENT" "$APP"
codesign --verify --deep --strict --verbose=2 "$APP"
echo "firma OK: $APP"

if [[ "$ID" != "-" ]]; then
  VER="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$PL")"
  PKG="$OUTD/OpenTraductor-$VER.pkg"
  productbuild --component "$APP" /Applications --sign "$INSTALLER_ID" "$PKG"
  echo "pkg listo: $PKG  (subir con la app Transporter)"
fi

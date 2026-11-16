#!/bin/zsh
# Firma, arma el .dmg, lo notariza con Apple y le engrapa el ticket. Lo corre la persona con su cuenta de Apple.
# Una sola vez (guarda las credenciales en el Llavero; usá una contraseña específica de app de appleid.apple.com):
#   xcrun notarytool store-credentials opentraductor-notary --apple-id TU@CORREO --team-id TEAMID
# Cada release:
#   packaging/macos/build_app.sh 1.0.0
#   packaging/macos/sign_and_notarize.sh "Developer ID Application: Tu Nombre (TEAMID)" 1.0.0
set -euo pipefail
HERE="${0:A:h}"; ROOT="${HERE:h:h}"; ID="${1:?Falta 'Developer ID Application: Nombre (TEAMID)'}"; VERSION="${2:-0.1.0}"; PROFILE="${NOTARY_PROFILE:-opentraductor-notary}"
DMG="$ROOT/dist/opentraductor-$VERSION.dmg"
"$HERE/sign_app.sh" "$ID"
"$HERE/make_dmg.sh" "$VERSION" "$ID"
echo "==> notarizando (puede tardar de 2 a 30 minutos)"
xcrun notarytool submit "$DMG" --keychain-profile "$PROFILE" --wait | tee "$ROOT/dist/notary.log"
grep -q "status: Accepted" "$ROOT/dist/notary.log" || { echo "Apple no aceptó el paquete. Detalle: xcrun notarytool log <id> --keychain-profile $PROFILE" >&2; exit 1; }
xcrun stapler staple "$DMG"
xcrun stapler validate "$DMG"
spctl --assess --type open --context context:primary-signature -v "$DMG"
shasum -a 256 "$DMG" | tee "$DMG.sha256"
echo "listo para publicar: $DMG"

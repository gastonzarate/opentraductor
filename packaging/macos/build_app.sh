#!/bin/zsh
# Arma dist/opentraductor.app (shell Swift + Python 3.12 embebido + runtime + UI) y la firma "ad hoc" para probar.
# Uso: packaging/macos/build_app.sh [version]      Salida: dist/opentraductor.app
# La firma con Developer ID y la notarización van aparte: packaging/macos/sign_and_notarize.sh
set -euo pipefail
HERE="${0:A:h}"; ROOT="${HERE:h:h}"; VERSION="${1:-0.1.0}"; BUILD="$(date +%y%m%d%H%M)"
OUT="${OUT:-$ROOT/dist}"; APP="$OUT/opentraductor.app"; RES="$APP/Contents/Resources"
# Intérprete base relocatable de uv (python-build-standalone). Se usa sys.base_prefix para no copiar un venv por error.
PYSRC="$(cd /tmp && uv python find 3.12 --no-project | head -1)"
PYHOME="$("$PYSRC" -c 'import sys; print(sys.base_prefix)')"
[[ -x "$PYHOME/bin/python3.12" ]] || { echo "No encuentro Python 3.12 de uv ($PYHOME). Ejecutá: uv python install 3.12" >&2; exit 1; }

echo "==> limpiando $APP"; rm -rf "$APP"; mkdir -p "$APP/Contents/MacOS" "$RES/app"

echo "==> compilando el shell (Swift)"
swiftc -O -target arm64-apple-macos14.0 -o "$APP/Contents/MacOS/OpenTraductor" "$HERE"/Sources/OpenTraductor/*.swift
sed -e "s/__VERSION__/$VERSION/" -e "s/__BUILD__/$BUILD/" "$HERE/Info.plist" > "$APP/Contents/Info.plist"

echo "==> copiando runtime y UI"
rsync -a --exclude '__pycache__' --exclude '*.pyc' "$ROOT/runtime" "$ROOT/ui" "$RES/app/"
cp "$ROOT/scripts/setup_moonshine.py" "$RES/app/" 2>/dev/null || true

echo "==> copiando Python 3.12 embebido desde $PYHOME"
rsync -a --exclude '__pycache__' --exclude 'test' --exclude 'tests' --exclude 'idlelib' --exclude 'tkinter' \
  --exclude 'turtledemo' --exclude 'ensurepip' --exclude '/lib/python3.12/site-packages' --exclude 'pyvenv.cfg' "$PYHOME/" "$RES/python/"
mkdir -p "$RES/python/lib/python3.12/site-packages"
rm -f "$RES/python/lib/python3.12/EXTERNALLY-MANAGED"
# Nada del bundle puede apuntar fuera de él: la notarización y la firma lo rechazan.
BAD="$(find "$RES/python" -type l -exec sh -c 'case "$(readlink "$1")" in /*) echo "$1";; esac' _ {} \;)"
[[ -z "$BAD" ]] || { echo "symlinks absolutos en el Python embebido:\n$BAD" >&2; exit 1; }

echo "==> instalando dependencias (uv, versiones fijadas en uv.lock)"
REQ="$(mktemp)"
( cd "$ROOT" && uv export --frozen --no-dev --no-hashes --no-emit-project --extra runtime --extra moonshine -o "$REQ" >/dev/null )
uv pip install --python "$RES/python/bin/python3.12" --break-system-packages --no-compile -r "$REQ" \
  --target "$RES/python/lib/python3.12/site-packages" 2>&1 | tail -3
rm -f "$REQ"
# uv resuelve las ruedas para la Mac que compila (macOS 26 hoy) y MLX publica una por versión de macOS: se fuerza
# la de macOS 14 para que la app corra en Sonoma o posterior (si no, MLX exige macOS 26 y se pierde casi todo el mercado).
MLX_URLS=("${(@f)$(python3 - "$ROOT/uv.lock" <<'PY'
import re, sys
t = open(sys.argv[1]).read()
for pkg, tag in (("mlx", r"cp312-cp312-macosx_14_0_arm64"), ("mlx-metal", r"py3-none-macosx_14_0_arm64")):
    blk = t.split(f'name = "{pkg}"\nversion', 1)[1].split("\n[[package]]", 1)[0]
    print(re.search(r'url = "([^"]*' + tag + r'\.whl)"', blk).group(1))
PY
)}")
(( ${#MLX_URLS} == 2 )) || { echo "No encuentro las ruedas de MLX para macOS 14 en uv.lock" >&2; exit 1; }
uv pip install --python "$RES/python/bin/python3.12" --break-system-packages --no-compile --no-deps --reinstall \
  --target "$RES/python/lib/python3.12/site-packages" "${MLX_URLS[@]}" 2>&1 | tail -3
find "$RES/python" -name '__pycache__' -type d -prune -exec rm -rf {} +
find "$RES/python/lib/python3.12/site-packages" -type d \( -name tests -o -name test \) -prune -exec rm -rf {} + 2>/dev/null || true

echo "==> firma ad hoc (solo para pruebas locales y de QA; no pasa Gatekeeper en otras Macs sin 'abrir igual')"
codesign --force --deep --sign - --entitlements "$HERE/entitlements.plist" "$APP" 2>&1 | tail -2 || true

du -sh "$APP"
echo "listo: $APP"

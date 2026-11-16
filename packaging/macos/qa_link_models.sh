#!/bin/zsh
# SOLO PARA QA / desarrollo: pone los modelos que ya tenés descargados (caché de Hugging Face y de opentraductor)
# en la carpeta de datos de la app, para probar el build sin bajar ~5 GB. El usuario final usa la descarga guiada.
#   qa_link_models.sh           → app directa: symlinks en ~/Library/Application Support/OpenTraductor
#   qa_link_models.sh --store   → app de tienda (sandbox): clones APFS (cp -c, no ocupan espacio extra) en el
#                                 contenedor, porque el sandbox no sigue symlinks que salen del contenedor.
set -euo pipefail
MODE="${1:-direct}"
HF_SRC="${HF_HOME:-$HOME/.cache/huggingface}"; MODELS_SRC="$HOME/.cache/opentraductor/models"
MOON_SRC="$HOME/Documents/Code/opentraductor/models/moonshine/assets/download.moonshine.ai/model"
[[ -d "$HF_SRC/hub" ]] || { echo "No hay caché de Hugging Face en $HF_SRC" >&2; exit 1; }

case "$MODE" in
  direct)
    SUP="$HOME/Library/Application Support/OpenTraductor"; mkdir -p "$SUP"
    rm -rf "$SUP/hf" "$SUP/models"; ln -s "$HF_SRC" "$SUP/hf"; ln -s "$MODELS_SRC" "$SUP/models"
    [[ -d "$MOON_SRC" ]] && { rm -rf "$SUP/moonshine"; ln -s "$MOON_SRC" "$SUP/moonshine"; }
    echo "enlazado en $SUP" ;;
  --store)
    SUP="$HOME/Library/Containers/com.opentraductor.app/Data/Library/Application Support/OpenTraductor"
    mkdir -p "$SUP"
    # -R conserva los symlinks relativos internos de la caché HF (snapshots → blobs), que quedan dentro del contenedor.
    rm -rf "$SUP/hf"; cp -cR "$HF_SRC" "$SUP/hf"
    [[ -d "$MODELS_SRC" ]] && { rm -rf "$SUP/models"; cp -cR "$MODELS_SRC" "$SUP/models"; }
    [[ -d "$MOON_SRC" ]] && { rm -rf "$SUP/moonshine"; cp -cR "$MOON_SRC" "$SUP/moonshine"; }
    echo "clonado en $SUP" ;;
  *) echo "uso: $0 [--store]" >&2; exit 2 ;;
esac

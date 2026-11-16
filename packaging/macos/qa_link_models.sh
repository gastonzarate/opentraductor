#!/bin/zsh
# SOLO PARA QA / desarrollo: enlaza los modelos que ya tenés descargados (caché de Hugging Face y de opentraductor)
# a la carpeta de datos de la app, para probar el build sin bajar ~5 GB. El usuario final usa la descarga guiada.
set -euo pipefail
SUP="$HOME/Library/Application Support/OpenTraductor"; mkdir -p "$SUP"
HF_SRC="${HF_HOME:-$HOME/.cache/huggingface}"; MODELS_SRC="$HOME/.cache/opentraductor/models"
[[ -d "$HF_SRC/hub" ]] || { echo "No hay caché de Hugging Face en $HF_SRC" >&2; exit 1; }
rm -rf "$SUP/hf" "$SUP/models"; ln -s "$HF_SRC" "$SUP/hf"; ln -s "$MODELS_SRC" "$SUP/models"
[[ -d "$HOME/Documents/Code/opentraductor/models/moonshine/assets/download.moonshine.ai/model" ]] && \
  { rm -rf "$SUP/moonshine"; ln -s "$HOME/Documents/Code/opentraductor/models/moonshine/assets/download.moonshine.ai/model" "$SUP/moonshine"; }
echo "enlazado en $SUP"

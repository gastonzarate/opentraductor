# Historial de perfiles

Registro de configuraciones anteriores que se conservan como evidencia. No son las vigentes: el perfil baseline actual y sus cifras están
en [`uso.md`](uso.md). Texto movido sin cambios desde el README.

**Historial: piloto T2 (superado).** Es una configuración anterior, con endpoint 96 y normalización. El perfil
vigente es C2, con sus cifras en `docs/uso.md`. Se conserva como referencia. Usa los dos extras, `moonshine`
incluido, y requiere los pesos en disco (ver `docs/moonshine.md`):

```bash
uv run --extra runtime --extra moonshine python -m runtime run --asr-backend moonshine-tiny --endpoint-ms 96 --skip-silent-tail --tts-trim-lead-ms 30 --tts-first-chunk-max-words 8 --normalize-asr-text
```

Medición de T2 (tiny, endpoint 96 ms, chunk 8, texto normalizado, recorte 30 ms; 3 clips, 1 repetición): mediana de
344 ms en PCM y 396 ms en audible simulado. No cumple 300 ms. El informe indica contención posible de CPU por
procesos ajenos durante la corrida: `artifacts/runtime-moonshine/T2-tiny-F-chunk8-norm-ep96/report.md`.

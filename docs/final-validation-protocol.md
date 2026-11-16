# Protocolo de validación final (held-out) — congelado, ejecutable, acotado

Alcance: usa el arnés existente (`benchmark run --split test`, `audible-score`, `device_loopback`). El script `scripts/eval/final_validation.py` solo congela, imprime comandos y verifica evidencia; **no ejecuta modelos ni abre audio/transcripciones del held-out** (solo cuenta filas y hablantes del manifest y calcula hashes). Plantilla de config: `docs/final-validation.config.example.json` (con `_example: true` el congelado se rechaza).

## 1. Qué se valida y unidad de análisis
- Held-out: split `test` de `data/manifests/openslr61_es_ar_sample.jsonl` (12 clips, 4 hablantes, 3 por hablante; reservados). Nunca se usa para ajustar. **Exposición parcial reconocida (no es inédito):** ver sección 6; queda registrada en `FREEZE.json → declared.heldout_exposure` y se imprime en `EVIDENCE.md`. Evaluación reservada con esa limitación, sin ajuste posterior.
- **Unidad de análisis: el hablante** (cluster). Las repeticiones técnicas (3 por clip) y los clips de un mismo hablante **no son muestras independientes**. Con 4 clusters no hay precisión fiable para p95/p99; los mínimos 20/100 clusters son una regla práctica del arnés, no una garantía estadística universal: se publican con su aviso de muestra baja.
- Un clip por stream (aislado). **No** se concatenan clips para simular conversación; la validación continua es un congelado y una corrida aparte, y no sustenta afirmaciones sobre conversación natural.

## 2. Qué NO certifica (se imprime siempre en `EVIDENCE.md`)
- **No certifica 300 ms**: los límites de fin de habla están anotados como `estimated` (tolerancia supuesta ±0.3 s, no validada), hay 4 hablantes, las repeticiones no son independientes y es un solo equipo.
- Primer PCM «disponible» (colector), reproducción **simulada** (cola) y salida a **dispositivo** (BlackHole, corrida separada con su propia incertidumbre) son tres métricas distintas y no se intercambian.
- Calidad: las referencias humanas y las generadas se puntúan por separado; hoy `reference_en` no es humana para estos ítems, así que BLEU/chrF contra referencias humanas no está disponible; el texto audible (Whisper `small.en`, solo salida inglesa) mide inteligibilidad/omisiones, no calidad humana. Sin MOS ni afirmación de naturalidad/similitud sin escucha humana.
- «**Offline**» aquí significa **sin red**: el mismo adaptador streaming y la misma config congelada bajo sandbox de macOS con red denegada (`sandbox-exec`, `(deny network*)`) y un canario que debe fallar con EPERM (errno 1) antes de correr; es un intento aparte (`offline-N`). NO es calidad sin latencia ni un adaptador batch. Validación **continua**: reservada y separada.

## 3. Congelado (antes de abrir nada)
`final_validation.py freeze --config <cfg.json> --out artifacts/final-validation/<id>` escribe `FREEZE.json` (inmutable; no se sobrescribe): config completa y su sha256; sha256 del manifest (bytes); conteos del test (ítems, hablantes); sha256 por archivo de `runtime/`, `benchmark/` y `scripts/eval/` (incluye validación y dispositivo) y de `uv.lock`, `pyproject.toml`, `.python-version`; `freeze_id`. Requisitos para congelar:
1. Candidato y config **finales** decididos por desarrollo, sin mirar el held-out. Flags efectivos incluidos (p. ej. `tts_bind_conjunctions=false`).
2. Árbol de código estable: runtime no edita `runtime/` ni yo `benchmark/` desde el congelado hasta el fin de la corrida. `command` y `check` **fallan** si algún hash cambió, se añadió o se eliminó un archivo (incluidos los locks, comparación en ambos sentidos).
3. Pesos: se fingerprintean en `run.json` (`adapter_metadata.weights`) al ejecutar; deben estar cacheados y sin descargas.
4. Hardware: lease exclusivo; el arnés registra carga y contención (`contention_possible`). Una corrida con contención se etiqueta, no se descarta.
5. `scripts/eval/device_loopback.py` ya acepta `--split test` (solo ese script y sus tests; sin tocar `benchmark/` ni `runtime/` durante el piloto de runtime). La salida a dispositivo del held-out es una corrida aparte, registrada como intento.

## 4. Ejecución (una vez por congelado)
`final_validation.py command --freeze <dir>` imprime, sin ejecutar: (1) `benchmark run --split test` con repeticiones/warmup/deadline congelados y export de audio; (2) `audible-score` independiente; (3) nota de dispositivo; (4) la variante «offline»; (5) continua reservada. Reglas:
- Warmup (2 streams) se guarda en `traces.jsonl` y se excluye de estadísticas; cold start (open, primer stream) se informa aparte.
- Fallos, timeouts, anticipados y silencios se cuentan: Kaplan-Meier con censura y el compuesto con penalización, sin descartar ni reintentar para mejorar la cifra. Anticipación se publica junto al lag, sin premiarla.
- Cada intento, también el fallido, se registra en `ATTEMPTS.jsonl` (`attempt --status ok|failed`). Si el código o la config cambian, se hace un **congelado nuevo** y el anterior queda como intento documentado. Prohibido ajustar tras ver resultados del held-out.

## 5. Lista de evidencia (`check`, pass/fail)
Código sin cambios (incluye `scripts/eval/`); `purpose=final_test`, split test; `adapter_config`, repeticiones, warmup y deadline iguales a lo congelado; n de ítems; corrida no abortada; no continuo; streams medidos = ítems × repeticiones (warmup aparte); suma de resultados = N (fallos incluidos); hablantes reportados; p95/p99 marcados como muestra baja; límites declarados imprecisos; métricas separadas (primer PCM / simulada / dispositivo); calidad por procedencia de referencia; `audible.json` vinculado a ESTA corrida (mismos WAV y `stream_id`, ASR independiente); export de audio con el sha256 del índice verificado contra los archivos; `system_name` con el `freeze_id`; la carpeta de la corrida figura en `ATTEMPTS.jsonl` como `ok`; en `offline-*`, canario EPERM. `EVIDENCE.md` termina siempre con el bloque «lo que NO certifica».

## 6. Nota de procedencia
Exposición parcial exacta: una vez, antes de este protocolo, al preparar el piloto interlingüe, se listaron en pantalla las filas del manifest de OpenSLR 61 incluidas las del split test (item_id, duración, hablante y los primeros ~60 caracteres de la transcripción española). No se escuchó audio del test, no se usó ningún resultado de modelos sobre el test y no se ajustó nada con ello; los pilotos posteriores usaron solo desarrollo. No se afirma que el held-out sea inédito.

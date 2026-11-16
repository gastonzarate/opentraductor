# Benchmark opentraductor

Este documento describe el arnés v0 y sus contratos. Ya existen experimentos con modelos reales y mediciones de salida a BlackHole; cada informe declara su alcance y no certifican por sí solos 300 ms. Véase el [desglose provisional por componente](../artifacts/analysis/coordinator-resume-20261115/component-runtime-1-provisional.md).

Desde el 2026-11-15, cada cambio de comportamiento debe evaluarse conjuntamente en calidad de traducción de un audio largo y latencia, según la [política de calidad y latencia](quality-latency-policy.md). El arnés largo está en desarrollo; no se presenta como una prueba ya ejecutada.

## Uso

```
.venv/bin/python -m pytest tests/benchmark          # pruebas de infraestructura
.venv/bin/python -m benchmark selftest --out artifacts/benchmark/selftest   # dummy + reloj virtual (etiquetado)
.venv/bin/python -m benchmark run --manifest data/manifests/X.jsonl --audio-root data/audio \
    --adapter paquete.modulo:Clase --adapter-config '{}' --split development --out artifacts/benchmark/<run>
```
El CLI verifica sha256 del audio por defecto (`--no-verify-sha256` lo desactiva, no recomendado). `--split test` activa `purpose=final_test`: el arnés rechaza usar `test` para desarrollo y rechaza mezclar splits.

## Reglas de medición

- **Reloj**: todo tiempo sale del reloj monotónico del arnés (`benchmark/clock.py`) al recibir el evento. `claimed_ns` del candidato se guarda como dato no confiable y se ignora.
- **Causalidad**: la muestra en la posición `s` ocurre en `t0 + s`. Un chunk `[a, b)` se entrega en `t0 + b`. `end_stream` se llama solo después de cubrir toda la ventana de censura con silencio, para que no sirva de pista de fin de frase.
- **Origen de latencia**: `t0 + utterance_end_s` anotado en el manifest. Nunca el endpoint del candidato ni el fin del VAD.
- **Primer audio, tres instantes que no se mezclan**: (1) *disponible* (titular) = `recv_ns` del arnés del primer chunk TTS con bloque de 10 ms con RMS > −50 dBFS (umbral configurable, queda en `run.json`); (2) *reproducción estimada* = cola simulada con todos los chunks previos, también los silenciosos, más el offset no silencioso; es una estimación; (3) *dispositivo* = solo con eventos `device_output` reales, si no «unavailable» (v0 no lo implementa).
- **Orígenes por modo**: `streaming` mide `end_to_first` desde el fin real anotado. `text` y `offline_file` no tienen fin de habla en tiempo real: miden `request_to_first` desde el inicio de la petición (origen etiquetado `request_start`) y reportan RTF de proceso aparte. `start_to_first` es otra métrica y no se mezcla.
- **Modos y relojes**: `text`, `offline_file`, `streaming`, reloj real/virtual y límites precisos/imprecisos se agregan por separado.
- **Clases estadísticas** (cada fragmento en una sola): `observed`; `timeout` (censura por derecha real, con su deadline); `failure` (error, cuelgue, silencio: sin latencia, no es censura); `anticipated` (primer PCM antes del fin anotado, contenido sin verificar: no se penaliza ni premia en la distribución observada y se publica junto al lag); `unmeasurable` (sin fin anotado o sin correspondencia: nunca se vuelve censura ni fallo).
- **Cuantiles**: (a) *latencia observada con censura*: Kaplan-Meier sobre observed+timeout, válido con deadlines heterogéneos; cuantil en zona censurada = «no identificable». (b) *compuesto tiempo-hasta-salida-válida*: fallo/timeout/anticipado = +inf; **no es latencia observada**; solo se calcula si ningún deadline es menor que algún valor observado, si no «no identificable». IC: bootstrap por cluster (hablante si se conoce; si no fuente/grabación; nunca el stream) con semilla. `low_sample` usa clusters independientes: repetir 10 clips 10 veces no habilita p99. Los IC de Wilson de las tasas suponen independencia. Se validan longitudes, `q∈(0,1]`, `n_boot>0`, `ci_level`, finitud.
- **Correspondencia fragmento→salida**: en modo aislado (un fragmento por stream) es por construcción. En modo continuo cada chunk TTS debe traer `source_span_s`; sin él el resultado es `no_correspondence`.
- **Backlog vs. lag semántico**: backlog es duración de audio encolado en una reproducción simulada. El lag semántico (inicio de reproducción − momento en que ocurrió el fin del tramo fuente) solo se calcula con `source_span_s`; si no, «unavailable». Es lag **por segmento/chunk** con el span que declara el adaptador (si el span es interpolado, el valor lo hereda): no es lag palabra a palabra y no verifica contenido. Un lag < 0 se cuenta aparte (`n_negative_unverified_anticipation`) como anticipación no verificada, no como lag bueno. La tendencia (`trend_s_per_s`) usa un punto por span fuente distinto (lag del primer chunk de cada segmento frente al fin del span en el reloj de la fuente) y solo se calcula con ≥ 5 spans distintos (`trend_min_n`, `n_distinct_source_spans`); si no, `null` con `trend_note`. Así una pendiente entre chunks de un mismo segmento (mismo span, reloj de recepción: velocidad de emisión del TTS) nunca se presenta como crecimiento de la conversación. Con n=3 se había visto ~23 s/s, que era ruido. El informe lo rotula «lag por segmento».
- **Cold start / calentamiento / repeticiones**: `open()` se mide aparte; los streams de calentamiento se guardan en `traces.jsonl` (`warmup: true`) y se excluyen de las estadísticas; el orden de ítems por repetición es determinista por semilla.
- **Robustez**: toda llamada al adaptador tiene límite (`call_timeout_s`); un cuelgue queda en `trace.error` (`AdapterHang`), cuenta como fallo y tras `abort_after_hangs` cuelgues consecutivos la corrida se aborta, con los streams omitidos contados como fallo (`meta.aborted`). `open()` fallido o colgado produce `FAILURE.json` y `RunError` (el CLI sale con código 2). La inferencia síncrona que bloquea el event loop no se puede interrumpir con `wait_for`: con reloj real un watchdog (hilo) termina el proceso con código 70 tras `watchdog_stall_s` y escribe `FAILURE.json` con contexto y resultados parciales. Adaptador y colector se cierran siempre (`finally`). `start_stream()` se mide en cold start (carga diferida).
- **Calidad**: se ejecuta dentro de cada corrida y va en `run.json`/`report.md`. Se puntúa el texto planeado y, con `--audible-asr modulo:funcion`, el texto audible (ASR independiente del candidato; tiene sesgo propio: errores del ASR se confunden con errores del TTS). Referencias humanas y generadas se puntúan por separado. BLEU/chrF (sacrebleu, con firma) sobre referencias disponibles; ítems sin referencia o sin hipótesis se cuentan y se informan. Métrica neural y texto audible quedan «unavailable» sin modelo/ASR independiente configurado. Texto planeado y texto audible se puntúan por separado. Ningún proxy es MOS humano. No se compara F0 palabra a palabra entre idiomas.
- **Escucha ciega**: `benchmark/listening.py` genera `ratings_template.csv` (IDs ciegos, orden distinto por evaluador) y `blind_key.json` aparte. Dimensiones: naturalidad, inteligibilidad, acento, expresividad, similitud de voz.
- **Privacidad**: las trazas guardan features de PCM, no audio. Artefactos y audio público quedan separados de la base SQLite de conversaciones.
- **Hardware**: `run.json` guarda CPU, versiones, hash del código del benchmark (sin Git), hash del manifest, configuración y carga del sistema al inicio y fin. Las corridas de rendimiento deben ser exclusivas (sin otras cargas de modelos).

## Contrato

Manifest JSONL (`benchmark/contracts.py`): campos obligatorios `item_id, recording_id, source_id, split(development|test), audio_path, sha256, sample_rate, channels, duration_s, dialect(es-AR|other|unknown), license, license_url, source_url, attribution, transcript_es_provenance, reference_en_provenance(human|generated|unknown), boundary_method`. Opcionales/nulables: `speaker_id`, `transcript_es`, `reference_en`, `utterance_start_s`, `utterance_end_s`, `boundary_uncertainty_s`, `consent_note`. Límites no humanos exigen `boundary_uncertainty_s`; `forced_alignment_auto` y `estimated` no cuentan como «precisos». Split aislado por hablante y por grabación; hablante desconocido se agrupa por `source_id`. Audio exigido: WAV PCM16 mono 16 kHz (no hay remuestreo implícito).

Adaptador (`Adapter`): `open(config)`, `start_stream`, `push_audio`, `end_stream`, `process_file`, `process_text`, `events()`, `close()`. Eventos: `asr_partial, asr_final, endpoint, mt_partial, mt_final, tts_chunk(pcm, planned_text_en, source_span_s, commit_id), error`.

**Commits/frases (`commit_id`)**: un commit es una frase emitida. `asr_final`, `mt_final` y todos los `tts_chunk` de esa frase comparten `commit_id`. El texto planeado de un fragmento es la concatenación ordenada (por llegada) de un texto por commit: los chunks de un commit no se duplican y dos commits con el mismo texto cuentan ambos (frase repetida legítima). Sin `commit_id`, chunks consecutivos con igual texto se funden y el resultado se marca `planned_text_ambiguous` (una repetición pegada no se distingue). En streams continuos cada commit se atribuye a un fragmento por `source_span_s`; `asr_final` sin span en continuo = `asr_status: no_correspondence`.

## Limitaciones conocidas de v0

- Sin corpus real ni candidato real: nada medido.
- `device_output` (salida a dispositivo) y ASR independiente para texto audible no están implementados; reportan «unavailable».
- Texto audible en streams continuos no implementado. No se provee un ASR independiente por defecto; sin él el texto audible es «unavailable».
- Anticipación: el contenido de una salida anticipada aún no se verifica automáticamente; se publica conteo y valor con signo.
- `asyncio.sleep` real puede dormir de más; la tardanza de envío se guarda (`sends_lateness_ms_max`) pero no invalida la corrida automáticamente.
- Velocidad: `offline_throughput` (RTF y × tiempo real) solo en text/offline_file; en streaming es "unavailable" porque wall/duración-de-replay no mide cómputo. `tts_emission_rtf` (intervalo de chunks/duración emitida; depende del pacing del candidato) y backlog de reproducción van separados.

## Export de audio, escucha ciega, texto audible y proxies

Todo opera solo sobre salidas sintéticas de candidatos a partir de audio público del benchmark. Nunca sobre conversaciones.

- **Export**: `run --export-audio-dir DIR --system-name NOMBRE` escribe un WAV PCM16 mono nativo por stream medido (`NOMBRE__<stream_id>.wav`) y `export_index.jsonl` (sistema, ítems, resultado, sha256, duración, `roundtrip_ok`, `scope`). Cada WAV se relee y compara al escribirlo; las salidas totalmente silenciosas no se exportan. Streams continuos se exportan pero no se usan como estímulos por ítem.
- **Escucha ciega**: `listening-prepare --index IDX... --raters R... --out DIR --seed N` verifica los índices (rechaza WAV ausentes o corruptos), copia los WAV con nombre ciego `S####.wav`, genera `ratings_template.csv` (orden distinto por evaluador) y `blind_key.json` aparte. Avisa si hay audio del adaptador dummy. `listening-summary --ratings CSV --key JSON` resume por sistema y dimensión (n, media, sd; celdas vacías/ inválidas contadas). Es puntuación humana 1-5, no MOS formal (ITU-T P.800) sin panel y diseño adecuados.
- **Texto audible**: `--audible-asr faster-whisper:<modelo>` usa un ASR local (faster-whisper, CPU int8, para no tocar la GPU del candidato; MIT) o `modulo:funcion`. Etiqueta y parámetros quedan en `run.json` (`audible_asr_label`). Whisper puede alucinar o «corregir» habla poco clara y con audio no inglés produce texto inventado: es evidencia de inteligibilidad, no verdad. Los pesos se descargan de Hugging Face la primera vez.
- **Proxy de naturalidad (UTMOSv2)**: `proxy-score --index IDX --checkpoint PESOS.pth [--fold --seed --num-repetitions --predict-dataset] --out JSON`. Código y pesos MIT (LICENSE del repo y ficha HF `license: mit` verificados; el modelo además descarga `facebook/wav2vec2-base`). Los pesos se descargan explícitamente (el instalador usa `wget`, que no existe en macOS por defecto): `curl -L -o fold0_s42_best_model.pth https://huggingface.co/sarulab-speech/UTMOSv2/resolve/main/fold0_s42_best_model.pth` (≈780 MB, sha256 registrado en cada resultado). El resultado se etiqueta «predictor automático, NO MOS humano»; su validez para TTS es→en y voces clonadas no está demostrada. Sin dependencia o pesos: `PluginUnavailable`, código de salida 3, sin puntaje.
- **Dependencias**: UTMOSv2 (`git+https://github.com/sarulab-speech/UTMOSv2.git`, arrastra torch/timm/transformers) y `faster-whisper` son opcionales y no están en el `pyproject` raíz. Validados en un venv aparte (`~/.cache/utmos-venv`, Python 3.12). ECAPA (similitud de voz) no está implementado.

## ASR del candidato (español) y escucha con referencia

- **WER/CER del ASR es** (`quality.candidate_asr_es`): `asr_final` ordenados por commit contra la transcripción de referencia, nivel corpus (errores totales / palabras de referencia), por procedencia de la transcripción (human/generated/unknown), separado de BLEU/chrF de la traducción. Normalización: minúsculas, sin puntuación (acentos se conservan). Hipótesis ausente = vacía (penaliza). Sin ningún `asr_final` emitido (p. ej. modelo end-to-end) = «unavailable».
- **Límites estimados**: el informe marca los grupos con límites no precisos con «tolerancia SUPUESTA, no cota validada ni IC; solo diagnóstico, no sirve para afirmar 300 ms».
- **Escucha con referencia** (`listening-prepare --manifest M --audio-root R`): cada estímulo `S####.wav` va con el audio fuente público del ítem `R####.wav` (columna `reference_id`; mismo ítem comparte referencia entre sistemas). `NA` por dimensión: `similitud_voz` solo se pide a sistemas declarados `--system-meta NOMBRE:voice_cloning=true` (voz fija = NA, no fallo de clonación); `expresividad` y `similitud_voz` son NA sin audio de referencia. `INSTRUCCIONES.md` explica escala, orden aleatorio por evaluador y uso de NA; el acento se juzga solo en la salida, no se atribuye acento a una fuente de dialecto desconocido. `blind_key.json` va aparte.
- **Controles de rango** (no sustituyen anclas P.800, ni vuelven MOS a un proxy): `--human-control WAV "LICENCIA"` (grabación humana EN con licencia) y `--degraded-controls N` (copias del mismo audio con paso bajo 3.4 kHz + ruido a 15 dB, determinista). Van en `controls_index.jsonl`, que `proxy-score --index A B controls_index.jsonl` puede puntuar para ver cómo se ubica UTMOSv2 frente al control humano y al degradado. `listening-summary` los separa en `controls`.

## Entorno de evaluación aislado (UTMOSv2 + faster-whisper)

No toca `.venv` del runtime ni depende de `/tmp`. Todo vive en `models/eval/` (ignorado por el proyecto).

```
scripts/eval/setup.sh      # crea models/eval/venv (Python 3.12), instala scripts/eval/requirements-eval.txt,
                           # descarga pesos UTMOSv2 con sha256 y fija revisiones HF (scripts/eval/weights.json)
scripts/eval/run.sh proxy-score --index OUT/export_index.jsonl OUT/controls_index.jsonl \
    --checkpoint models/eval/utmosv2/fold0_s42_best_model.pth --out OUT/proxy.json
scripts/eval/run.sh run --manifest ... --adapter ... --audible-asr faster-whisper:small.en --out ...
```

`run.sh` ejecuta el CLI en ese venv, sin red (`HF_HUB_OFFLINE=1`) y con `HF_HOME=models/eval/hf`; `setup.sh` apunta `refs/main` de cada modelo a la revisión fijada para que la carga offline resuelva exactamente a ella. Fijado en `scripts/eval/weights.json`: pesos UTMOSv2 (`fold0_s42`, sha256 `c8149d98…`), código UTMOSv2 (commit `cc2700db…`, MIT), `facebook/wav2vec2-base@0b5b8e86…` (Apache-2.0), `timm/tf_efficientnetv2_s.in21k_ft_in1k@ea9abc14…` (Apache-2.0), `Systran/faster-whisper-small.en@d1d751a5…` y `tiny.en@0d3d19a3…` (MIT). Cambiar un valor cambia la medición. No se usa un grupo `eval` en `pyproject` porque torch/transformers pesan y pueden chocar con el runtime.

## Reproducibilidad: qué registra `run.json`

- `environment.benchmark_code_sha256`, `environment.project.runtime_code_sha256` (código del runtime/candidato: todo directorio de primer nivel salvo datos/artefactos/modelos/tests/docs/benchmark, con hash por archivo si caben), `lock_files` (sha256 de `uv.lock`, `pyproject.toml`, `.python-version`).
- `environment.all_packages` (entorno completo) y `environment.key_packages` (mlx, mlx-audio, parakeet-mlx, ctranslate2, torch, transformers, misaki, huggingface-hub, faster-whisper, utmosv2, timm...; `null` = no instalado).
- `adapter_metadata` = `Adapter.describe()`: **cada adaptador debe declarar** revisión y hash de pesos (`snapshot.weights_fingerprint(ruta)`), versiones de motores, hash/commit del código del adaptador y parámetros de decodificación. Sin `describe()` la corrida avisa en `adapter_metadata_warning` que pesos y versiones del candidato no quedan registrados. `version = "0"` no basta.
- `weights_fingerprint` no relee modelos enormes: en cachés de Hugging Face un blob LFS se llama como su sha256 (se usa tal cual), y el resto se cachea por (ruta, tamaño, mtime) en `artifacts/hash-cache.json` (`OPENTRADUCTOR_HASH_CACHE` lo cambia).

Nota UTMOSv2: elige tramos de audio al azar; el plugin fija las semillas (python/numpy/torch = `--seed`) antes de cada archivo, por lo que el puntaje del mismo WAV es reproducible (antes variaba ~0.05 entre corridas). Con `--num-repetitions` > 1 se promedian tramos.

## Auditoría de límites (estimate_boundaries.py v2)

Hallazgo (revisión independiente, sin usar el VAD del candidato como verdad): la v1 usaba umbral relativo `P10 + 10 dB`. En grabaciones con cola casi digital (P10 ≈ −64…−74 dBFS) el umbral caía a −54…−64 dBFS y contaba dither/ruido de cola como habla: FLEURS 1516 terminaba en 11.26 s (real ≈ 9.6 s) y 1593 en 10.77 s (≈ 9.57 s). En FLEURS 1549 un ruido aislado a ~10 s (tras ~0.6 s de silencio) extendía el fin a 10.18 s (último tramo sostenido de habla: 9.23 s). OpenSLR sufría lo mismo (p. ej. fin 5.02 s vs 3.53 s con caída clara de nivel). Los inicios de la v1 (0.3–0.4 s) eran ruido de fondo: la habla de FLEURS empieza entre 1.1 y 2.4 s en la mayoría de clips.

Corrección (v2): piso absoluto −50 dBFS; cierre (huecos ≤ 150 ms) y apertura (tramos ≥ 120 ms) para ignorar clics y respiraciones; descarte de tramos de borde < 0.5 s aislados por huecos > 0.5 s (registrados en `edge_runs_discarded`); verificación cruzada con un segundo método en la banda 300–3400 Hz (si inicio o fin difieren > 0.3 s, `unknown`); habla que toca el final del archivo (< 0.15 s) = `unknown`; límite de inicio tardío 3 s; plausibilidad caracteres/s 6–28 (lectura rápida es-AR observada en desarrollo: 19–23). Resultado: FLEURS 1516 → 2.36–9.59 s, 1549 → 1.14–9.23 s (coincide con el VAD de Runtime a ~0.07 s, aunque no se usa como referencia), 1553 = `unknown` (habla hasta el final del archivo, ruido ≈ −50 dBFS); 9 de 10 con límites; OpenSLR 24/24.

Procedencia: cada item cambiado conserva su estimación anterior en `boundary_history` (con `estimator_version`) y `boundary_estimator.version = 2`; el metadata JSON guarda antes/después y la validación por item. No se tocaron audios, transcripciones ni referencias (verificado campo a campo). Las tolerancias (±0.3 s) siguen siendo SUPUESTAS, no validadas: los límites sirven para diagnóstico de latencia, nunca para afirmar 300 ms. Los parámetros se fijaron mirando solo clips de desarrollo; el split de test se procesó con los parámetros ya congelados y no se usó para ajustarlos. Limitación: el descarte de tramos de borde también eliminaría una palabra real aislada al principio o al final (queda registrado).

## Controles de voz humana EN

`scripts/datasets/extract_voice_controls.py` (`uv run --with pyarrow python scripts/datasets/extract_voice_controls.py --n 4`) extrae 4 grabaciones humanas en inglés de FLEURS en_us/validation (CC-BY-4.0) desde el parquet cacheado, con los mismos ids que la muestra es_419, y escribe `data/audio/controls/fleurs_en_us_<id>.wav` y `data/manifests/voice_controls.jsonl` (hash, licencia, atribución, género del hablante, procedencia del parquet y del audio fuente). Son anclas de rango, no ítems de evaluación ni voz de clonación. Uso: `listening-prepare ... --human-control data/audio/controls/fleurs_en_us_1516.wav "CC-BY-4.0 (Conneau et al. 2022, FLEURS en_us)"`.

## Evaluación solo texto (MT / TTS) para el set de reuniones

`benchmark/adapters/text_mt_tts.py` (`TextMtTtsAdapter`, modo `text`) reutiliza `runtime.components.MarianCT2` y `KokoroTTS` sin modificarlos (import perezoso: el benchmark no depende del runtime). MT y TTS corren en hilos dedicados (MLX liga streams al hilo que los crea); modo offline forzado; sin descargas.

```
.venv/bin/python -m benchmark run --text-manifest data/manifests/text_meetings_ar_dev.jsonl \
    --adapter benchmark.adapters.text_mt_tts:TextMtTtsAdapter --adapter-config '{"stages": "mt"}' \
    --split development --repetitions 3 --warmup 2 --out artifacts/benchmark/text-mt
# '{"stages": "mt+tts", "voice": "af_heart"}' añade Kokoro; --export-audio-dir exporta los WAV (24 kHz)
```

- **Latencia**: petición → primer output, en grupos separados `output_kind=text` (primer `mt_final`, sin audio) y `output_kind=audio` (primer PCM no silencioso). Nunca se mezcla con streaming ni entre sí. Los 54 ítems son texto: no hay `end_to_first`.
- **Calidad**: BLEU/chrF de `mt_text` (lo que dice la MT) y de `planned_text` (lo que se planea sintetizar), por procedencia (las referencias del set son `generated`, no humanas). Cada cláusula TTS es un commit: chunks de una cláusula no se duplican y una cláusula repetida cuenta dos veces.
- **Chequeos de superficie** (`quality.text_checks`, por fenómeno): números (secuencia de dígitos), negaciones (token de negación en origen y salida) y términos técnicos (lista editable en `benchmark/textchecks.py`) con ejemplos marcados. **Son señales para revisión humana, no exactitud semántica**: no detectan decimales mal colocados, números en palabras, negaciones sin token ni traducciones legítimas de términos.
- **Reproducibilidad**: `describe()` registra revisiones del Marian/Kokoro del runtime, huella de pesos (`weights_fingerprint`), sha256 de `runtime/components.py` y del adaptador, versiones de ctranslate2/mlx/mlx-audio/misaki.
- Limitación de pruebas: el reloj virtual no espera hilos reales; las pruebas usan trabajadores en línea y una prueba separada con reloj real y hilos. Los modelos reales todavía no se ejecutaron (requiere ventana de cómputo coordinada con Runtime).

## Contención del sistema

La exclusividad es solo del equipo opentraductor: otros proyectos pueden consumir CPU/GPU y no se detienen. `run.json` registra al inicio y al final `loadavg`, núcleos, procesos ajenos con ≥ 50 % de CPU (solo nombre y %) y `contention_possible`. Si hubo contención, `meta.run_status_label` y el informe dicen «PILOTO con contención posible; no es comparación final». Sin contención observada tampoco se afirma máquina en reposo ni GPU exclusiva (no se mide uso de GPU). La comparación final requiere una ventana de carga estable con app y configuraciones congeladas.

## Escucha ciega en HTML local (`benchmark/listening_ui.py`)

Reproducible, sin servidor ni red ni dependencias nuevas:

```
.venv/bin/python -m benchmark listening-ui --package artifacts/benchmark/listening-pilot1 --out artifacts/benchmark/listening-pilot1-html
```

- Entrada: paquete de `listening-prepare` (`stimuli/`, `blind_key.json`). Salida en carpeta **aparte** (se rechaza una salida dentro del paquete): `index.html`, `stimuli/` (copia con rutas relativas `stimuli/S####.wav`, `R####.wav`), `INSTRUCCIONES.md`, `listening_ui_manifest.json` (sha256 de cada WAV). Se puede entregar esa carpeta sola (doble clic en `index.html` o servirla como estática).
- Ciego: la salida no contiene nombres de sistema/modelo, ids de ítem, nombres de archivo originales, textos de referencia ni `blind_key.json`. `build` busca esas cadenas en todos los archivos (también en los bytes de los WAV; en WAV solo cadenas de 8+ caracteres, para evitar falsos positivos) y borra la salida si hay fuga. El HTML sí lleva qué dimensiones son `NA` fijas por estímulo y qué `R####` acompaña a cada uno, igual que la plantilla CSV. Límite: la duración o el timbre de un audio puede delatar un control (p. ej. el degradado); no se oculta.
- Aleatorización por evaluador: el navegador baraja con `mulberry32(fnv1a("semilla:evaluador"))` (semilla = la del paquete, `--seed` la cambia). `listening_ui.rater_order` es la misma función en Python; un test la compara con node. Mismo nombre ⇒ mismo orden y se retoma lo guardado en `localStorage`; nombres distintos ⇒ órdenes distintos.
- Evaluación: naturalidad, inteligibilidad, acento, expresividad y similitud de voz, con botones de radio 1-5 y `NA` (no sliders: un slider tiene valor inicial y se podría confundir «sin tocar» con una puntuación; los radios son además accesibles con teclado y lector de pantalla). Nada está puntuado de antemano; `NA` solo donde `blind_key` lo fija (`similitud_voz` con voz fija, `expresividad`/`similitud_voz` sin referencia) o si el evaluador lo elige. Sin referencia española no se pide expresividad; con referencia, se muestra primero.
- Accesibilidad: `lang="es"`, enlace para saltar, `fieldset/legend`, foco visible, contraste y modo oscuro, `aria-live` para el estado, controles nativos de audio. No se probó con un lector de pantalla real ni en un navegador gráfico en este entorno (sin Chrome ni jsdom); sí `node --check` del script y equivalencia del barajado con Python.
- Exportación: `Exportar JSON` (`opentraductor-listening-ratings/v1`: `rater`, `ratings[]` con `stimulus_id`, `reference_id`, cinco dimensiones como número, `"NA"` o `""`, `comentario`, `escuchas_completas`) y `Exportar CSV` (mismas columnas que `ratings_template.csv`). Exportar incompleto avisa y deja las celdas vacías: no se rellenan. `listening-summary --ratings A.json B.json C.csv --key blind_key.json` (y `listening.summarize_ratings`) lee ambos formatos, uno o varios evaluadores; rechaza esquema desconocido y `stimulus_id` que no esté en la clave. Sigue siendo puntuación subjetiva 1-5, no MOS formal (ITU-T P.800).
- Privacidad: las puntuaciones solo viven en el `localStorage` del navegador hasta exportar; no hay ninguna petición de red.

## Alternativa MT a Marian opus-mt-es-en (solo documental; nada instalado ni ejecutado)

Problema medido (piloto, 54 casos contra referencias *generadas*): omisión de «No veo el gráfico», `15:30`→`3:30`, voseo imperativo perdido. Marian actual: `Helsinki-NLP/opus-mt-es-en`, licencia Apache-2.0 según su tarjeta HF, `pytorch_model.bin` de 312 MB (fp32), 34 ms p50 medido.

**Recomendación: MADLAD-400-3B-MT** (`google/madlad400-3b-mt`).
- Licencia: Apache-2.0 (tarjeta HF y metadatos de la API de HF). Uso comercial permitido.
- Tamaño: 2 940 374 016 parámetros (API de HF, `safetensors.total`); T5 codificador-decodificador entrenado solo para traducción, entrada `<2en> texto`. La tarjeta ofrece una versión cuantizada de 1.65 GB (candle, GGUF); existen conversiones CTranslate2 int8 de terceros (p. ej. `santhosh/madlad400-3b-ct2`, `SoybeanMilk/madlad400-3b-mt-ct2-int8_float16`; la licencia de esas copias no está verificada: convertir uno mismo desde el original con `ct2-transformers-converter`).
- Evidencia primaria: Kudugunta et al. 2023, arXiv:2309.04662 (resumen: modelo MT «competitivo con modelos significativamente más grandes»; dominios reportados). No leí las tablas es→en ni hay evidencia publicada sobre voseo, horas o negación: **la mejora en esos tres puntos es una hipótesis a medir, no un hecho**. Es traducción a nivel de documento/párrafo (el corpus se llama «Document-Level»), lo que favorece no perder oraciones, pero también es una hipótesis.
- Costo: ~38 veces más parámetros que Marian. Esperar latencia mucho mayor que los 34 ms; solo una medición dirá si cabe en el presupuesto (el objetivo de <300 ms al primer audio incluye ASR y TTS). Si no cabe, sirve como MT de referencia de calidad para medir cuánto de la falla es del modelo.

Descartadas con motivo documentado: `opus-mt-tc-big-cat_oci_spa-en` (CC-BY-4.0, 577 MB fp32) reporta spa-eng news-test2008 27.9 BLEU, idéntico a opus-mt-es-en (27.9): sin mejora documentada para es→en. NLLB-200 (CC-BY-NC) no es comercial. Qwen3-1.7B/4B (Apache-2.0, 2.03 B/4.02 B parámetros según HF) son LLM generales: más riesgo de omitir o parafrasear y de latencia de prefill; segunda opción si MADLAD no cabe.

**Antes de cambiar de modelo, probar sin costo de modelo** (hipótesis): dividir en oraciones antes de Marian (la omisión de la 2.ª oración es compatible con entrada multi-oración fuera de lo que Marian vio al entrenar) y proteger horas/números (placeholder) o verificar con `textchecks`. Son cambios de runtime, no de benchmark: se proponen al coordinador.

Plan de medición cuando haya ventana de cómputo: mismos 54 casos, MADLAD-3B (CT2 int8, CPU y GPU si aplica) con `benchmark run` texto, reportando p50/p95/p99 con tamaño de muestra, BLEU/chrF, y las señales de `textchecks` (números, negaciones, omisiones). Las referencias del set siguen siendo `generated`, no humanas.

## Adaptador experimental Qwen3-TTS con clonación (`benchmark/adapters/text_mt_qwen3_clone.py`)

Solo texto, aislado: MT Marian + TTS `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit@50f45ef0` (solo desde disco). No cambia el default del runtime ni toca `runtime/`. **Código y pruebas con fakes únicamente; el modelo no se ha ejecutado** (a la espera de una ventana de cómputo). No hay ninguna afirmación de velocidad ni de calidad.

```
python -m benchmark run --text-manifest data/manifests/text_meetings_ar_dev.jsonl \
  --adapter benchmark.adapters.text_mt_qwen3_clone:TextMtQwen3CloneAdapter \
  --adapter-config '{"streaming_interval": 0.5, "ref_text": null}' --split development --out artifacts/benchmark/<carpeta>
```

- Referencia: `data/audio/examples/kyutai-0a67.wav` + `.json` (CC0, voluntario, no es voz del usuario). Se verifica sha256 y que el JSON declare `license` y `consent_source`; si no, falla.
- `ref_text` configurable. Con texto: clonación ICL de la API (audio + transcripción). Con `null` (hoy no hay transcripción verificada): la API instalada no usa ICL y cae a la ruta Base con embedding de hablante; es otro modo. `describe()` registra `clone_mode` (`icl` | `speaker_embedding_only`) y no deben mezclarse en el análisis.
- API instalada (`mlx_audio/tts/models/qwen3_tts`, `Model.generate`): se usan `ref_audio`, `ref_text`, `lang_code`, `stream=True`, `streaming_interval` (s; el modelo lo traduce a `int(intervalo*12.5)` tokens por chunk), `streaming_context_size`, y opcionales `temperature/top_k/top_p/max_tokens`. Un test compara los kwargs con la firma instalada. `seed` fija `mx.random.seed` antes de cada síntesis.
- `split_clauses` (False por defecto): una síntesis por enunciado; True repite el prefill de la referencia por cláusula.
- Medición: primer PCM disponible, primer no silencioso y producción continua (chunks, RTF de emisión, backlog) salen de las métricas existentes del modo texto+audio. Qué se mide con `streaming_interval` pequeño (sobrecoste por chunk, calidad en bordes) queda por medir.
- `describe()` guarda revisión, huellas de pesos, referencia (hash, licencia), `clone_mode` y parámetros de generación.
- Pendiente/limitaciones: sin transcripción de referencia verificada no se prueba ICL; el idioma/acento de la referencia no está verificado.

## Escucha con muestra de voz y avisos de pares (`--voice-reference`)

`listening-prepare --voice-reference ITEM_ID=WAV` (repetible) adjunta una **muestra de voz** (p. ej. la CC0 `kyutai-0a67`) como `R####.wav` solo para `similitud_voz`; `expresividad` pasa a `NA` (la muestra no es el original del ítem) y `INSTRUCCIONES.md` lo explica sin afirmar idioma ni acento de la muestra. Si los sistemas no tienen los mismos textos, el resultado trae `warning_unpaired` (sistema y texto confundidos: sin comparación entre sistemas). `listening-ui` usa el `INSTRUCCIONES.md` del paquete.

`scripts/eval/leading_quiet.py` (offline, solo numpy) explica la brecha entre primer PCM «disponible» y «reproducción estimada» con la quietud inicial del audio (ver `artifacts/benchmark/clone-pilot1/RESULTS.md`).

## Salida a BlackHole medida por el arnés (`benchmark/device_loopback.py`, solo código; sin ejecutar)

Verifica pipeline → `Player` real → BlackHole 2ch → captura del arnés. No es audio escuchado en Meet/Discord, no usa micrófono físico ni acústica. Se mide solo métricas por bloque; no se graba audio.

- `benchmark.adapters.device_loopback:DeviceLoopbackAdapter` envuelve un adaptador STREAMING (`inner`, por defecto `runtime.adapters.baseline:BaselineAdapter`) sin alterar sus eventos: el replay causal a 20 ms, el colector y las métricas son los existentes. Escribe cada `tts_chunk` al `Player` y anota el instante con el reloj del arnés.
- Dispositivos: solo uno cuyo nombre contenga «blackhole» con entrada y salida; si hay otro o varios, falla (nunca el dispositivo por defecto, ni micrófono ni altavoces).
- Reloj: `time.monotonic_ns` (el de `RealClock`) leído dentro del callback de captura. Inicio estimado del bloque = callback − (`currentTime` − `inputBufferAdcTime`). Se publican bloque (10 ms), jitter p95 de callbacks medido en el silencio previo, latencia de entrada mediana e incertidumbre total = bloque + jitter. El evento `device_output` NO lo emite el candidato; la métrica de dispositivo del informe estándar sigue «unavailable» y este informe es aparte (`device_loopback.json` + `device_loopback_summary.json`).
- Silencio y contaminación: silencio previo obligatorio (`pre_silence_s`, umbral −50 dBFS); si hay audio, `ContaminationError` y se aborta antes de medir. Entre clips (un clip a la vez) se exige cola del Player vacía y captura en silencio; audio fuera de las ventanas de clip aborta; audio antes de la primera escritura audible ⇒ `contaminated_before_write`; energía capturada ≫ escrita ⇒ `contamination_suspected_energy`. Solo clips `ok` (inequívocos) tienen latencia, asociada al `commit_id` del primer chunk audible. Audio ajeno superpuesto con el propio solo se sospecha por energía, no se puede excluir.
- Salida del resumen: `write_to_device_ms`, `available_to_device_ms` (vs. primer PCM disponible del colector) y `end_to_device_ms` (vs. fin anotado del manifest; con `boundary_method` estimado hereda esa incertidumbre).
- Limpieza: streams de captura y Player se paran siempre (también si falla `open` o `calibrate`); timeouts de drenaje (`drain_timeout_s`).
- Pruebas con fakes (`tests/benchmark/test_device_loopback.py`): análisis, reloj/latencia de callback, selección de dispositivo, contaminación, cierre.
- Comando exacto: `scripts/eval/device_loopback.py command --out artifacts/benchmark/device-loopback-1` (imprime; solo escribe el subconjunto de manifest); `run` ejecuta (requiere lease y BlackHole cargado); `merge` re-une offline.

**Hallazgos de la primera ejecución real (BlackHole 2ch, este equipo):** (1) Iniciar la captura de BlackHole antes o a menos de ~10 ms de arrancar el `OutputStream` del `Player` deja la captura muerta tras ~17 bloques (AUHAL `-10863`/`-50`, sin excepción). Orden obligatorio: `Player` primero, ≥0.2 s de separación (`player_to_capture_settle_s`=0.3), captura después. (2) Una captura detenida no es silencio: `wait_quiet` falla si no llegan bloques y un hueco de callbacks > 100 ms invalida la medición. (3) `merge_with_traces` exige un solo segmento fuente y el mismo `commit_id`; publica el callback BRUTO y el onset estimado por ADC (la corrección varió 16–57 ms entre clips). (4) La validez (`validity`) no deja pasar como ok estados de captura, descartes del Player, chunks tardíos ni huecos de captura; los `underruns` solo avisan.

## Piloto interlingüe ES-AR → EN (preparado; sin ejecutar)

Corrige el piloto de clonación anterior (muestra de voz CC0 en inglés: no valida transferencia ES→EN; queda como control histórico). Diseño: 3 clips OpenSLR 61 de **desarrollo** (nunca held-out) con audio original en español argentino humano y su transcripción humana (`docs/crosslingual-voice-pilot.json`). Evalúa MT + TTS/clonación con transcripción humana; no es ASR ni latencia extremo a extremo; no promete transferencia emocional (se puntúa solo tras escucha).
- `scripts/eval/crosslingual_manifest.py --out artifacts/benchmark/crosslingual-pilot/text_manifest.jsonl`: manifest de texto (verifica sha256, humano, development).
- Qwen3 clon con condicionamiento por ítem: `--adapter-config '{"conditioning_file":"docs/crosslingual-voice-pilot.json","streaming_interval":0.2,"seed":1234,"max_tokens":150}'`. Cada `text_es` se asocia por igualdad exacta (único) con SU audio original y su transcripción humana (`clone_mode=icl_per_item`); sin coincidencia falla, nunca usa otra voz. Que ICL con transcripción en español y texto objetivo en inglés funcione bien NO está verificado.
- Comparación: Kokoro (voz fija) y Qwen 0.2 s con el mismo texto de entrada, 1 warmup + 1 rep, mismas MT.
- Escucha: `listening-prepare --manifest data/manifests/openslr61_es_ar_sample.jsonl --audio-root data/audio --note artifacts/benchmark/crosslingual-pilot/note.md ...` → `R####.wav` es el original español del ítem (`source_utterance`); Kokoro `similitud_voz`=NA, `expresividad` y similitud se piden frente al original.

## Player `latency` low vs default (receta preparada; sin ejecutar)

- `device_loopback` acepta `player_latency` (`None|"low"|"high"`, CLI `--player-latency default|low|high`) solo para el `OutputStream` del Player; la captura, el orden Player→captura y el detector no cambian entre brazos. El informe registra `requested_latency_mode`, `reported_stream_latency_s` (lo que DECLARA PortAudio al abrir; no es una medición física) y `blocksize` (0 = variable). Los bloques parcialmente rellenados no invalidan.
- **Paso 1, sin modelos** (`scripts/eval/player_latency_probe.py`, `benchmark/player_probe.py`): ráfagas de tono sintético precalculado, brazos ABBA (default, low, low, default) × rondas, cada brazo con Player y captura nuevos. Reporta valores crudos, mediana/cuartiles por modo, incertidumbre de captura y un veredicto descriptivo fijado de antemano: «efecto candidato» solo si |Δmediana| > 2×incertidumbre y el signo coincide en todas las parejas de brazos; si no, «sin efecto distinguible». No es una prueba estadística. Comando: `.venv/bin/python scripts/eval/player_latency_probe.py run --out artifacts/benchmark/player-latency-probe-1` (~1 min; abre BlackHole).
- **Paso 2, solo si el paso 1 muestra efecto:** repetir `scripts/eval/device_loopback.py run` con `--player-latency low` y el mismo conjunto de 3 clips y config que se confirme tras el continuo; el brazo default sale de la misma configuración. No recortar silencios de forma distinta entre brazos.

## Semántica de informes en stream continuo (revisión C1)
- **Calidad MT «unavailable»:** si hay `mt_final` en las trazas pero no se pueden atribuir a ítems (el stream continuo no trae correspondencia fuente→salida por ítem), la razón dice «SÍ emitió N mt_final… sin calidad de MT por ítem. No significa que falte la MT»; solo dice «no emitió ningún mt_final» si no hay ninguno.
- **RTF de emisión TTS:** `(recv último − recv primer chunk) / Σ duración salvo el último` (campos `formula`, `valid_as_synthesis_cost`). En streaming incluye la espera de audio de entrada/replay entre chunks: **no es costo de síntesis** y un valor > 1 no demuestra que el TTS sea más lento que tiempo real. La evidencia de espera real es backlog, `underrun_s`, el lag por segmento y el audible estimado.
- **Cuantiles en stream continuo:** el IC bootstrap por hablante/cluster no captura la dependencia temporal (backlog y lag se arrastran entre ítems del mismo stream); el informe lo avisa en cada tabla y los IC son optimistas.
- `report.write_report(run_dir, out_path=...)` regenera un informe viejo desde `run.json` + `traces.jsonl` sin tocar los archivos de la corrida (corrige las razones rotuladas antes del cambio); ejemplo: `artifacts/benchmark/report-fixes/C1-report.md`.

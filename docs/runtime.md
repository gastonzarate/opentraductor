# opentraductor: diseño del runtime de macOS (fase de investigación)

Estado: diseño. No hay código, drivers instalados ni micrófono activado. Candidatos y fuentes en `docs/model-candidates.json`.
Rutas de implementación (`runtime/`, `ui/`) pendientes de confirmación del coordinador y de la validación del benchmark.

## 1. Presupuesto de latencia

Meta: primer audio traducido < 300 ms desde el **fin real** de la frase (instante acústico en que el hablante deja de hablar según la referencia del corpus), no desde la decisión del VAD.

```
t0 fin real de voz
 ├─ espera de endpoint (hangover VAD / semántico)   E
 ├─ ASR: último parcial estable                      A
 ├─ MT incremental: texto traducible                 M
 ├─ TTS: primer chunk                                T
 └─ cola de salida + buffer CoreAudio                O
primer audio = E + A + M + T + O
```

Un hangover VAD típico de 200-300 ms consumiría todo el presupuesto. Por eso el diseño evita depender de E: ASR y MT trabajan sobre parciales mientras se habla, y el TTS arranca con texto traducido estable antes de que termine la frase (la salida puede empezar antes de t0). Una salida anticipada no cuenta como éxito por sí sola: el benchmark debe verificar que el contenido emitido corresponde a la frase de origen y no es una traducción prematura o errónea. También se mide el **lag continuo** entre el habla de origen y la salida. Nada de esto está medido todavía.

Reglas para el benchmark (propiedad del coordinador/ingeniería del benchmark; las listo como requisitos del runtime):
- Cada evento lleva marca monotónica (`mach_absolute_time`) en captura, ASR parcial/final, MT, primer sample TTS y primer sample entregado al dispositivo de salida.
- La salida se mide en el callback de CoreAudio con latencia del dispositivo (`kAudioDevicePropertyLatency` + safety offset + buffer), no al salir del modelo.
- El runtime no puede usar contexto futuro: solo audio ya entregado al adaptador.

## 2. Arquitectura de procesos

Un proceso Python (arranque) con hilos/tareas asíncronas y colas acotadas; el modelo corre en procesos de trabajo separados solo si el benchmark demuestra interferencia GIL/GPU.

```
CoreAudio input ──► ring buffer (lock-free) ──► resampler 16 kHz ──► VAD/segmentador
                                                             │
                                                    chunks 20-80 ms
                                                             ▼
                                              ASR streaming (parciales + finales)
                                                             ▼
                                          estabilizador de texto (prefijo acordado)
                                                             ▼
                                              MT incremental (por cláusula)
                                                             ▼
                                              TTS streaming (chunks PCM)
                                                             ▼
                                   mezclador de salida ──► dispositivo virtual (micrófono para Meet/Discord)
                                                      └──► monitor local opcional (depuración)
```

Cada etapa implementa el contrato de adaptador que defina el benchmark (a coordinar; este documento no lo fija). Esperado: `push(audio|text, t_in)`, iterador de eventos con `t_emit`, `cancel()`, `reset()`.

### Captura
- AVAudioEngine o CoreAudio HAL (`AudioUnit` input) a 48 kHz, conversión a 16 kHz mono float32 con resampler de fase lineal de bajo retraso.
- Buffers de 10-20 ms. Ring buffer SPSC; el callback de audio no asigna memoria ni toma locks.
- Selector de dispositivo de entrada por UID de CoreAudio (no por nombre); persistir el UID elegido en la base de configuración.
- El permiso de micrófono (TCC) se solicita solo cuando el usuario activa captura. Hasta que el coordinador lo autorice, el runtime solo lee archivos del corpus.

### VAD y segmentación
- Candidatos Silero VAD o TEN VAD; licencia sin verificar, no adoptar antes de leerla. Procesa en CPU, fuera del camino GPU.
- Dos señales distintas: (a) `speech_active` de baja latencia para detener TTS al interrumpir; (b) endpoint para cerrar segmento. Endpoint adaptativo: hangover corto tras puntuación final del ASR, más largo si el parcial termina a media cláusula.
- Se registra siempre la marca del VAD aparte de la marca de fin real para que el benchmark exponga el error del VAD.
- Segmentación máxima de seguridad (por ejemplo 15-20 s) para acotar memoria y contexto.

### Cancelación y backpressure
- Cada segmento tiene `generation_id` y un `seq` de orden. Solo se descartan **parciales reemplazados** (hipótesis de ASR/MT que una versión posterior reescribe), no traducciones pendientes de contenido ya dicho.
- Que el mismo hablante vuelva a hablar mientras suena una traducción es normal en traducción simultánea. El runtime **no corta** la traducción pendiente: mantiene el orden y encola el nuevo segmento.
- Colas acotadas en orden. Cuando una cola se desborda se aplica una política explícita (por ejemplo acelerar levemente la salida o acortar pausas) y el desbordamiento o cancelación se muestra en la interfaz y se guarda como métrica (`overflow_ms`, `cancelled`), sin ocultarlo. Entrada de audio: nunca bloquea; si debe descartar, cuenta `dropped_ms`.
- Cancelación real solo por acción del usuario o error de etapa; se registra con motivo.
- Una sola carga de modelos a la vez durante benchmarks (regla del coordinador).

### Reproducción y dispositivo virtual
- Salida principal: un dispositivo de entrada virtual que Meet/Discord vean como micrófono.
- Opciones evaluadas:
  1. **BlackHole**, dependencia externa bajo GPL-3.0 ([repositorio y términos](https://github.com/ExistentialAudio/BlackHole)). No se redistribuye con la app: el usuario lo instala por separado y la app solo lo selecciona como dispositivo por UID. Las condiciones de licencia para otros usos las indica su README.
  2. **Driver propio** (AudioServerPlugIn de CoreAudio): control total y sin dependencia GPL, mayor costo y firma/notarización.
  3. Aggregate/Multi-Output de macOS para el monitor local, ver advertencias de deriva de reloj en la documentación de BlackHole.
- Recomendación: opción 1 como dependencia externa no empaquetada. No se instala nada hasta la señal del coordinador.
- Monitor local opcional: segundo stream hacia otro dispositivo de salida, con volumen y mute independientes; apagado por defecto.
- La salida se resamplea al formato del dispositivo (24/48 kHz) una sola vez, al final.

## 3. Registro local

Una base SQLite pequeña de la app, separada por completo de los artefactos del benchmark y de los audios públicos.

Tablas propuestas (texto y métricas únicamente):

```sql
sessions(id INTEGER PK, started_at TEXT, ended_at TEXT, config_json TEXT);
segments(id INTEGER PK, session_id INTEGER, seq INTEGER,
         source_text TEXT, target_text TEXT,
         t_speech_end_ms INTEGER NULL, t_first_audio_ms INTEGER NULL,
         endpoint_wait_ms INTEGER, asr_ms INTEGER, mt_ms INTEGER, tts_first_ms INTEGER,
         lag_p50_ms INTEGER, dropped_ms INTEGER, overflow_ms INTEGER, cancelled INTEGER, cancel_reason TEXT);
settings(key TEXT PK, value TEXT);   -- UID de dispositivos, modelo activo
```

- La aplicación **solo guarda textos y métricas**. No graba ni conserva audio de conversaciones, y no tiene opción para hacerlo. Los audios públicos y artefactos del benchmark viven aparte (`data/`) y no se mezclan con esta base.
- Los textos de reuniones no se envían a ningún servicio externo; el código no incluye clientes de red en el camino de ejecución (verificable con una prueba que bloquee sockets salientes).
- WAL activado; escritura asíncrona desde un hilo dedicado para no afectar el camino de audio.
- Los textos son datos personales: permitir borrado por sesión y exportación.

## 4. Interfaz

Una app de barra de menú con: selector de micrófono de entrada, selector de dispositivo virtual de salida, selector opcional de monitor local, selector de modelos por etapa, indicador de latencia (último segmento y lag continuo), indicador visible de desbordamiento/cancelación, y visor del historial de texto. No hay grabación de audio. Tecnología (SwiftUI o Python con interfaz web local) se decide tras el benchmark; con Python, el control corre en `127.0.0.1` solamente.

## 5. Plan de integración (sin ejecutar)

1. Esperar la señal de benchmark validado y la confirmación de rutas. Mientras tanto, solo investigación y verificación de licencias/rutas instalables.
2. Implementar adaptadores contra el contrato del benchmark, empezando por la baseline P1. Orden: ASR → TTS → MT, cada uno con prueba de humo sobre el corpus público.
3. Medir cada adaptador de forma aislada y luego en cascada, con hardware exclusivo coordinado.
4. Implementar captura, VAD y cola de salida, primero contra archivos (reproducción simulada), luego con dispositivo real.
5. Con aprobación del coordinador: instalar dispositivo virtual y probar con Meet/Discord.
6. Priorizar siempre la ruta ejecutable ya instalable en este Mac antes de modelos grandes. Evaluar P2 (Hibiki-Zero) solo después de medir la baseline y si hay ruta Metal; portarlo se decide con el coordinador.

## 6. Riesgos conocidos

- **Endpoint**: puede consumir por sí solo el presupuesto de 300 ms. Mitigación: salida anticipada con parciales estables; costo en calidad a medir.
- **Español argentino**: ninguna fuente consultada documenta rendimiento en rioplatense (voseo, ‘sh’ de ll/y), tampoco para Whisper. Requiere subconjunto específico en el corpus.
- **Licencias**: Hibiki-Zero (CC BY-NC-SA 4.0), Voxtral TTS (CC BY-NC) y SeamlessStreaming (CC-BY-NC 4.0) restringen uso comercial; BlackHole es GPL-3.0 y queda como dependencia externa. Verificadas en tarjeta/README: opus-mt-es-en (Apache-2.0), Qwen3-ASR (Apache-2.0), Qwen3-TTS (Apache-2.0), Whisper large-v3-turbo (MIT), Parakeet v3 (CC-BY 4.0), Kokoro (Apache-2.0). Código de Pocket TTS: MIT. **Pendiente**: licencia de los pesos de Pocket TTS (tarjeta HF inaccesible) y de cada voz usada.
- **Streaming real del ASR**: Qwen3-ASR en streaming oficial exige vLLM (CUDA); Parakeet-TDT es offline por segmento. El streaming en Mac será implementación de ventana/caché de terceros, por validar.
- **Ruta Metal de Hibiki-Zero**: el README oficial declara GPU NVIDIA; el soporte MLX solo está confirmado para el Hibiki original FR→EN.
- **Cifras anunciadas** (Pocket TTS ~200 ms, Qwen3-TTS 97 ms, Voxtral ~90 ms) corresponden al fabricante, otro hardware y excluyen ASR, MT y endpoint.

## 8. Dependencias y revisiones de la baseline (verificadas por instalación, sin inferencia)

Entorno de prueba aislado fuera de la carpeta fuente: `uv venv --python 3.12 /tmp/kestrel-venv`. Instalación resuelta sin conflictos con `uv pip install pocket-tts parakeet-mlx mlx-audio ctranslate2 sentencepiece transformers torch huggingface_hub sounddevice`.

| Paquete (PyPI) | Versión resuelta | Licencia (PyPI) | Uso |
|---|---|---|---|
| mlx | 0.32.4 | MIT | runtime GPU |
| parakeet-mlx | 0.7.0 | Apache-2.0 | ASR |
| mlx-audio | 0.6.5 | MIT | Kokoro/Qwen3-TTS en MLX |
| pocket-tts | 4.0.1 | sin campo en PyPI; LICENSE del repo = MIT | TTS CPU |
| ctranslate2 | 4.9.0 | MIT | MT (Marian) |
| transformers | 5.23.0 | Apache-2.0 | conversión/tokenizer |
| sentencepiece | 0.2.3 | Apache-2.0 | tokenizer Marian |
| torch | 2.15.0 | BSD/Apache | requerido por pocket-tts y la conversión |
| huggingface-hub | 2.4.0 | Apache-2.0 | descarga con revisión fija |
| sounddevice | 0.5.6 | MIT | captura/reproducción PortAudio |

Todos con Python 3.12. `kokoro` y `kokoro-mlx` exigen `<3.13`; compatibles con 3.12. Pendientes de verificar: G2P de Kokoro (`misaki`, posible `espeak-ng`, ausente en este equipo) y la licencia de `silero-vad` (PyPI no declara).

Revisiones de modelos (fijar con `revision=`):

| Modelo | Repo@revisión | Licencia | Estado |
|---|---|---|---|
| ASR | `mlx-community/parakeet-tdt-0.6b-v3@ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15` | CC-BY 4.0 | descargado (2.3 GB) |
| MT | `Helsinki-NLP/opus-mt-es-en@c96e2c5399ebfae4fc43d9669556b9afa74bb69d` | Apache-2.0 | descargado y convertido: `~/.cache/opentraductor/models/opus-mt-es-en-ct2-int8` (80 MB, `conversion.json` con comando y versiones) |
| TTS (alternativa) | `kyutai/pocket-tts@db04f0e653473e5c7333c69686ac92b7460b0c1d` | CC-BY 4.0 (metadata HF) | repo gated, sin token; no se elude el acceso. Ya no bloquea la baseline |
| Voces Pocket | `kyutai/tts-voices@323332d33f997de8394f24a193e1a76df720e01a` | por carpeta: CC0 (voice-donations, voice-zero), CC-BY 4.0 (vctk, alba-mackenna); CC-BY-NC (expresso, ears) excluidas | pendiente |
| TTS baseline provisional | `mlx-community/Kokoro-82M-bf16@a71e4d38b236d968966a2002c4c895dbd12b1c3c` | Apache-2.0 | descargado (322 MB) |

Baseline provisional: Parakeet + Marian (CTranslate2 int8) + Kokoro. G2P de Kokoro: `misaki[en]` 0.9.4 con `en-core-web-sm` 3.8.0, probado sin espeak-ng del sistema. `silero-vad` es MIT (LICENSE del repo leído). Prueba de importaciones (ctranslate2, sentencepiece, parakeet_mlx, pocket_tts, mlx_audio, mlx, sounddevice) correcta; MLX ve el dispositivo GPU. Alternativa de clonación posterior: Qwen3-TTS 0.6B en MLX con muestra CC0.

Los pesos están en la caché estándar de Hugging Face del usuario, fuera de la carpeta fuente. No se ejecutó ninguna inferencia.

## 9. Prueba de inferencia sin red (a ejecutar tras validar el benchmark)

Información del coordinador: macOS 26.6 arm64; PortAudio enumera solo micrófono y altavoces integrados (BlackHole ausente; instalador 0.7.1 descargado, no instalado: no tocar el driver todavía).

- Ejecutar la inferencia dentro de `sandbox-exec -p '(version 1)(allow default)(deny network*)' .venv/bin/python ...`. Un canario `connect_ex(1.1.1.1:443)` devuelve EPERM, lo que confirma el bloqueo sin desconectar la red del sistema.
- Además: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` y rutas de snapshot locales.
- Esta prueba cubre solo inferencia. La interfaz en localhost se evalúa por separado con permiso de loopback.

## 7. Fuentes primarias consultadas

- Hibiki-Zero: https://github.com/kyutai-labs/hibiki-zero · https://huggingface.co/kyutai/hibiki-zero-3b-pytorch-bf16 · https://arxiv.org/abs/2602.11072
- Hibiki: https://github.com/kyutai-labs/hibiki
- Kyutai STT/TTS: https://huggingface.co/kyutai/stt-2.6b-en · https://huggingface.co/kyutai/tts-1.6b-en_fr
- Pocket TTS: https://github.com/kyutai-labs/pocket-tts
- Parakeet TDT v3: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- Qwen3-ASR: https://github.com/QwenLM/Qwen3-ASR · Qwen3-TTS: https://arxiv.org/abs/2601.15621
- Voxtral TTS: https://arxiv.org/abs/2603.25551 · https://mistral.ai/news/voxtral-tts/
- Kokoro: https://huggingface.co/hexgrad/Kokoro-82M · mlx-audio: https://github.com/Blaizzy/mlx-audio
- SeamlessStreaming: https://huggingface.co/facebook/seamless-streaming
- opus-mt-es-en: https://huggingface.co/Helsinki-NLP/opus-mt-es-en
- BlackHole: https://github.com/ExistentialAudio/BlackHole
- MLX y Neural Accelerators del M5: https://machinelearning.apple.com/research/exploring-llms-mlx-m5

Parte del contenido de fuentes web se parafraseó para respetar licencias de contenido.

## Opciones de optimización del pipeline (`PipelineConfig`, todas apagadas por defecto)

Medidas solo sobre 3 clips DEVELOPMENT (`artifacts/runtime-opt/`, `artifacts/runtime-moonshine/`); piloto, no conclusión estadística. Los límites de fin de habla son estimados.

| Opción | Efecto | Limitación conocida |
| --- | --- | --- |
| `endpoint_silence_ms` | Silencio del VAD para cerrar segmento (250 por defecto). | 96–160 ms puede sobre-segmentar con pausas naturales; sin medir en continuo. |
| `asr_chunk_ms` | Audio por paso del ASR. | Con Parakeet, 96 ms es peor que 160: cada `add()` cuesta ~110–150 ms fijos y el ASR no sigue el tiempo real (WER 0.558 en el piloto). |
| `asr_skip_silent_tail` | Al endpoint no se envía al ASR la cola de audio que es solo silencio según el VAD. | Depende del VAD: un falso negativo en el final de una palabra suave perdería voz. |
| `tts_trim_lead_margin_ms` | Recorta el silencio inicial del primer chunk TTS de cada frase (umbral −60 dBFS) y deja ese margen. | Reduce el tiempo de reproducción estimada, no el «PCM disponible» (que usa −50 dBFS). |
| `asr_backend` | `parakeet` (hilo MLX), `moonshine-tiny` o `moonshine-small` (CPU, hilo propio). | Moonshine emite texto sin puntuación ni mayúsculas. |
| `tts_first_chunk_max_words` | Parte la primera cláusula en cabeza + resto. **Objetivo flexible, no tope**: la cabeza puede superar N si hay racha de números/identificadores o si no hay corte antes de N, y en el segundo pase puede terminar en palabra funcional. Nunca corta junto a dígitos o identificadores. Conserva todas las palabras. | Heurístico; el corte `solemnity | and respect` produjo un artefacto audible («In respect»). |
| `tts_bind_conjunctions` | Con el anterior, no corta justo antes ni después de and/or/but/nor/yet. | Solo garantiza la propiedad textual; no se afirma que resuelva el artefacto acústico hasta medirlo. |
| `speculative_mt_tts` (+ `spec_silence_ms`, `spec_max_entries`) | Tras una pausa VAD y con el ASR al día (toda la voz procesada, cola solo silencio) precalcula MT y, si el ASR no comparte el hilo MLX (Moonshine), el primer fragmento TTS del parcial. **Nada especulativo se emite**: el caché solo se usa si el texto final coincide EXACTAMENTE (misma normalización); si no, se descarta y corre la ruta normal. Una especulación por segmento, caché acotado, cede ante audio confirmado en curso, se cancela en stop/restart/cierre. Se compara fuente, configuración (voz, velocidad, recorte, troceo, normalización) y generación del stream ANTES de esperar la tarea: un fallo conocido no espera ni lanza TTS nuevo. Contadores en `adapter_metadata.speculation`: started, hit, miss_text, miss_config, error, cancelled, skipped_busy, skipped_limit, inflight_unmeasured, `discarded_observed_ms` (cómputo observado y COMPLETADO de resultados descartados; lo cancelado en vuelo no se mide, así que no es el desperdicio total) y `overlap_estimate_ms` (estimación de espera evitada en aciertos, NO ahorro causal de latencia). | Sin medir con modelos reales. Con Parakeet solo se especula la MT. Si el hablante reanuda tras la pausa, el parcial queda obsoleto (fallo, cómputo desperdiciado). Los trabajos enviados a un executor son FIFO y no cancelables: una MT/TTS especulativa ya lanzada ocupa su hilo hasta terminar y puede retrasar hasta ~100 ms el audio confirmado que llegue justo después; no hay prioridad absoluta. Reiniciar un stream con id vivo cierra y drena primero la sesión anterior. |
| `asr_normalize_text` | Mayúscula inicial y punto final antes de MT. El crudo se conserva (`text_es_raw`, `asr_final.text_raw`; en benchmark: `adapter_metadata.asr_text_log`). | Sin beneficio medido (BLEU 24.5 → 18.2 en 3 frases por mayúsculas de Marian); no activar por defecto. |

Receta de cola (no ejecutada): `scripts/eval/continuous_recipe.sh`. Es una concatenación de grabaciones reales (8 ítems development, sin split test) con pausas sintéticas de 1.5 s; no es conversación natural. Cobertura por ítem con `scripts/eval/continuous_coverage.py` (separa segmentos múltiples de duplicados exactos; más de 8 `asr_final` no prueba omisión ni sobre-segmentación). El texto audible en continuo queda pendiente: el arnés no exporta WAV por ítem.

### Cierre de stream y generaciones

Cada `start_stream` crea una generación; `Engine._live_gen` guarda la vigente por id. `end_stream` espera a drenar las salidas hasta `end_stream_drain_timeout_s` (5 s por defecto). Si expira emite `end_stream_drain_timeout` y desde ese momento NADA de esa generación se emite (ni `mt_final`, ni `tts_chunk`, ni errores): se purgan sus segmentos de la cola, el trabajo confirmado que siga en curso se descarta al terminar y la síntesis en el hilo se corta al terminar la cláusula actual. Un trabajo ya enviado a un executor no es cancelable: sigue ocupando su hilo y puede retrasar la salida de la sesión nueva. `stale_dropped` cuenta lo descartado.

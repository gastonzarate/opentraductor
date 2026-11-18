# opentraductor

Traductor de voz **100 % local** de español (Argentina) a inglés (EE. UU.) para macOS con Apple Silicon.
Pensado para reuniones (Google Meet, Discord) a través de un micrófono virtual.

> **Estado real:** el perfil candidato provisional y sus cifras finales están en
> [`docs/uso.md`](docs/uso.md#estado-actual). Dos corridas finales, sin mezclarlas:
> - **Offline (sin red, 36 streams):** mediana de 409 ms hasta el primer audio disponible.
> - **Salida a BlackHole 2ch (12 clips, corrida aparte):** mediana de 548 ms desde el fin de habla anotado hasta el
>   inicio de sonido estimado, corregido por ADC.
>
> Ambas quedan por encima de 300 ms, y el objetivo no está certificado: los límites de anotación son estimados y la
> muestra es de 4 hablantes. No hay prueba con micrófono físico ni en Meet o Discord. El servicio y la interfaz están
> probados con motor y audio **falsos** (pruebas automáticas).

## Privacidad

- Todo el procesamiento es local. Tras el `setup`, el runtime fuerza modo offline para los modelos.
- **Edición Mac App Store:** sin licencia propia ni activación (cobra la tienda). La única conexión de red es la descarga
  de modelos del primer arranque, desde Hugging Face; no envía audio, textos ni datos del equipo. Después, cero red.
- **Edición de descarga directa (con licencia): única conexión de red tras el setup: la activación de la licencia.** Se hace una sola vez, al ingresar la clave
  (`POST https://api.lemonsqueezy.com/v1/licenses/activate`). Envía la clave y el nombre de esta Mac. **No envía audio,
  textos, historial ni datos del equipo.** Después la app funciona sin red; no hay revalidación periódica. La prueba gratis
  de 7 días no usa red: se cuenta en local desde la primera vez que abrís la app (`license.json`, junto al historial).
- **No se graba ni se guarda audio.** Solo se guardan textos (original y traducción) en una base SQLite local
  pequeña, que podés exportar o borrar desde la interfaz.
- El micrófono se abre únicamente cuando pulsás **Iniciar** y se cierra al **Parar**.
- La interfaz escucha solo en `127.0.0.1`, exige un token por ejecución y valida `Host`/`Origin`.
- Sin telemetría. Sin CDN: los recursos de la interfaz son locales.

## Requisitos

- macOS en Apple Silicon (probado: M5 Pro, macOS 26), Python 3.12, [uv](https://docs.astral.sh/uv/).
- Espacio en disco: unos 3 GB para los pesos base (aproximado). No incluye el extra Moonshine ni la clonación.

## Instalación

### Instalación base

```bash
uv sync --extra runtime
uv run --extra runtime python -m runtime.setup   # descarga modelos (ÚNICO paso con red) y convierte Marian
uv run --extra runtime python -m runtime check   # verifica que todo está en disco
```

Siempre con `--extra runtime`. Sin él, uv quita las dependencias del runtime.

### Moonshine (opcional, experimental)

```bash
uv sync --extra runtime --extra moonshine
```

Si usás Moonshine, conservá **ambos extras en todo comando**: `--extra runtime --extra moonshine`. Un comando con
solo `--extra runtime` vuelve a quitar el paquete `moonshine-voice`.

Modelos (revisiones fijas en `runtime/config.py`):

| Etapa | Modelo | Licencia |
|---|---|---|
| ASR | `mlx-community/parakeet-tdt-0.6b-v3` (NVIDIA Parakeet TDT) | CC-BY 4.0 |
| Traducción | `Helsinki-NLP/opus-mt-es-en` convertido a CTranslate2 int8 | Apache-2.0 |
| TTS | `mlx-community/Kokoro-82M-bf16` (voces en-US) | Apache-2.0 |
| VAD | Silero VAD | MIT |

El código del proyecto es MIT. Los pesos y los datos de evaluación mantienen sus propias licencias y
atribuciones (ver `docs/model-candidates.json`, `docs/datasets.md`). Kokoro no clona voces. La clonación
(p. ej. Qwen3-TTS) no está integrada; su integración es posterior.

## Uso

```bash
uv run --extra runtime python -m runtime devices     # lista micrófonos y salidas (no abre ninguno)
uv run --extra runtime python -m runtime run         # abre http://127.0.0.1:8877/ (usar --port para otro)
```

El perfil baseline vigente (C2) y sus cifras están en [`docs/uso.md`](docs/uso.md). El piloto T2, una configuración anterior
ya superada, se conserva con su comando y su medición en [`docs/historial-perfiles.md`](docs/historial-perfiles.md).

La guía completa del perfil candidato, con su comando, la configuración de audio y las limitaciones, está en `docs/uso.md`.

La clonación de voz tiene un piloto fuera de la app ([RESULTS](artifacts/benchmark/crosslingual-pilot/RESULTS.md)).
La página de escucha con la referencia española está en
`artifacts/benchmark/listening-crosslingual-pilot1-html/index.html`. No está integrada en la aplicación.

Opciones de ASR y TTS:

- `--asr-backend`: `parakeet` por defecto. `moonshine-tiny` y `moonshine-small` son experimentales.
- `--endpoint-ms`: silencio que cierra un segmento (60 a 2000). Si se indica, prevalece sobre el valor guardado en
  la base. Si no, rige el guardado o el valor por defecto (250).
- `--audio-latency {default,low,high}`: latencia que se pide a la biblioteca de audio. `default` mantiene la
  actual (`high`). `low` es experimental: en la prueba con BlackHole (96.2 ms con default frente a 99.2 ms con low,
  diferencia de 3 ms dentro de una incertidumbre de 20.7 ms) no mostró ahorro, y puede cortar audio en equipos
  ocupados. `high` es la opción más estable. Si se indica, prevalece sobre el valor guardado. La interfaz tiene el mismo selector y solo se cambia
  con la sesión detenida. El estado muestra la latencia que informa el stream, no una medición física.
- `--skip-silent-tail`: no envía al ASR la cola silenciosa del segmento.
- `--tts-trim-lead-ms`: recorta el silencio inicial del TTS, conservando ese margen en ms.
- `--tts-first-chunk-max-words N` (entero mayor que 0) y `--normalize-asr-text`: experimentales, apagados por
  defecto. El segundo normaliza el texto del ASR antes de traducir. El historial guarda el texto normalizado y el
  original del ASR por separado. Los registros anteriores a esa función no tienen original guardado, y no se
  reconstruye.

Estas opciones no garantizan el objetivo de 300 ms. Alteran el procesamiento, y su efecto se midió solo de forma
experimental. La configuración efectiva del motor aparece en el estado del servicio y se guarda con cada sesión
(sin audio).

Opciones de `run`: `--port`, `--db RUTA` (por defecto `~/.local/share/opentraductor/history.sqlite3`),
`--endpoint-ms` (silencio que cierra un segmento, por defecto 250), `--no-warmup`.

En la interfaz elegí micrófono, salida (para reuniones: el micrófono virtual) y, opcionalmente, un monitor
local para escuchar lo que se envía. Los textos en vivo, el historial (exportar/borrar) y las colas
(desborde/cancelación) se ven en pantalla.

### Micrófono virtual para reuniones (BlackHole)

La app no incluye ni instala drivers. [BlackHole](https://github.com/ExistentialAudio/BlackHole) es una
dependencia externa bajo GPL-3.0: instalala vos (requiere tu contraseña de administrador), elegí
"BlackHole 2ch" como salida en la app y como micrófono en Meet/Discord. La interfaz tiene una pantalla
**Micrófono virtual** que detecta BlackHole 2ch y, si falta, muestra los 3 pasos (instalar, reiniciar la app, elegirlo en la
reunión). Hasta que esté instalado, la interfaz solo lista tus dispositivos reales y podés probar con altavoces/monitor local.

## Pruebas y benchmark

```bash
uv run --extra runtime python -m pytest tests -q            # incluye tests/runtime y tests/benchmark
uv run --extra runtime python -m benchmark run --help       # arnés de benchmark (ver docs/benchmark.md)
```

Adaptador de la baseline: `runtime.adapters.baseline:BaselineAdapter`.

## Estructura

- `runtime/` motor (`pipeline.py`), componentes (`components.py`), servicio (`service.py`), HTTP local
  (`http_control.py`), historial (`text_store.py`), audio (`audio_io.py`), CLI (`__main__.py`, `setup.py`).
- `ui/` interfaz web local. `benchmark/` arnés de medición. `docs/` diseño, candidatos y datasets.
- `data/` corpus públicos de evaluación (separados del historial de conversaciones).

## Limitaciones conocidas

- Latencia por encima de 300 ms en las dos corridas finales (ver `docs/uso.md`, sección «Estado actual»).
  El endpoint y el cierre del ASR son los mayores costos.
- Audio de salida solo en el dispositivo elegido; sin cancelación de eco ni mezcla de varias voces.
- Español rioplatense sin evaluación específica completa; ver informes del benchmark.

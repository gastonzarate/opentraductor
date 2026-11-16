# Candidato ASR: Moonshine streaming en español

Estado: preparación. Instalación aislada y pesos descargados. Sin inferencia, sin micrófono y sin evaluación. No es una adopción ni una medida de rendimiento. Tras la ventana de benchmark se evaluarán uno o dos clips, no una rejilla.

## Fuentes primarias

- Modelos disponibles: https://github.com/moonshine-ai/moonshine/blob/main/docs/models/available-models.md
- Licencias: https://github.com/moonshine-ai/moonshine/blob/main/docs/license.md y https://www.moonshine.ai/license
- Paquete: https://pypi.org/project/moonshine-voice/0.1.6/
- API incremental: https://github.com/moonshine-ai/moonshine/blob/main/docs/api/classes.md
- Cabeceras C con las arquitecturas: https://github.com/moonshine-ai/moonshine/blob/main/core/moonshine-c-api.h

## Qué se instaló

Script reproducible: `scripts/setup_moonshine.py` (solo stdlib de Python; llama a uv). Todo queda bajo `models/moonshine/`, que está en `.gitignore`.

| Elemento | Valor |
| --- | --- |
| Paquete | `moonshine-voice==0.1.6` (licencia MIT en PyPI) |
| Wheel | `moonshine_voice-0.1.6-py3-none-macosx_15_0_arm64.whl`, sha256 `71f0fa9d0b01fa317f86a85e846e9faa247be985f92b50970817dd999e400ff4`, 18 670 056 bytes |
| Requisito de SO | macOS 15 o superior, arm64 (esta Mac: macOS 26.6, arm64) |
| Python del venv | 3.12.16 (`models/moonshine/venv`) |
| Dependencias | numpy 2.5.4, sounddevice 0.5.6, requests, tqdm, etc. (ver `uv pip list`) |

El wheel arm64 existe y se instaló. Los pesos español streaming están disponibles, así que la pregunta del coordinador tiene respuesta afirmativa para ambas arquitecturas.

## Modelos español streaming

Revisión fijada en el paquete: `quantized_26_08_24`. Todos los archivos están en `models/moonshine/manifest.json` con URL, tamaño y sha256.

| Modelo | Arquitectura (C API) | Parámetros | WER de la tabla del vendor | Tamaño de archivos | Ruta |
| --- | --- | --- | --- | --- | --- |
| tiny-streaming-es | `MOONSHINE_MODEL_ARCH_TINY_STREAMING` = 2 | 34 M | 6.2 % | 32.0 MB | `models/moonshine/assets/download.moonshine.ai/model/tiny-streaming-es/quantized_26_08_24/` |
| small-streaming-es | `MOONSHINE_MODEL_ARCH_SMALL_STREAMING` = 4 | 123 M | 4.9 % | 128.0 MB | `models/moonshine/assets/download.moonshine.ai/model/small-streaming-es/quantized_26_08_24/` |

Los WER vienen de la tabla de docs/models/available-models.md del vendor. Son figuras del vendor, no mediciones nuestras.

Se descartó el modelo Spanish Base (58 M, no streaming). Es Community no comercial y el vendor lo marca como deprecado frente a los streaming.

## Licencia

Decisión del TL (revisada en fuentes primarias): `tiny-streaming-es` y `small-streaming-es` se usan bajo **MIT**.

Fuentes y versiones:
- `https://github.com/moonshine-ai/moonshine/blob/main/docs/models/available-models.md`: nombra los modelos
  streaming en español como MIT.
- `https://github.com/moonshine-ai/moonshine/blob/main/docs/license.md`: lista de modelos no MIT, que no incluye
  los dos streaming.
- Paquete `moonshine-voice==0.1.6` (PyPI, licencia MIT).
- Revisión de pesos `quantized_26_08_24` (ver `models/moonshine/manifest.json`).

Discrepancia registrada (no bloquea el uso):
- Al descargar, la CLI imprime «Using a model released under the non-commercial Moonshine Community License» para
  tiny y small. Esto sale del condicional `wanted_language != 'en'` de `moonshine_voice/download.py` (línea 562,
  versión 0.1.6). Es un aviso genérico heredado para cualquier idioma distinto de inglés, no específico de los
  artefactos streaming.
- No hay archivo de licencia por peso dentro del paquete ni de los assets.

Spanish Base (58 M, no streaming) es Community no comercial y no se usa. Las licencias de modelos legacy no cambian.

## API para streaming

Según docs/api/classes.md del vendor (no ejecutada):

- `Transcriber` (archivos o entrada propia): `start()`, `add_audio(samples, sample_rate)`, `stop()`. `stop()` marca el segmento activo como completo y dispara los eventos, así que hace de vaciado final.
- `Transcriber.create_stream()` para varias fuentes de audio en un mismo modelo.
- `add_listener()` con `TranscriptEventListener`, eventos `on_line_started`, `on_line_updated`, `on_line_text_changed`, `on_line_completed`, `on_error`. Las líneas tienen estado y latencia del último paso en ms (`latency_ms`, parte de la C API).
- `MicTranscriber` abre el micrófono, por eso no lo he usado.
- Hay un `flush()` solo en `TextToSpeech`. Para ASR el vaciado es `stop()`. No encontré un flush parcial de ASR en la documentación.
- Los parámetros de streaming se cargan con `streaming_config.json` (en cada modelo).

## Verificación hecha

- `moonshine-voice --version` → `0.1.6`.
- `manifest.json` tiene 2 modelos con 8 archivos cada uno, todos verificados por sha256 y tamaño durante la ejecución del script.
- No se cargó ningún modelo, no hubo inferencia y no se abrió el micrófono.

## Pendiente antes de cualquier evaluación

1. Confirmar la licencia de los pesos español streaming (ver sección anterior).
2. Medir en la ventana de benchmark, con uno o dos clips del corpus de desarrollo, y no antes.
3. Decidir si el latency real del streaming de Moonshine cumple con el objetivo del proyecto. Esto no se puede afirmar a partir de los docs.

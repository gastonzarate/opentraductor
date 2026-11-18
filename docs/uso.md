# Uso (perfil baseline candidato)

Este perfil baseline candidato es **experimental**. No es la configuración por defecto y no cumple el objetivo de 300 ms.

## Perfil baseline candidato: comando

Instalá con los dos extras y conservalos en cada comando que use Moonshine:

```bash
uv sync --extra runtime --extra moonshine
```

Arranque del perfil:

```bash
.venv/bin/python -m runtime run --port 8877 --asr-backend moonshine-tiny --endpoint-ms 160 --tts-trim-lead-ms 30 --skip-silent-tail --tts-first-chunk-max-words 8 --tts-bind-conjunctions --tts-trim-tail off
```

Con uv, equivalente: `uv run --extra runtime --extra moonshine python -m runtime run ...` (mismos flags).

Luego abrí http://127.0.0.1:8877/ en el navegador.

## Qué cambia respecto del default de la CLI

| Opción | Default de la CLI | Este perfil |
| --- | --- | --- |
| `--port` | 8877 | 8877 |
| `--asr-backend` | `parakeet` | `moonshine-tiny` |
| `--endpoint-ms` | sin flag: rige el valor guardado en la base, o 250 | 160 (prevalece sobre el guardado) |
| `--tts-trim-lead-ms` | sin recorte | 30 |
| `--tts-trim-tail` | on desde T-47 (recorta el silencio final de la frase, deja 80 ms) | off (camino anterior; también `OPENTRADUCTOR_TTS_TRIM_TAIL=0`) |
| `--skip-silent-tail` | apagado | encendido |
| `--tts-first-chunk-max-words` | sin límite | 8 (objetivo heurístico, no límite duro) |
| `--tts-bind-conjunctions` | apagado | encendido |
| `--normalize-asr-text` | apagado | apagado (no incluido) |
| `--audio-latency` | sin flag: rige el valor guardado, o `default` | `default` |

El `--audio-latency` no garantiza `default`: si guardaste `low` en la interfaz, sin flag rige `low`. La medición
del perfil se hizo con `default`. Antes de medir, elegí **Predeterminada (la actual)** en el selector de latencia
de la interfaz, con la sesión detenida.

## Configuración de audio

- **Entrada:** tu micrófono físico, en la app.
- **Salida:** BlackHole 2ch, en la app. Para Google Meet o Discord, elegí BlackHole 2ch como micrófono en la aplicación de reuniones.
- **Monitor (opcional):** sirve para escuchar la traducción en tu equipo. Si lo activás, usá auriculares para no generar eco.
- **Iniciar y parar:** desde la interfaz. El micrófono solo se abre al pulsar Iniciar y se cierra al parar.
- **Historial:** guarda solo textos (original y traducción) en la base SQLite local. No se guarda audio.

BlackHole es un controlador externo que instala el usuario (ver README).

## Medición y limitaciones

Diagnóstico de desarrollo, anterior a la validación final (ver «Estado actual» al final):

- **No cumple 300 ms.** Medición C2 (tiny, endpoint 160, conjunciones ligadas, specOFF): 8 ítems de `development`, en un stream continuo.
  - Primer audio disponible: mediana de 495.8 ms.
  - Reproducción estimada (cola simulada, no es un dispositivo): mediana de 526 ms y máximo observado de 1.72 s.
  - Informe técnico: [C2 report.md](../artifacts/runtime-continuous/C2-tiny-F-chunk8-bind-ep160-specOFF/report.md).
  - Con un solo stream y 8 ítems no es posible un intervalo de confianza poblacional fiable. Los detalles están en el informe técnico.
- **Sin prueba física en Meet ni Discord.** La medición es de benchmark, no de la app con dispositivos reales.
- **Clonación de voz:** experimental, solo en el benchmark. No está integrada en la app ni usa la voz del usuario.
- **Referencia de escucha ES→EN:** [página de escucha del piloto](../artifacts/benchmark/listening-crosslingual-pilot1-html/index.html). No hay calificaciones humanas.

## Estado actual

Validación final con la configuración congelada del perfil (freeze `ac19c38f766e`), sobre el conjunto de prueba
(held-out) de OpenSLR 61: 12 clips de 4 hablantes. Esto no es una tarea completa. **No certifica ni refuta el objetivo
de 300 ms.**

El conjunto de prueba **no es totalmente inédito**. Antes del protocolo, se listaron en pantalla metadatos del
manifest, incluidos los del split de prueba: identificador, duración, hablante y los primeros ~60 caracteres de la
transcripción. No se escuchó el audio, no se usaron resultados del pipeline sobre la prueba y no se usó para ajustar.
Detalle en [FREEZE.json](../artifacts/final-validation/tiny-ep160-v1/FREEZE.json).

**Corrida offline (sin red, 36 streams = 12 clips × 3 repeticiones, sin fallos).**
Informe: [offline-1/RESULTS.md](../artifacts/final-validation/tiny-ep160-v1/offline-1/RESULTS.md)

- Primer audio disponible: mediana de 409 ms (rango observado 328–535 ms).
- Reproducción simulada (cola, no es dispositivo): mediana de 441 ms.
- Los 36 valores observados quedan por encima de 300 ms. Como los límites de anotación son estimados (±0.3 s, no validados), la corrida no permite afirmar ni descartar el objetivo.
- Calidad del ASR contra la transcripción humana: WER 0.050, CER 0.023.
- Calidad de la traducción: no medida, porque no hay referencias humanas en inglés.

**Salida a BlackHole 2ch (12 clips, corrida separada, sin sandbox de red).**
Informe: [device-1/RESULTS.md](../artifacts/final-validation/tiny-ep160-v1/device-1/RESULTS.md)

- Fin anotado de habla → callback bruto en BlackHole: mediana de 602 ms.
- Fin anotado de habla → inicio de sonido estimado (corregido por ADC): mediana de 548 ms.
- Escritura al reproductor → inicio de sonido: mediana de 158 ms.
- No hubo micrófono físico ni monitor. No es una prueba en Meet ni en Discord.

**Lo que no se afirma**

- Que el perfil cumpla 300 ms.
- Conversación natural: no se midió con audio continuo de reuniones.
- Naturalidad y calidad de la traducción: sin evaluación humana (no hay MOS).
- Clonación de voz: solo en el benchmark. No está integrada en la app ni usa la voz del usuario. La referencia española está en la [página de escucha del piloto](../artifacts/benchmark/listening-crosslingual-pilot1-html/index.html).
- Pendiente: prueba humana, micrófono físico, Meet y Discord, y el objetivo de 300 ms.

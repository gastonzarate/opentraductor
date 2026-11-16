# Uso (perfil baseline candidato y modo incremental experimental)

Este perfil baseline candidato es **experimental**. No es la configuración por defecto y no cumple el objetivo de 300 ms.

## Modo incremental experimental

Alternativa al motor baseline. Está integrado en la CLI y en la interfaz. Según el coordinador, las pruebas de infraestructura 2A y 2B con MLX
pasaron. Además, una prueba real de integración ejecutó `RuntimeService`, la fábrica `runtime.inc_mode.build` y el puente con un audio
público de control, dos veces (control y reinicio), con captura y salida inyectadas en memoria y sin hardware. Verificó texto por sesión,
PCM entregado al sink, aislamiento del reinicio, cierre y SQLite solo de textos. No valida calidad ni latencia: el piloto de 3 audios
muestra latencia alta (no es una validación de latencia), y siguen pendientes las pruebas con micrófono, interfaz con modelos y Meet o
Discord. No se prueba 300 ms. Las cifras de más abajo son del perfil baseline medido y no se trasladan al modo nuevo.

### Comando

```bash
.venv-inc-llm/bin/python -m runtime check --engine-mode incremental_experimental
.venv-inc-llm/bin/python -m runtime run --engine-mode incremental_experimental
```

`check` es de solo lectura: verifica el entorno, las huellas de la variante, del motor base y del manifiesto del modelo, y los
archivos locales de Moonshine, Kokoro, g2p y VAD. No carga modelos ni descarga nada. Si algo falta o no coincide, el modo no inicia.

### Entorno aislado

Se usa el entorno `.venv-inc-llm`, que ya existe. No ejecutes `uv sync` para él: esa orden gestiona el entorno `.venv` del proyecto y sus
extras. Con otro intérprete (por ejemplo `.venv`), `check` marca el entorno como incorrecto y el modo queda bloqueado. Las rutas
(`artifacts/analysis/incremental-llm-v1`, `incremental-adapter-v1`, `llm-mp4-a/models/qwen3-4b-instruct-2507-4bit`) y las huellas son fijas
para esta Mac y no se cambian por flags.

### Cómo funciona

Moonshine en streaming → prefijo estable del texto → Qwen sin caché (traducción) → Kokoro (voz).

- El audio entra de forma continua. El modo no espera necesariamente al fin del turno: puede traducir y hablar un fragmento cuando el
  prefijo del texto ya es estable.
- No reproduce una continuación meramente especulativa. Si una unidad se cancela o se interrumpe tras haber entregado audio parcial,
  esa entrega se registra con su estado (interrumpida o entregada parcial), sin atribuirle reproducción confirmada.
- Cada fragmento de voz se entrega al reproductor mientras la captura sigue. Si una cola se llena, se descarta audio nuevo y se avisa en
  pantalla (desborde).

### Voz y controles

- La voz es estándar y fija. El selector de voz queda deshabilitado y no se envía.
- No aplican en este modo: la voz, el silencio de cierre de segmento (`--endpoint-ms`) y los flags propios del baseline
  (`--asr-backend`, `--skip-silent-tail`, `--tts-*`, `--normalize-asr-text`, `--no-warmup`). La CLI los rechaza.
- Siguen igual: micrófono y salida (incluido BlackHole 2ch), monitor y volumen, latencia de audio y el historial habitual.
- La interfaz muestra «Traducción simultánea · Experimental». El estado del servicio incluye el detalle técnico.

### Textos guardados

Además del historial habitual, el modo guarda solo textos en una base SQLite aparte, `incremental_units.sqlite3`, junto al historial
(tablas `inc_lines` e `inc_units`). No guarda audio.

- El español se guarda por línea y el inglés por unidad emitida. No hay relación registrada entre una unidad y una línea.
  Por eso el historial habitual muestra el inglés de cada unidad con el español vacío.
- Las líneas de español del ASR también se guardan. De las unidades en inglés solo se guardan las que llegaron a entregar audio, con estado
  explícito: entregada, entregada parcial o interrumpida. Una continuación especulativa descartada sin audio entregado no se guarda.
- «Exportar» incluye también estos textos: las líneas de español (incluso las que no llegaron a traducirse) y las unidades de inglés con su
  estado, cada grupo por separado y sin alineación entre ellos. «Borrar todo» borra el historial habitual y estas tablas en una sola
  operación por base; si una parte falla, el error lo indica y no se presenta como borrado completo. Mientras hay una sesión incremental
  activa, «Borrar todo» se rechaza: hay que parar la sesión primero. Funciona igual al reabrir la aplicación en modo baseline si quedó una
  base incremental anterior. «Borrar» por sesión usa un mapping registrado por sesión; para datos anteriores al mapping solo actúa si el
  identificador existe y su prefijo no es ambiguo, y en otro caso se rechaza sin borrar.
- «Entregada» significa que el audio se envió al reproductor. No prueba que se haya escuchado: la reproducción no se verifica.

### Diferencia con el baseline

| | Baseline (perfil medido) | Incremental experimental |
| --- | --- | --- |
| Validación | corridas finales en este documento («Estado actual») | pruebas de infraestructura 2A/2B; integración real de servicio, fábrica y puente con captura/salida inyectadas (sin hardware); micrófono, UI con modelos y Meet pendientes |
| Cifras de latencia | sí, por encima de 300 ms | sin validación de latencia; el piloto de 3 audios muestra latencia alta; 300 ms no probado |
| Cierre de segmento | por silencio (`--endpoint-ms`) | por prefijo estable; sin ese control |
| Voz | seleccionable | estándar fija |

Límites conocidos: el cierre de `stop` espera la quietud de eventos del motor, que es una heurística provisional, no una prueba de
vaciado completo. La interfaz se probó solo con pruebas automáticas y una revisión del coordinador, no con una sesión de reunión.

## Perfil baseline candidato: comando

Instalá con los dos extras y conservalos en cada comando que use Moonshine:

```bash
uv sync --extra runtime --extra moonshine
```

Arranque del perfil:

```bash
.venv/bin/python -m runtime run --port 8877 --asr-backend moonshine-tiny --endpoint-ms 160 --tts-trim-lead-ms 30 --skip-silent-tail --tts-first-chunk-max-words 8 --tts-bind-conjunctions
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

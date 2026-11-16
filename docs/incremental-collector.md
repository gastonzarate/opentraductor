# Collector incremental causal, opt-in

Estado: integrado de forma opt-in en el replay, el runner y la CLI mediante `--incremental-collector`; los defaults permanecen intactos. No cambia el runtime, no ejecuta modelos, no crea alineaciones y no mide dispositivos físicos.

## Alcance

El observer recibe dos hooks del arnés:

1. `record_input_delivery` inmediatamente antes de entrar a `Adapter.push_audio`. El replay ya esperó hasta `t0 + end_s`; la frontera es el **final de la última muestra** del bloque, nunca su inicio.
2. `record_adapter_event` inmediatamente después de `clock.now_ns()` en `EventCollector`, antes de convertir el evento al formato histórico.

Cada stream mantiene una secuencia `event_seq` única que intercala entregas y eventos. `chunk_id` también lo asigna el arnés, solo para chunks TTS. `commit_id` y `source_span_s` continúan siendo claims del candidato; se guardan como `emission_id` y `source_span_claim`, pero no crean `RefUnit`, `AtomAssessment` ni `AlignmentResult`.

## Uso opt-in

```python
from benchmark.incremental.collector import IncrementalCollector
from benchmark.replay import EventCollector, run_stream

observer = IncrementalCollector(threshold_dbfs=-50.0)
collector = EventCollector(adapter, clock, observer=observer)
collector.start()
trace = await run_stream(adapter, clock, collector, stream_id, pcm, segments)
await collector.stop()

incremental_run = observer.to_incremental_run(stream_id)
observer.write_json(stream_id, "artifacts/diagnostic/collector.json")
```

Sin `observer=...`, `EventCollector` y `run_stream` conservan el comportamiento anterior.

## Runner y CLI opt-in

`benchmark run --incremental-collector` crea `incremental-collector.jsonl` dentro de `--out` y agrega el bloque `meta.incremental_collector` a `run.json`. El default es `False`: sin el flag no se instancia el observer ni se crea el JSONL. El runner acepta el flag solo para adaptadores `streaming` y exige `out_dir` para no perder evidencia.

El JSONL incluye warmup y streams medidos, propaga failure/timeout y conserva `alignment=null`, `ref_units=[]`. `meta.incremental_collector` separa contadores `warmup` y `measured`, incluidos expected, missing y aborted skips; el bloque `total` se rotula explícitamente `warmup_plus_measured`. Los outcomes del informe histórico siguen derivándose de los mismos `StreamTrace`; una muerte del iterador marca el trace y no puede convertirse en éxito.

El control público LibriVox tiene un manifest local de development, sin transcripción ni límites inventados:

- `artifacts/corpus-candidates/librivox-proverbios-01/manifest.jsonl`
- audio verificado por el manifest: `proverbios_01_rva_mono16k_pcm16.wav`

Comando preparado, **no ejecutado** todavía porque requiere lease de inferencia:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. .venv/bin/python -m benchmark run \
  --manifest artifacts/corpus-candidates/librivox-proverbios-01/manifest.jsonl \
  --audio-root artifacts/corpus-candidates/librivox-proverbios-01 \
  --split development \
  --adapter runtime.adapters.baseline:BaselineAdapter \
  --adapter-config '{"asr_backend":"moonshine-tiny","endpoint_silence_ms":160,"tts_trim_lead_margin_ms":30,"asr_skip_silent_tail":true,"tts_first_chunk_max_words":8,"tts_bind_conjunctions":true,"asr_normalize_text":false,"speculative_mt_tts":false}' \
  --repetitions 1 --warmup 0 --deadline-s 30 --call-timeout-s 60 --watchdog-stall-s 120 \
  --incremental-collector \
  --system-name diagnostic-librivox-incremental \
  --out artifacts/runtime-continuous/L3-librivox-incremental-dev
```

No se pasa `--continuous`: el WAV ya es un capítulo único de 258,48275 s y el runner no debe concatenarlo ni insertar gaps entre clips. El replay sí agrega su silencio final de censura después del archivo. El resultado será solo diagnóstico de causalidad, onset y cola; sin alineación independiente, las métricas semánticas quedan `unavailable`.

## Eventos de ciclo de vida (schema v2)

El adapter legacy conserva `Adapter.unit_events_supported=False`. En ese caso, ausencia de eventos significa instrumentación desconocida y `IncrementalRun.unit_done/unit_cancelled` quedan `None`. Un adapter que implementa el contrato declara `unit_events_supported=True`; entonces cero eventos válidos se conserva como `[]`, distinto de unknown.

Eventos standalone, sin retener el último chunk:

- `EventKind.UNIT_DONE`: `commit_id` no vacío, `revision≥0`, `unit_n_chunks≥0`, `unit_total_samples≥0`. Los enteros no aceptan booleanos. El adapter debe emitirlo una sola vez después de completar la unidad y contar únicamente chunks/muestras cuya emisión terminó correctamente.
- `EventKind.UNIT_CANCELLED`: `commit_id` no vacío, `revision≥0`, `before_emit_claim` booleano y `unit_reason` opcional. No acepta los contadores de DONE. Se emite una sola vez al transicionar a cancelada.
- `EventKind.ERROR` puede incluir `commit_id`, `revision` y `unit_reason`; sigue marcando fallo visible. No existe `UNIT_FAILED` en este hito.
- `TTS_CHUNK.revision` se copia a `CandidateFields` para correlacionar revisiones de un mismo `commit_id`.

El collector asigna `HarnessStamp` y convierte estos eventos a `UnitDoneRecord`/`UnitCancelledRecord`. Conserva duplicados para que `unit_closure` los detecte y rechaza eventos malformados como evidencia inválida. Una cancelación continúa siendo un claim del candidato estampado por el arnés; solo el contraste posterior con chunks/playout permite verificar si llegó antes de emisión.

`opentraductor-incremental-collector/v2` agrega los campos lifecycle a cada evento y el bloque `incremental.unit_events`. Los JSONL v1 existentes, incluido L3, no se reescriben.

## Validación en el borde

Entrada:

- PCM s16le: bytes no vacíos y cantidad par.
- 16 kHz mono por contrato de `Adapter.push_audio`.
- `start_s` finito, no negativo y contiguo con la entrega anterior.
- `end_s == start_s + samples/16000`, con media muestra de tolerancia de representación.
- El rango no puede superar `StreamTrace.input_duration_s`.
- Se registra `InputDelivery(seq, end_s, actual_ns)` aunque no haya salida.

Salida TTS:

- `tts_chunk` debe traer PCM s16le no vacío, cantidad par y sample rate entero positivo.
- Eventos que no sean TTS no pueden transportar PCM en este observer.
- RMS y primer bloque no silencioso se calculan con las utilidades existentes de `benchmark.audio` y el umbral configurado.
- El collector persiste tamaño, muestras, duración, SHA-256, RMS/onset y texto. Nunca retiene los bytes PCM.
- Un `EventKind.ERROR`, un PCM inválido, la muerte del iterador de eventos o un timeout quedan como fallo/censura en `to_incremental_run`; no se convierten en éxito. Si el iterador compartido muere, el replay marca los streams activos y rechaza como fallidos los streams posteriores que intenten reutilizar ese `EventCollector`.

`source_span_s` inválido se conserva como error de claim y no invalida las mediciones físicas. Los timestamps `claimed_ns` del candidato se guardan como metadato no confiable.

## Persistencia

`write_json` escribe de forma atómica un registro por stream con:

- `t0_ns`, `end_ns`, duración y modo `playback=simulated`;
- entregas de entrada y timeline causal;
- eventos con texto y metadatos PCM, sin PCM;
- chunks del contrato incremental accesibles mediante `to_incremental_run`;
- `alignment=None`, `ref_units=[]` y `semantic_metrics=unavailable_without_independent_alignment`.

No se exportan WAV, audio codificado ni contenido de entrada. Si otro componente necesita audio para un alineador futuro, deberá producir un artefacto separado y enlazarlo por hash; este collector no amplía esa superficie.

## Semántica temporal

- `recv_ns` sale del reloj del arnés al recibir cada evento.
- `delivered_until_s` es la frontera conocida en ese instante.
- La entrega y `SendRecord.actual_ns` se estampan dentro del coroutine `deliver`, inmediatamente antes de ejecutar `push_audio`; no se toma el reloj antes de que el timeout/scheduler programe ese coroutine. No existe un `await` entre el stamp, la actualización del frontier y la entrada al candidato. Los eventos recibidos durante una demora previa conservan el frontier anterior.
- La reproducción incremental sigue siendo simulada por recepción y duración PCM. No es CoreAudio, BlackHole ni audio escuchado por otra aplicación.
- Los eventos recibidos fuera de un stream activo se guardan como `stray_events`; no se adjudican a otro stream.

## Evidencia y límites

Pruebas dedicadas usan un adaptador falso sobre reloj virtual para eventos antes, durante y después de `push_audio`, empates de timestamp, dos bloques causales, salida audible/silenciosa, PCM malformado, evento error y timeout. Hay además un smoke corto con `RealClock`, sin dispositivo físico.

Este módulo valida instrumentación causal y persistencia sin audio. No valida traducción, naturalidad, conversación sostenida ni el objetivo de 300 ms. Sin anotación independiente, toda métrica semántica debe permanecer `unavailable`.

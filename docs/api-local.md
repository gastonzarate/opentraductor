# API HTTP local (runtime/http_control.py)

Contrato entre el motor y la UI. Solo escucha en 127.0.0.1 y pide el token de sesión (`Authorization: Bearer …`).
Este archivo documenta cambios de comportamiento y rutas nuevas; la lista completa está en `http_control.py`.

## /api/license (sin licencia desde T-70)

El repo es open source: no hay prueba, clave ni compuerta en `/api/start`, y `runtime/license.py` ya no existe.
`GET /api/license` devuelve `{"state": "disabled", "can_use": true}` solo para que la UI web oculte su tarjeta
(se quita junto con esa tarjeta). En la tienda (`OPENTRADUCTOR_EDITION=store`) da 404; `POST /api/license/activate` da 404 siempre.

## /events: eventos de frase (T-55)

Un `seg_id` identifica una sola frase.

| Evento | Campos | Cambio |
|---|---|---|
| `partial` | `seg_id`, `text_es`, `from_seg_id?` | `text_es` nunca llega vacío. `from_seg_id` solo lo mandaba el modo incremental (retirado en T-70); la UI puede seguir aceptándolo. |
| `segment` | `seg_id`, `text_es`, `text_es_raw`, `text_en`, `ts` | Sin cambios. Solo cierra la fila de su `seg_id`. |
| `discarded` (nuevo) | `seg_id`, `text_es`, `reason` | `reason` ∈ `short` (bajo `min_speech_ms`), `empty` (ASR final vacío). `cancelled` ya no se emite (era del modo incremental, retirado en T-70). La fila queda visible como descartada. Si no hay fila y `text_es` es `""`, no se muestra nada. Clientes viejos lo ignoran. |

Con `mt_stream_first_chunk` (T-58, apagado por defecto, solo Qwen) el audio del 1.er fragmento de una frase puede empezar a sonar antes de que llegue su `segment` (el `text_en` completo llega igual, unos ms después). Ese 1.er `tts_chunk` trae `planned_text_en` nulo; los siguientes traen el texto completo. Las rutas y los campos no cambian. `describe` suma `mt_stream` (segments, head_early, mismatch, lead_ms, tts_in_mt_ms).

Con `spec_on_silence` (T-73, apagado por defecto, env `OPENTRADUCTOR_SPEC_ON_SILENCE=1`) el motor emite un evento interno `spec_result` por frase antes de su `mt_final`: `{stream_id, seg_id, spec: hit|miss|none|error|cancelled, launched, wasted_ms, claim_wait_ms}`. Es para métricas del banco de medición; la UI puede ignorarlo. `describe.spec` suma replaced, cancel_voice, from_finish, hit_norm, late_ms y on_silence. `mt_stream_tts` / `OPENTRADUCTOR_MT_STREAM_TTS` es alias de `mt_stream_first_chunk`. `OPENTRADUCTOR_TTS_FIRST_WORDS=N` (T-67) fuerza el largo del 1.er fragmento de Kokoro (`tts_first_chunk_max_words`, default 2 desde el A/B final `artifacts/t67-final`; 0 = solo por puntuación). Desde ese A/B también son default `asr_process` (Moonshine en proceso aparte; env `OPENTRADUCTOR_ASR_PROCESS=0` lo apaga) y `switch_interval_ms` 1 (env `OPENTRADUCTOR_SWITCH_INTERVAL_MS=off` lo apaga). La API no cambia.

## Retraso de la voz en charla continua (T-92, BUG-9)

Rutas sin cambios. Las métricas del estado suman `playback_skipped_s`: segundos de voz en inglés vieja que el Player saltó sin reproducir (frases enteras) porque la cola pasó de 6 s (`player_skip_old_s`, env `OPENTRADUCTOR_PLAYER_SKIP_OLD_S`). `playback_dropped_ms` sigue existiendo y ahora solo cuenta una frase nueva que sola no entra en 60 s. Los subtítulos (`segment`) no cambian: las frases saltadas y las frases que son solo muletilla («eh», «¿se entiende?», «got it?») siguen llegando como `segment`, pero no se dicen. Interruptor general `tts_catchup` (env `OPENTRADUCTOR_TTS_CATCHUP=0` vuelve al camino anterior: velocidad fija, se dicen las muletillas, el Player descarta lo nuevo pasados 60 s). Umbral de salto `player_skip_old_s` = 8 (T-93). T-93, recorte de la frase en curso: `player_cut_cur_s` = 5 (env `OPENTRADUCTOR_PLAYER_CUT_CUR_S`): cuando llega una frase nueva y la cola sigue por encima de ese valor, se descartan las cláusulas aún no empezadas de la frase que suena (la que suena termina entera). Nunca se saltan ni recortan frases cuyo inglés tiene cifras o nombres propios. Métricas nuevas: `playback_cut_cur_s` (incluida en `playback_skipped_s`) y `playback_skipped_segs`, una lista con un registro por salto o recorte: `seg_id` (el mismo de `segment` y del historial), `text_en`, `s` (segundos quitados), `reason` (`skip_old` | `cut_cur`), `queue_s` (cola tras quitar) y `t` (epoch s). Lo mismo sale en el log `opentraductor.player`. Silencio de cada chunk de Kokoro recortado a ≤ 80 ms al inicio y al final (`tts_trim_keep_ms`, env `OPENTRADUCTOR_TTS_TRIM_KEEP_MS`); el total quitado sale en `metrics.tts_trimmed_silence_s` del status (segundos acumulados del motor; `null` sin motor cargado). En estas env, vacío o negativo = apagado. Defaults en `runtime/config.py`: `tts_speed_adaptive` (Kokoro a 1,0× con ≤1 s de cola y hasta 1,30× con ≥3 s, T-93; env `OPENTRADUCTOR_TTS_SPEED_ADAPTIVE=0` lo apaga; `OPENTRADUCTOR_TTS_SPEED_LO_S`, `_HI_S` y `_MAX` cambian la rampa) y `tts_skip_fillers` (env `OPENTRADUCTOR_TTS_SKIP_FILLERS=0`).

## Sin descartes y voz simultánea (T-101, T-102)

Rutas sin cambios. Métrica nueva en `metrics` del status: `tts_max_speed`, la velocidad máxima de la voz que usó el motor en la sesión (1.0 = nunca aceleró; `null` sin motor). Con `no_drop` (default, T-101) `overflow_count`, `dropped_ms`, `playback_skipped_s` y `playback_dropped_ms` deben quedar en 0: el atraso se compensa acelerando la voz hasta 1,5×.
Modo «voz simultánea» (T-102, encendido por defecto; `OPENTRADUCTOR_SIMUL_VOICE=0` lo apaga, `OPENTRADUCTOR_SIMUL_VOICE_K`, default 6): cada trozo de la frase sale con k palabras confirmadas, no solo en comas y nexos. Los eventos no cambian: los trozos siguen saliendo como `simul_clause` y la frase cierra con un único `segment`.
Modo «voz al ritmo del orador» (T-103, encendido por defecto desde el 25/11; `OPENTRADUCTOR_PACE_VOICE=0` lo apaga; `OPENTRADUCTOR_PACE_MAX` tope, default 1,3; `OPENTRADUCTOR_PACE_CPS`, default 11,4): la velocidad de cada frase se ajusta para que dure lo que duró el habla, y se usa la mayor entre esta y la de T-101. Rutas y eventos sin cambios.

## Tope al atraso con la Mac saturada (T-105, BUG-10)

Rutas y eventos sin cambios. Métricas nuevas en `metrics` del status: `asr_skipped_s` (segundos de audio de frases enteras que el reconocedor saltó sin reconocer), `asr_skip_events` (tandas de salto) y `asr_skip_segs` (frases saltadas); `null` sin motor. Cuando al empezar una frase el audio pendiente del reconocedor (`asr_backlog_s`) pasa de 20 s (`asr_skip_old_s`, env `OPENTRADUCTOR_ASR_SKIP_OLD_S`; vacío, 0 o negativo = apagado), esa frase y las siguientes se saltan enteras hasta que el pendiente baja a 3 s (`asr_skip_target_s`, env `OPENTRADUCTOR_ASR_SKIP_TARGET_S`). Una frase saltada no produce `asr_partial`, `segment` ni voz; nunca se corta una frase a medias. Encendido por defecto. Con carga normal no salta nada (charla 1: `asr_backlog_s` máx 1,4 s). Cada tanda queda en el log del motor (`[asr] T-105 ...`). Es la única pérdida que sigue existiendo con `no_drop`: `overflow_count`, `dropped_ms` y los contadores del Player siguen en 0.

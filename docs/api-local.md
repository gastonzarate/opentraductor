# API HTTP local (runtime/http_control.py)

Contrato entre el motor y la UI. Solo escucha en 127.0.0.1 y pide el token de sesión (`Authorization: Bearer …`).
Este archivo documenta cambios de comportamiento y rutas nuevas; la lista completa está en `http_control.py`.

## /api/license (sin licencia desde T-70)

El repo es open source: no hay prueba, clave ni compuerta en `/api/start`, y `runtime/license.py` ya no existe.
`GET /api/license` devuelve `{"state": "disabled", "can_use": true}` solo para que la UI web oculte su tarjeta
(se quita junto con esa tarjeta). En la tienda (`OPENTRADUCTOR_EDITION=store`) da 404; `POST /api/license/activate` da 404 siempre.

## /events: eventos de frase (T-55)

Contrato completo: `memory/decisiones/t55-contrato-eventos-frase.md`. Un `seg_id` identifica una sola frase.

| Evento | Campos | Cambio |
|---|---|---|
| `partial` | `seg_id`, `text_es`, `from_seg_id?` | `text_es` nunca llega vacío. `from_seg_id` solo lo mandaba el modo incremental (retirado en T-70); la UI puede seguir aceptándolo. |
| `segment` | `seg_id`, `text_es`, `text_es_raw`, `text_en`, `ts` | Sin cambios. Solo cierra la fila de su `seg_id`. |
| `discarded` (nuevo) | `seg_id`, `text_es`, `reason` | `reason` ∈ `short` (bajo `min_speech_ms`), `empty` (ASR final vacío). `cancelled` ya no se emite (era del modo incremental, retirado en T-70). La fila queda visible como descartada. Si no hay fila y `text_es` es `""`, no se muestra nada. Clientes viejos lo ignoran. |

Con `mt_stream_first_chunk` (T-58, apagado por defecto, solo Qwen) el audio del 1.er fragmento de una frase puede empezar a sonar antes de que llegue su `segment` (el `text_en` completo llega igual, unos ms después). Ese 1.er `tts_chunk` trae `planned_text_en` nulo; los siguientes traen el texto completo. Las rutas y los campos no cambian. `describe` suma `mt_stream` (segments, head_early, mismatch, lead_ms, tts_in_mt_ms).

Con `spec_on_silence` (T-73, apagado por defecto, env `OPENTRADUCTOR_SPEC_ON_SILENCE=1`) el motor emite un evento interno `spec_result` por frase antes de su `mt_final`: `{stream_id, seg_id, spec: hit|miss|none|error|cancelled, launched, wasted_ms, claim_wait_ms}`. Es para métricas de @bench; la UI puede ignorarlo. `describe.spec` suma replaced, cancel_voice, from_finish, hit_norm, late_ms y on_silence. `mt_stream_tts` / `OPENTRADUCTOR_MT_STREAM_TTS` es alias de `mt_stream_first_chunk`. `OPENTRADUCTOR_TTS_FIRST_WORDS=N` (T-67) fuerza el largo del 1.er fragmento de Kokoro (`tts_first_chunk_max_words`, default 2 desde el A/B final `artifacts/t67-final`; 0 = solo por puntuación). Desde ese A/B también son default `asr_process` (Moonshine en proceso aparte; env `OPENTRADUCTOR_ASR_PROCESS=0` lo apaga) y `switch_interval_ms` 1 (env `OPENTRADUCTOR_SWITCH_INTERVAL_MS=off` lo apaga). La API no cambia.

## Retraso de la voz en charla continua (T-92, BUG-9)

Rutas sin cambios. Las métricas del estado suman `playback_skipped_s`: segundos de voz en inglés vieja que el Player saltó sin reproducir (frases enteras) porque la cola pasó de 6 s (`player_skip_old_s`, env `OPENTRADUCTOR_PLAYER_SKIP_OLD_S`). `playback_dropped_ms` sigue existiendo y ahora solo cuenta una frase nueva que sola no entra en 60 s. Los subtítulos (`segment`) no cambian: las frases saltadas y las frases que son solo muletilla («eh», «¿se entiende?», «got it?») siguen llegando como `segment`, pero no se dicen. Interruptor general `tts_catchup` (env `OPENTRADUCTOR_TTS_CATCHUP=0` vuelve al camino anterior: velocidad fija, se dicen las muletillas, el Player descarta lo nuevo pasados 60 s). Umbral de salto `player_skip_old_s` = 6 (T-93). T-93, recorte de la frase en curso: `player_cut_cur_s` (apagado por defecto; env `OPENTRADUCTOR_PLAYER_CUT_CUR_S=3` lo prende): cuando llega una frase nueva y la cola sigue por encima de ese valor, se descartan las cláusulas aún no empezadas de la frase que suena (la que suena termina entera). Métrica nueva `playback_cut_cur_s` (incluida en `playback_skipped_s`). En las dos env, vacío o negativo = apagado. Defaults en `runtime/config.py`: `tts_speed_adaptive` (Kokoro a 1,0× con ≤2 s de cola y hasta 1,30× con ≥5 s, T-93; env `OPENTRADUCTOR_TTS_SPEED_ADAPTIVE=0` lo apaga; `OPENTRADUCTOR_TTS_SPEED_LO_S`, `_HI_S` y `_MAX` cambian la rampa) y `tts_skip_fillers` (env `OPENTRADUCTOR_TTS_SKIP_FILLERS=0`).

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

Con `spec_on_silence` (T-73, apagado por defecto, env `OPENTRADUCTOR_SPEC_ON_SILENCE=1`) el motor emite un evento interno `spec_result` por frase antes de su `mt_final`: `{stream_id, seg_id, spec: hit|miss|none|error|cancelled, launched, wasted_ms, claim_wait_ms}`. Es para métricas de @bench; la UI puede ignorarlo. `describe.spec` suma replaced, cancel_voice, from_finish, hit_norm, late_ms y on_silence. `mt_stream_tts` / `OPENTRADUCTOR_MT_STREAM_TTS` es alias de `mt_stream_first_chunk`. `OPENTRADUCTOR_TTS_FIRST_WORDS=N` (T-67, solo env, para A/B) fuerza el largo del 1.er fragmento de Kokoro (`tts_first_chunk_max_words`; 0 = solo por puntuación); la API no cambia.

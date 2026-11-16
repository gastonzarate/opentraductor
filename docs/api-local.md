# API HTTP local (runtime/http_control.py)

Contrato entre el motor y la UI. Solo escucha en 127.0.0.1 y pide el token de sesión (`Authorization: Bearer …`).
Este archivo documenta cambios de comportamiento y rutas nuevas; la lista completa está en `http_control.py`.

## /api/license y /api/license/activate

| Edición | `GET /api/license` | `POST /api/license/activate` |
|---|---|---|
| Directa (Lemon Squeezy) | estado de la prueba o la licencia | activa una clave |
| Sin gestor de licencia (pruebas, desarrollo) | `{"state": "disabled", "can_use": true}` | 404 |
| Tienda (`OPENTRADUCTOR_EDITION=store`) | **404 `not_found`** | **404 `not_found`** |

En la edición tienda el motor no importa `runtime/license.py` (sign_store.sh lo borra del bundle, guía 2.4.5(vi))
y `/api/start` no tiene compuerta de licencia: cobra la App Store. La UI nativa de la tienda no llama a estas rutas.

## /events: eventos de frase (T-55)

Contrato completo: `memory/decisiones/t55-contrato-eventos-frase.md`. Un `seg_id` identifica una sola frase.

| Evento | Campos | Cambio |
|---|---|---|
| `partial` | `seg_id`, `text_es`, `from_seg_id?` | `text_es` nunca llega vacío. En modo incremental la fila en vivo es `"{sid}:live:{n}"` (una por frase). El primer `partial` con el id de commit trae `from_seg_id` = id live que reemplaza: la UI reasigna esa fila, no crea otra. |
| `segment` | `seg_id`, `text_es`, `text_es_raw`, `text_en`, `ts` | Sin cambios. Solo cierra la fila de su `seg_id`. |
| `discarded` (nuevo) | `seg_id`, `text_es`, `reason` | `reason` ∈ `short` (bajo `min_speech_ms`), `empty` (ASR final vacío), `cancelled` (unidad incremental cancelada o sin audio). La fila queda visible como descartada. Si no hay fila y `text_es` es `""`, no se muestra nada. Clientes viejos lo ignoran. |

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

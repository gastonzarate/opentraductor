"""Prueba gratis de 7 días y activación de licencia con Lemon Squeezy.

Privacidad: la ÚNICA conexión de red del runtime tras el ``setup`` es ``activate()``. Envía la clave de licencia y el
nombre de esta Mac (``instance_name``) por HTTPS a la API de licencias de Lemon Squeezy. No envía audio, textos,
historial ni identificadores del equipo. Tras activar, el estado se guarda en local y la app funciona sin red
(no hay revalidación periódica).

Estado en disco (``license.json``, permisos 0600):
  first_use   epoch de la primera vez que se abrió la app (empieza la prueba)
  last_seen   mayor epoch observado: si el reloj se atrasa, se usa este valor (evita alargar la prueba con la fecha)
  license     {key, instance_id, instance_name, activated_at} una vez activada

Límites conocidos (decididos con un precio de USD 19): borrar ``license.json`` reinicia la prueba y no hay
revocación remota después de activar. No se agregan defensas porque no valen su costo.
"""
from __future__ import annotations

import json
import math
import os
import socket
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .http_control import ServiceError

TRIAL_DAYS = 7
ACTIVATE_URL = "https://api.lemonsqueezy.com/v1/licenses/activate"
HTTP_TIMEOUT_S = 10.0
MAX_KEY_LEN = 200
MAX_RESPONSE_BYTES = 64 * 1024

# Completar al crear la tienda en Lemon Squeezy (Settings > Stores, y el ID del producto).
# Con None no se comprueba: una clave válida de CUALQUIER producto de la cuenta activaría la app.
# Hay que fijarlos antes de publicar el .dmg (pendiente en T-3).
EXPECTED_STORE_ID: int | None = None
EXPECTED_PRODUCT_ID: int | None = None

NETWORK_NOTICE = (
    "La activación es la única conexión a internet de la app después de instalarla. "
    "Envía tu clave de licencia y el nombre de esta Mac a Lemon Squeezy, una sola vez. "
    "No envía audio ni texto."
)


def default_license_path() -> Path:
    base = Path(os.environ.get("OPENTRADUCTOR_DATA", "~/.local/share/opentraductor")).expanduser()
    return base / "license.json"


def mac_name() -> str:
    name = socket.gethostname().strip() or "Mac"
    return name[:100]


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


def _hint(key: str) -> str:
    return "…" + key[-4:] if len(key) >= 8 else "…"


Opener = Callable[[urllib.request.Request, float], Any]


def _default_open(req: urllib.request.Request, timeout: float):
    return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310 - URL fija https


class LicenseManager:
    def __init__(
        self,
        path: Path | str | None = None,
        *,
        clock: Callable[[], float] = time.time,
        opener: Opener = _default_open,
        instance_name: str | None = None,
    ) -> None:
        self._path = Path(path) if path else default_license_path()
        self._clock = clock
        self._open = opener
        self._instance_name = instance_name or mac_name()
        self._lock = threading.Lock()

    # ------------------------------------------------------------ disco
    def _load(self) -> dict:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save(self, data: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, prefix=".license-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
            os.chmod(tmp, 0o600)
            os.replace(tmp, self._path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _touch(self) -> tuple[dict, float]:
        """Carga el estado, fija la primera vez y el último instante visto. Llamar con el lock tomado."""
        data = self._load()
        real = self._clock()
        seen = data.get("last_seen")
        now = max(real, float(seen)) if isinstance(seen, (int, float)) else real
        changed = False
        if not isinstance(data.get("first_use"), (int, float)):
            data["first_use"] = now
            changed = True
        if now > float(data.get("last_seen") or 0):
            data["last_seen"] = now
            changed = True
        if changed:
            try:
                self._save(data)
            except OSError:
                pass  # sin disco no se puede persistir: la prueba sigue contando en esta ejecución
        return data, now

    # ------------------------------------------------------------ estado
    def status(self) -> dict:
        with self._lock:
            data, now = self._touch()
        lic = data.get("license") if isinstance(data.get("license"), dict) else None
        ends = float(data["first_use"]) + TRIAL_DAYS * 86400
        left = max(0.0, ends - now)
        days_left = math.ceil(left / 86400) if left > 0 else 0
        if lic:
            state = "licensed"
        elif left > 0:
            state = "trial"
        else:
            state = "expired"
        return {
            "state": state,
            "licensed": state == "licensed",
            "can_use": state != "expired",
            "trial_days": TRIAL_DAYS,
            "trial_days_left": days_left,
            "trial_ends_at": _iso(ends),
            "key_hint": _hint(str(lic.get("key", ""))) if lic else None,
            "instance_name": self._instance_name,
            "network_notice": NETWORK_NOTICE,
        }

    def allows_use(self) -> bool:
        return self.status()["can_use"]

    # -------------------------------------------------------- activación
    def activate(self, key: Any) -> dict:
        if not isinstance(key, str):
            raise ServiceError("invalid_key", "Ingresá la clave de licencia.", 400)
        key = key.strip()
        if not key or len(key) > MAX_KEY_LEN or any(ord(c) < 33 or ord(c) > 126 for c in key):
            raise ServiceError("invalid_key", "La clave no tiene un formato válido. Copiala de nuevo del mail de compra.", 400)
        with self._lock:
            data, _ = self._touch()
            if isinstance(data.get("license"), dict):
                raise ServiceError("already_active", "Esta Mac ya tiene una licencia activada.", 409)
        body = self._post(key)  # sin el lock: la red puede tardar hasta HTTP_TIMEOUT_S
        lk = body.get("license_key") if isinstance(body.get("license_key"), dict) else {}
        meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
        inst = body.get("instance") if isinstance(body.get("instance"), dict) else {}
        if not body.get("activated"):
            limit, used = lk.get("activation_limit"), lk.get("activation_usage")
            if isinstance(limit, int) and isinstance(used, int) and used >= limit:
                raise ServiceError(
                    "activation_limit",
                    f"Esta clave ya está activada en {limit} Macs, que es el máximo. Escribinos para liberar una.",
                    409,
                )
            raise ServiceError("license_rejected", "La clave no es válida o fue desactivada. Revisá que esté completa.", 403)
        if lk.get("status") not in (None, "active"):
            raise ServiceError("license_rejected", "La clave no está activa.", 403)
        if EXPECTED_STORE_ID is not None and meta.get("store_id") != EXPECTED_STORE_ID:
            raise ServiceError("license_rejected", "Esta clave no corresponde a opentraductor.", 403)
        if EXPECTED_PRODUCT_ID is not None and meta.get("product_id") != EXPECTED_PRODUCT_ID:
            raise ServiceError("license_rejected", "Esta clave no corresponde a opentraductor.", 403)
        with self._lock:
            data, now = self._touch()
            data["license"] = {
                "key": key,
                "instance_id": str(inst.get("id", "")),
                "instance_name": self._instance_name,
                "activated_at": _iso(now),
            }
            try:
                self._save(data)
            except OSError as exc:
                raise ServiceError("storage", "La clave es válida, pero no se pudo guardar en esta Mac. Revisá el espacio y los permisos.", 500) from exc
        return self.status()

    def _post(self, key: str) -> dict:
        form = urllib.parse.urlencode({"license_key": key, "instance_name": self._instance_name}).encode("ascii")
        req = urllib.request.Request(
            ACTIVATE_URL,
            data=form,
            method="POST",
            headers={"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"},
        )
        try:
            with self._open(req, HTTP_TIMEOUT_S) as resp:
                raw = resp.read(MAX_RESPONSE_BYTES)
        except urllib.error.HTTPError as exc:  # 400/404: la API explica el motivo en el cuerpo
            try:
                raw = exc.read(MAX_RESPONSE_BYTES)
            except Exception:  # noqa: BLE001
                raw = b""
            if exc.code >= 500 or exc.code == 429:
                raise ServiceError("network", "El servicio de licencias no responde ahora. Reintentá en unos minutos.", 502) from exc
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise ServiceError(
                "network",
                "No se pudo conectar para activar la licencia. Revisá tu conexión a internet y reintentá. "
                "Solo se envían la clave y el nombre de esta Mac.",
                502,
            ) from exc
        try:
            body = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            body = None
        if not isinstance(body, dict):
            raise ServiceError("network", "Respuesta inesperada del servicio de licencias. Reintentá más tarde.", 502)
        return body

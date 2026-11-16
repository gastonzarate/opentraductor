"""Capa HTTP local de opentraductor (contrato v0 acordado con la UI).

Solo expone el control del servicio. No abre micrófono, no carga modelos, no
toca el pipeline: todo eso lo hace el objeto ``service`` que recibe el
constructor.

Seguridad:
- Bind a 127.0.0.1. Host y Origin deben ser 127.0.0.1 o localhost con el puerto propio.
- Token por ejecución. Se inyecta en index.html como <meta name="ot-token">.
  /api/* lo exige en el header X-OT-Token. /api/events (SSE) también acepta ?token=,
  porque EventSource no puede enviar headers.
- Rutas estáticas en lista fija: index.html, app.js, styles.css.
- Cuerpos JSON con tope de tamaño. Errores siempre como {"error": {"code", "message"}}.

Interfaz que debe cumplir ``service`` (todos los métodos son síncronos):
    status() -> dict
    devices() -> dict                      {"inputs": [...], "outputs": [...]}
    voices() -> dict                       {"voices": [{"id", "label"}]}
    configure(values: dict) -> dict        devuelve el status actualizado
    start() -> dict | None                 status (o None: respuesta {"ok": true})
    stop() -> dict | None
    history(limit: int, offset: int) -> dict   {"items": [...], "total": int}
    export_history(format: str) -> str     "json" o "txt"
    delete_history(session_id: str | None) -> int   None borra todo
    subscribe(callback: Callable[[dict], None]) -> handle
    unsubscribe(handle) -> None

Eventos que el servicio publica con ``callback``: dicts con "type", por ejemplo
{"type": "state", "state": "running"}. El callback se llama desde hilos del
servicio. Solo encola y vuelve, sin bloquear.

Errores de negocio: el servicio lanza ``ServiceError(code, message, status)``.
``ConflictError(message)`` responde 409 con code 'conflict'. ``ValueError(message)``
responde 400 con code 'invalid' y ese mensaje. Cualquier otra excepción responde
500 con un mensaje genérico, sin detalles.
"""
from __future__ import annotations

import hmac
import html
import json
import queue
import secrets
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlsplit

UI_DIR = Path(__file__).resolve().parent.parent / "ui"
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/setup.js": ("setup.js", "text/javascript; charset=utf-8"),
}
MAX_BODY_BYTES = 16 * 1024
SSE_QUEUE_MAX = 512
SSE_KEEPALIVE_S = 15.0
SSE_POLL_S = 0.5  # cada cuánto el hilo SSE revisa la señal de parada
EXPORT_FORMATS = ("json", "txt")
META_TAG = '<meta name="ot-token" content="{token}">'


class ServiceError(Exception):
    """Error de negocio del servicio. Se traduce a JSON con su código y estado HTTP."""

    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


class ConflictError(ServiceError):
    """Cambio no permitido con la sesión en marcha (HTTP 409, code 'conflict')."""

    def __init__(self, message: str):
        super().__init__("conflict", message, 409)


class _Client:
    def __init__(self) -> None:
        self.queue: queue.Queue[dict | None] = queue.Queue(maxsize=SSE_QUEUE_MAX)


class ControlServer:
    """Servidor HTTP local. Arranca en un hilo y se cierra con ``stop()``."""

    def __init__(
        self,
        service: Any,
        port: int = 8877,
        ui_dir: Path | None = None,
        token: str | None = None,
        license: Any | None = None,  # noqa: A002 - LicenseManager; None = sin prueba ni licencia (pruebas, modo desarrollo)
        store: bool = False,  # edición Mac App Store: /api/license* no existe (404)
    ) -> None:
        self._service = service
        self._license = None if store else license
        self._store = store
        self._ui_dir = Path(ui_dir) if ui_dir else UI_DIR
        self._token = token or secrets.token_urlsafe(32)
        self._lock = threading.Lock()
        self._clients: list[_Client] = []
        self._httpd = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
        self._httpd.daemon_threads = True
        self._httpd.control = self  # type: ignore[attr-defined]
        self._port = self._httpd.server_address[1]
        self._thread: threading.Thread | None = None
        self._sub_handle: Any = None
        self._stopped = False

    # ------------------------------------------------------------ propiedades
    @property
    def port(self) -> int:
        return self._port

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._port}/"

    @property
    def token(self) -> str:
        return self._token

    def allowed_hosts(self) -> set[str]:
        return {f"127.0.0.1:{self._port}", f"localhost:{self._port}"}

    def allowed_origins(self) -> set[str]:
        return {f"http://127.0.0.1:{self._port}", f"http://localhost:{self._port}"}

    # ------------------------------------------------------------- ciclo
    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("el servidor ya está en marcha")
        self._sub_handle = self._service.subscribe(self._broadcast)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, name="ot-http", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self._sub_handle is not None:
            try:
                self._service.unsubscribe(self._sub_handle)
            finally:
                self._sub_handle = None
        with self._lock:
            for c in self._clients:
                self._put_end(c)
        if self._thread is not None:
            self._httpd.shutdown()
            self._thread.join(timeout=5)
        self._httpd.server_close()

    def __enter__(self) -> "ControlServer":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # ------------------------------------------------------------- SSE
    def _add_client(self) -> _Client:
        c = _Client()
        with self._lock:
            self._clients.append(c)
        return c

    def _remove_client(self, c: _Client) -> None:
        with self._lock:
            if c in self._clients:
                self._clients.remove(c)

    def _broadcast(self, event: dict) -> None:
        """Llamado desde el servicio. No bloquea: si un cliente va atrasado, descarta lo más viejo."""
        with self._lock:
            clients = list(self._clients)
        for c in clients:
            try:
                c.queue.put_nowait(event)
            except queue.Full:
                try:
                    c.queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    c.queue.put_nowait(event)
                except queue.Full:
                    pass

    @staticmethod
    def _put_end(c: _Client) -> None:
        """Deja la señal de fin aunque la cola esté llena: descarta lo más viejo si hace falta."""
        try:
            c.queue.put_nowait(None)
            return
        except queue.Full:
            pass
        try:
            c.queue.get_nowait()
        except queue.Empty:
            pass
        try:
            c.queue.put_nowait(None)
        except queue.Full:
            pass


class _Handler(BaseHTTPRequestHandler):
    server_version = "opentraductor"
    sys_version = ""
    protocol_version = "HTTP/1.1"

    # Sin registro por stderr: podría mostrar rutas o tokens.
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    @property
    def ctl(self) -> ControlServer:
        return self.server.control  # type: ignore[attr-defined]

    # ----------------------------------------------------------- envío
    def _send(self, status: int, body: bytes, content_type: str, extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'",
        )
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _error(self, status: int, code: str, message: str) -> None:
        # El cuerpo de la petición puede no estar leído: no reusar la conexión.
        self.close_connection = True
        self._json(status, {"error": {"code": code, "message": message}})

    # ----------------------------------------------------- validaciones
    def _check_host_origin(self) -> bool:
        host = self.headers.get("Host", "")
        if host not in self.ctl.allowed_hosts():
            self._error(HTTPStatus.FORBIDDEN, "forbidden_host", "Host no permitido.")
            return False
        origin = self.headers.get("Origin")
        if origin is not None and origin not in self.ctl.allowed_origins():
            self._error(HTTPStatus.FORBIDDEN, "forbidden_origin", "Origen no permitido.")
            return False
        return True

    def _token_ok(self, supplied: str | None) -> bool:
        return supplied is not None and hmac.compare_digest(supplied, self.ctl.token)

    def _require_token(self, query: dict[str, list[str]] | None = None) -> bool:
        supplied = self.headers.get("X-OT-Token")
        if supplied is None and query is not None:
            supplied = (query.get("token") or [None])[0]
        if not self._token_ok(supplied):
            self._error(HTTPStatus.UNAUTHORIZED, "unauthorized", "Falta o es inválido el token de sesión.")
            return False
        return True

    def _consume_body(self) -> bool:
        """Lee el cuerpo completo de la petición, aunque la ruta no lo use.

        Así la conexión keep-alive queda alineada para la siguiente petición.
        Devuelve False si ya respondió con error (Content-Length inválido o demasiado grande).
        """
        self._body = b""
        length_header = self.headers.get("Content-Length")
        if length_header is None:
            return True
        try:
            length = int(length_header)
        except ValueError:
            self._error(HTTPStatus.BAD_REQUEST, "bad_length", "Content-Length inválido.")
            return False
        if length < 0:
            self._error(HTTPStatus.BAD_REQUEST, "bad_length", "Content-Length inválido.")
            return False
        if length > MAX_BODY_BYTES:
            self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "too_large", "Cuerpo demasiado grande.")
            return False
        self._body = self.rfile.read(length) if length else b""
        return True

    def _read_json_body(self) -> Any | None:
        if self.headers.get("Content-Length") is None:
            self._error(HTTPStatus.LENGTH_REQUIRED, "length_required", "Falta Content-Length.")
            return None
        raw = self._body or b""
        if not raw.strip():
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._error(HTTPStatus.BAD_REQUEST, "bad_json", "El cuerpo no es JSON válido.")
            return None

    # ----------------------------------------------------------- rutas
    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_HEAD(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def do_DELETE(self) -> None:  # noqa: N802
        self._dispatch("DELETE")

    def do_PUT(self) -> None:  # noqa: N802
        self._error(HTTPStatus.METHOD_NOT_ALLOWED, "method_not_allowed", "Método no permitido.")

    do_PATCH = do_PUT  # noqa: N815

    def _dispatch(self, method: str) -> None:
        try:
            if not self._consume_body():
                return
            if not self._check_host_origin():
                return
            parts = urlsplit(self.path)
            path = unquote(parts.path)
            query = parse_qs(parts.query, keep_blank_values=False)

            if path in STATIC_FILES:
                if method != "GET":
                    return self._error(HTTPStatus.METHOD_NOT_ALLOWED, "method_not_allowed", "Método no permitido.")
                return self._serve_static(path)

            if not path.startswith("/api/"):
                return self._error(HTTPStatus.NOT_FOUND, "not_found", "Ruta no encontrada.")

            if path == "/api/events":
                if method != "GET":
                    return self._error(HTTPStatus.METHOD_NOT_ALLOWED, "method_not_allowed", "Método no permitido.")
                if not self._require_token(query):
                    return
                return self._serve_sse()

            if not self._require_token():
                return

            return self._api(method, path, query)
        except ServiceError as exc:
            self._error(exc.status, exc.code, exc.message)
        except ValueError as exc:
            # Valor inválido que el servicio rechaza con un mensaje para el usuario.
            self._error(HTTPStatus.BAD_REQUEST, "invalid", str(exc) or "Valor inválido.")
        except Exception:  # noqa: BLE001 - nunca filtrar detalles internos
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal", "Error interno del servicio.")

    def _api(self, method: str, path: str, query: dict[str, list[str]]) -> None:
        svc = self.ctl._service
        if path == "/api/devices/refresh":  # acción explícita: GET /api/devices nunca toca PortAudio
            if method != "POST":
                return self._error(HTTPStatus.METHOD_NOT_ALLOWED, "method_not_allowed", "Método no permitido.")
            return self._json(HTTPStatus.OK, svc.refresh_devices())
        if path == "/api/virtual-mic":  # solo lectura: usa la lista de dispositivos ya enumerada
            if method != "GET":
                return self._error(HTTPStatus.METHOD_NOT_ALLOWED, "method_not_allowed", "Método no permitido.")
            from . import virtual_mic

            return self._json(HTTPStatus.OK, virtual_mic.detect(svc.devices()))

        if path in ("/api/license", "/api/license/activate"):
            if self.ctl._store:  # tienda: sin licencia propia (guía 2.4.5(vi)); la ruta no existe
                return self._error(HTTPStatus.NOT_FOUND, "not_found", "Ruta no encontrada.")
            lic = self.ctl._license
            if lic is None:  # sin gestor de licencia (pruebas o desarrollo): todo permitido
                if path == "/api/license":
                    return self._json(HTTPStatus.OK, {"state": "disabled", "licensed": False, "can_use": True})
                return self._error(HTTPStatus.NOT_FOUND, "not_found", "Ruta no encontrada.")
            if path == "/api/license":
                if method != "GET":
                    return self._error(HTTPStatus.METHOD_NOT_ALLOWED, "method_not_allowed", "Método no permitido.")
                return self._json(HTTPStatus.OK, lic.status())
            if method != "POST":
                return self._error(HTTPStatus.METHOD_NOT_ALLOWED, "method_not_allowed", "Método no permitido.")
            body = self._read_json_body()
            if body is None:
                return
            if not isinstance(body, dict):
                return self._error(HTTPStatus.BAD_REQUEST, "bad_body", "Se esperaba un objeto JSON.")
            return self._json(HTTPStatus.OK, lic.activate(body.get("license_key")))

        routes_get = {
            "/api/status": lambda: svc.status(),
            "/api/devices": lambda: svc.devices(),
            "/api/voices": lambda: svc.voices(),
        }
        if path in routes_get:
            if method != "GET":
                return self._error(HTTPStatus.METHOD_NOT_ALLOWED, "method_not_allowed", "Método no permitido.")
            return self._json(HTTPStatus.OK, routes_get[path]())

        if path == "/api/config":
            if method != "POST":
                return self._error(HTTPStatus.METHOD_NOT_ALLOWED, "method_not_allowed", "Método no permitido.")
            body = self._read_json_body()
            if body is None:
                return
            if not isinstance(body, dict):
                return self._error(HTTPStatus.BAD_REQUEST, "bad_body", "Se esperaba un objeto JSON.")
            return self._json(HTTPStatus.OK, svc.configure(body))

        if path in ("/api/start", "/api/stop"):
            if method != "POST":
                return self._error(HTTPStatus.METHOD_NOT_ALLOWED, "method_not_allowed", "Método no permitido.")
            if path == "/api/start" and self.ctl._license is not None and not self.ctl._license.allows_use():
                return self._error(
                    HTTPStatus.PAYMENT_REQUIRED,
                    "license_required",
                    "Terminó la prueba gratis de 7 días. Ingresá tu clave de licencia para seguir traduciendo.",
                )
            result = svc.start() if path == "/api/start" else svc.stop()
            # El servicio puede devolver el status actualizado; si no, respuesta mínima.
            return self._json(HTTPStatus.OK, result if isinstance(result, dict) else {"ok": True})

        if path == "/api/history":
            if method == "GET":
                limit = _int_param(query, "limit", 50, 1, 500)
                offset = _int_param(query, "offset", 0, 0, 10_000_000)
                return self._json(HTTPStatus.OK, svc.history(limit, offset))
            if method == "DELETE":
                return self._json(HTTPStatus.OK, {"deleted": svc.delete_history(None)})
            return self._error(HTTPStatus.METHOD_NOT_ALLOWED, "method_not_allowed", "Método no permitido.")

        if path == "/api/history/export":
            if method != "GET":
                return self._error(HTTPStatus.METHOD_NOT_ALLOWED, "method_not_allowed", "Método no permitido.")
            fmt = (query.get("format") or ["json"])[0]
            if fmt not in EXPORT_FORMATS:
                return self._error(HTTPStatus.BAD_REQUEST, "bad_format", "Formato no soportado: usá json o txt.")
            text = svc.export_history(fmt)
            ctype = "application/json; charset=utf-8" if fmt == "json" else "text/plain; charset=utf-8"
            filename = f"opentraductor-historial.{fmt}"
            return self._send(
                HTTPStatus.OK,
                text.encode("utf-8"),
                ctype,
                {"Content-Disposition": f'attachment; filename="{filename}"'},
            )

        if path.startswith("/api/history/session/"):
            if method != "DELETE":
                return self._error(HTTPStatus.METHOD_NOT_ALLOWED, "method_not_allowed", "Método no permitido.")
            session_id = path[len("/api/history/session/"):]
            if not session_id or "/" in session_id or len(session_id) > 128:
                return self._error(HTTPStatus.BAD_REQUEST, "bad_session", "ID de sesión inválido.")
            return self._json(HTTPStatus.OK, {"deleted": svc.delete_history(session_id)})

        return self._error(HTTPStatus.NOT_FOUND, "not_found", "Ruta no encontrada.")

    def _serve_static(self, path: str) -> None:
        filename, ctype = STATIC_FILES[path]
        data = (self.ctl._ui_dir / filename).read_bytes()
        if filename == "index.html":
            meta = META_TAG.format(token=html.escape(self.ctl.token, quote=True))
            text = data.decode("utf-8")
            if "</head>" not in text:
                return self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal", "index.html inválido.")
            data = text.replace("</head>", meta + "\n</head>", 1).encode("utf-8")
        self._send(HTTPStatus.OK, data, ctype)

    def _serve_sse(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        if self.command == "HEAD":
            return  # solo cabeceras: un HEAD no debe abrir el flujo
        client = self.ctl._add_client()
        try:
            self.wfile.write(b": conectado\n\n")
            self.wfile.flush()
            since_keepalive = 0.0
            while not self.ctl._stopped:
                try:
                    ev = client.queue.get(timeout=SSE_POLL_S)
                except queue.Empty:
                    since_keepalive += SSE_POLL_S
                    if since_keepalive >= SSE_KEEPALIVE_S:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        since_keepalive = 0.0
                    continue
                if ev is None:
                    break
                payload = json.dumps(ev, ensure_ascii=False)
                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.ctl._remove_client(client)


def _int_param(query: dict[str, list[str]], name: str, default: int, lo: int, hi: int) -> int:
    raw = (query.get(name) or [None])[0]
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ServiceError("bad_param", f"Parámetro {name} inválido.", HTTPStatus.BAD_REQUEST)
    if value < lo or value > hi:
        raise ServiceError("bad_param", f"Parámetro {name} fuera de rango.", HTTPStatus.BAD_REQUEST)
    return value

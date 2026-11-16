"""Pruebas de runtime/http_control.py con un servicio falso.

No abre micrófono, no carga modelos y no usa el pipeline. El servidor escucha
en 127.0.0.1 con puerto efímero (port=0).
"""
from __future__ import annotations

import http.client
import json
import threading
import time
from pathlib import Path

import pytest

from runtime.http_control import ConflictError, ControlServer, ServiceError

UI_DIR = Path(__file__).resolve().parents[2] / "ui"


class FakeService:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.subs: dict[int, object] = {}
        self._next = 0
        self.fail_with: Exception | None = None

    def _maybe_fail(self) -> None:
        if self.fail_with is not None:
            raise self.fail_with

    def status(self):
        self._maybe_fail()
        return {"state": "idle", "error": None, "preflight": [], "models": {}, "config": {}, "metrics": {}}

    def devices(self):
        return {"inputs": [{"id": "in1", "name": "Mic", "default": True}], "outputs": []}

    def voices(self):
        return {"voices": [{"id": "af_heart", "label": "Heart"}]}

    def configure(self, values):
        self.calls.append(("configure", values))
        return self.status()

    def start(self):
        self.calls.append(("start",))

    def stop(self):
        self.calls.append(("stop",))

    def history(self, limit, offset):
        self.calls.append(("history", limit, offset))
        return {"items": [{"id": "1", "session_id": "s", "ts": "t", "text_es": "hola", "text_en": "hello"}], "total": 1}

    def export_history(self, format):
        self.calls.append(("export", format))
        return '[{"text_es": "hola"}]' if format == "json" else "ES: hola\n"

    def delete_history(self, session_id=None):
        self.calls.append(("delete", session_id))
        return 1

    def subscribe(self, callback):
        self._next += 1
        self.subs[self._next] = callback
        return self._next

    def unsubscribe(self, handle):
        self.subs.pop(handle, None)
        self.calls.append(("unsubscribe", handle))

    def emit(self, event: dict) -> None:
        for cb in list(self.subs.values()):
            cb(event)


@pytest.fixture
def svc():
    return FakeService()


@pytest.fixture
def server(svc):
    srv = ControlServer(svc, port=0, ui_dir=UI_DIR, token="tok-de-prueba-123")
    srv.start()
    yield srv
    srv.stop()


def _conn(srv: ControlServer) -> http.client.HTTPConnection:
    return http.client.HTTPConnection("127.0.0.1", srv.port, timeout=5)


def _req(srv, method, path, body=None, headers=None):
    conn = _conn(srv)
    h = {"X-OT-Token": srv.token}
    h.update(headers or {})
    data = None
    if body is not None:
        data = body if isinstance(body, (bytes, str)) else json.dumps(body)
        h.setdefault("Content-Type", "application/json")
    conn.request(method, path, body=data, headers=h)
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    return resp.status, dict(resp.getheaders()), raw


def _json(srv, method, path, body=None, headers=None):
    status, hdrs, raw = _req(srv, method, path, body, headers)
    return status, json.loads(raw.decode("utf-8")) if raw else None


# ------------------------------------------------------------- autenticación
def test_api_without_token_is_rejected(server):
    status, data = _json(server, "GET", "/api/status", headers={"X-OT-Token": ""})
    assert status == 401
    assert data["error"]["code"] == "unauthorized"


def test_api_with_wrong_token_is_rejected(server):
    status, data = _json(server, "GET", "/api/status", headers={"X-OT-Token": "otro"})
    assert status == 401


def test_api_with_token_works(server):
    status, data = _json(server, "GET", "/api/status")
    assert status == 200
    assert data["state"] == "idle"


def test_index_injects_token_meta(server):
    status, hdrs, raw = _req(server, "GET", "/", headers={"X-OT-Token": ""})
    assert status == 200
    text = raw.decode("utf-8")
    assert f'<meta name="ot-token" content="{server.token}">' in text
    assert hdrs["Content-Security-Policy"].startswith("default-src 'self'")


# ------------------------------------------------------- Host y Origin
def test_foreign_host_is_rejected(server):
    status, data = _json(server, "GET", "/api/status", headers={"Host": "evil.example"})
    assert status == 403
    assert data["error"]["code"] == "forbidden_host"


def test_localhost_host_is_accepted(server):
    status, _ = _json(server, "GET", "/api/status", headers={"Host": f"localhost:{server.port}"})
    assert status == 200


def test_foreign_origin_is_rejected(server):
    status, data = _json(server, "GET", "/api/status", headers={"Origin": "https://evil.example"})
    assert status == 403
    assert data["error"]["code"] == "forbidden_origin"


def test_own_origin_is_accepted(server):
    status, _ = _json(server, "GET", "/api/status", headers={"Origin": f"http://127.0.0.1:{server.port}"})
    assert status == 200


# --------------------------------------------------------- activos estáticos
@pytest.mark.parametrize("path,ctype", [
    ("/app.js", "text/javascript"),
    ("/styles.css", "text/css"),
])
def test_static_assets_served(server, path, ctype):
    status, hdrs, raw = _req(server, "GET", path, headers={"X-OT-Token": ""})
    assert status == 200
    assert hdrs["Content-Type"].startswith(ctype)
    assert len(raw) > 0


@pytest.mark.parametrize("path", [
    "/runtime/text_store.py",
    "/../runtime/http_control.py",
    "/%2e%2e/runtime/config.py",
    "/app.js.bak",
    "/etc/passwd",
])
def test_non_allowlisted_paths_are_not_served(server, path):
    status, _, raw = _req(server, "GET", path, headers={"X-OT-Token": ""})
    assert status == 404
    assert b"def " not in raw and b"class " not in raw


# ------------------------------------------------------- configurar y control
def test_config_start_stop_call_service(server, svc):
    status, data = _json(server, "POST", "/api/config", {"voice": "af_heart", "monitor_volume": 0.4})
    assert status == 200 and data["state"] == "idle"
    assert ("configure", {"voice": "af_heart", "monitor_volume": 0.4}) in svc.calls
    assert _json(server, "POST", "/api/start")[0] == 200
    assert _json(server, "POST", "/api/stop")[0] == 200
    assert ("start",) in svc.calls and ("stop",) in svc.calls


def test_config_rejects_bad_json_and_non_objects(server):
    status, data = _json(server, "POST", "/api/config", b"{no es json")
    assert status == 400 and data["error"]["code"] == "bad_json"
    status, data = _json(server, "POST", "/api/config", [1, 2])
    assert status == 400 and data["error"]["code"] == "bad_body"


def test_config_body_size_limit(server, svc):
    big = {"voice": "x" * (32 * 1024)}
    status, data = _json(server, "POST", "/api/config", big)
    assert status == 413
    assert data["error"]["code"] == "too_large"
    assert not any(c[0] == "configure" for c in svc.calls)


def test_post_without_content_length_is_refused(server):
    conn = _conn(server)
    conn.putrequest("POST", "/api/config", skip_accept_encoding=True)
    conn.putheader("X-OT-Token", server.token)
    conn.endheaders()
    resp = conn.getresponse()
    assert resp.status == 411
    conn.close()


def test_wrong_method_is_405(server):
    status, data = _json(server, "POST", "/api/status")
    assert status == 405
    assert data["error"]["code"] == "method_not_allowed"


def test_unknown_api_route_is_404(server):
    status, data = _json(server, "GET", "/api/nada")
    assert status == 404


# -------------------------------------------------------------- historial
def test_history_params_are_validated_and_passed(server, svc):
    status, data = _json(server, "GET", "/api/history?limit=10&offset=5")
    assert status == 200 and data["total"] == 1
    assert ("history", 10, 5) in svc.calls
    status, data = _json(server, "GET", "/api/history?limit=abc")
    assert status == 400 and data["error"]["code"] == "bad_param"
    status, _ = _json(server, "GET", "/api/history?limit=0")
    assert status == 400
    status, _ = _json(server, "GET", "/api/history?limit=501")
    assert status == 400


def test_history_defaults(server, svc):
    _json(server, "GET", "/api/history")
    assert ("history", 50, 0) in svc.calls


def test_export_json_and_txt_with_attachment_header(server, svc):
    status, hdrs, raw = _req(server, "GET", "/api/history/export?format=txt")
    assert status == 200
    assert hdrs["Content-Type"].startswith("text/plain")
    assert 'attachment; filename="opentraductor-historial.txt"' == hdrs["Content-Disposition"]
    assert raw.decode("utf-8") == "ES: hola\n"
    status, hdrs, raw = _req(server, "GET", "/api/history/export?format=json")
    assert hdrs["Content-Type"].startswith("application/json")
    assert ("export", "json") in svc.calls


def test_export_bad_format_is_400(server, svc):
    status, data = _json(server, "GET", "/api/history/export?format=pdf")
    assert status == 400 and data["error"]["code"] == "bad_format"
    assert not any(c[0] == "export" and c[1] == "pdf" for c in svc.calls)


def test_delete_all_and_one_session(server, svc):
    assert _json(server, "DELETE", "/api/history")[0] == 200
    assert ("delete", None) in svc.calls
    status, data = _json(server, "DELETE", "/api/history/session/abc123")
    assert status == 200 and data["deleted"] == 1
    assert ("delete", "abc123") in svc.calls


def test_delete_session_rejects_nested_or_long_ids(server, svc):
    status, data = _json(server, "DELETE", "/api/history/session/a/b")
    assert status == 404 or status == 400
    status, data = _json(server, "DELETE", "/api/history/session/" + "x" * 200)
    assert status == 400
    assert not any(c[0] == "delete" and c[1] and len(c[1]) > 128 for c in svc.calls)


# ----------------------------------------------------------- errores del servicio
def test_service_error_maps_to_json(server, svc):
    svc.fail_with = ServiceError("conflict", "No se puede cambiar el dispositivo mientras traduce.", 409)
    status, data = _json(server, "GET", "/api/status")
    assert status == 409
    assert data == {"error": {"code": "conflict", "message": "No se puede cambiar el dispositivo mientras traduce."}}


def test_unexpected_exception_does_not_leak_details(server, svc):
    svc.fail_with = RuntimeError("ruta secreta /Users/ejemplo/algo")
    status, data = _json(server, "GET", "/api/status")
    assert status == 500
    assert data["error"]["code"] == "internal"
    assert "secreta" not in json.dumps(data)


# ----------------------------------------------------------------- SSE
def _open_sse(srv, token=None):
    conn = _conn(srv)
    path = "/api/events"
    if token is not None:
        path += "?token=" + token
    conn.request("GET", path)
    resp = conn.getresponse()
    return conn, resp


def test_sse_requires_token(server):
    conn, resp = _open_sse(server, token="mal")
    assert resp.status == 401
    conn.close()


def test_sse_delivers_service_events(server, svc):
    conn, resp = _open_sse(server, token=server.token)
    assert resp.status == 200
    assert resp.getheader("Content-Type").startswith("text/event-stream")
    assert resp.fp.readline().startswith(b": conectado")
    # El cliente se registra antes de que el servicio emita.
    deadline = time.time() + 2
    while time.time() < deadline and not svc.subs:
        time.sleep(0.01)
    svc.emit({"type": "segment", "seg_id": "7", "text_es": "hola", "text_en": "hello"})
    line = b""
    while not line.startswith(b"data:"):
        line = resp.fp.readline()
    event = json.loads(line[len(b"data:"):].decode("utf-8"))
    assert event == {"type": "segment", "seg_id": "7", "text_es": "hola", "text_en": "hello"}
    conn.close()


def test_sse_slow_client_drops_oldest_not_blocking(server, svc):
    conn, resp = _open_sse(server, token=server.token)
    resp.fp.readline()  # conectado
    deadline = time.time() + 2
    while time.time() < deadline and not svc.subs:
        time.sleep(0.01)
    start = time.time()
    for i in range(2000):  # más que SSE_QUEUE_MAX sin leer
        svc.emit({"type": "partial", "seg_id": str(i), "text_es": "x"})
    assert time.time() - start < 5, "el emisor no debe bloquearse con un cliente lento"
    conn.close()


def test_stop_closes_sse_and_unsubscribes(svc):
    srv = ControlServer(svc, port=0, ui_dir=UI_DIR, token="t" * 20)
    srv.start()
    conn, resp = _open_sse(srv, token=srv.token)
    resp.fp.readline()
    srv.stop()
    assert any(c[0] == "unsubscribe" for c in svc.calls)
    assert svc.subs == {}
    conn.close()
    # Ya no acepta conexiones nuevas.
    with pytest.raises(OSError):
        c2 = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=1)
        c2.request("GET", "/")
        c2.getresponse()


def test_stop_is_idempotent_and_context_manager_works(svc):
    with ControlServer(svc, port=0, ui_dir=UI_DIR, token="u" * 20) as srv:
        assert srv.url.startswith("http://127.0.0.1:")
    srv.stop()  # segunda llamada no falla


def test_generated_token_is_long_and_unique(svc):
    a = ControlServer(svc, port=0, ui_dir=UI_DIR)
    b = ControlServer(svc, port=0, ui_dir=UI_DIR)
    try:
        assert len(a.token) >= 32
        assert a.token != b.token
    finally:
        a.stop() if a._thread else None
        b.stop() if b._thread else None
        a._httpd.server_close()
        b._httpd.server_close()


# ------------------------------------------- conexión keep-alive y cuerpos
def _same_conn_request(conn, method, path, body=None, token="tok-de-prueba-123"):
    headers = {"X-OT-Token": token}
    data = None
    if body is not None:
        data = body
        headers["Content-Type"] = "application/json"
    conn.request(method, path, body=data, headers=headers)
    resp = conn.getresponse()
    return resp.status, resp.read()


def test_post_with_empty_object_then_get_on_same_connection(server, svc):
    conn = _conn(server)
    try:
        status, _ = _same_conn_request(conn, "POST", "/api/config", body="{}")
        assert status == 200
        status, raw = _same_conn_request(conn, "GET", "/api/status")
        assert status == 200 and json.loads(raw)["state"] == "idle"
    finally:
        conn.close()


def test_start_and_stop_with_body_do_not_corrupt_connection(server, svc):
    conn = _conn(server)
    try:
        status, _ = _same_conn_request(conn, "POST", "/api/start", body="{}")
        assert status == 200
        status, _ = _same_conn_request(conn, "POST", "/api/stop", body='{"x": 1}')
        assert status == 200
        status, raw = _same_conn_request(conn, "GET", "/api/devices")
        assert status == 200 and json.loads(raw)["inputs"][0]["id"] == "in1"
    finally:
        conn.close()
    assert ("start",) in svc.calls and ("stop",) in svc.calls


def test_post_without_body_then_get_on_same_connection(server):
    conn = _conn(server)
    try:
        status, _ = _same_conn_request(conn, "POST", "/api/start")
        assert status == 200
        status, _ = _same_conn_request(conn, "GET", "/api/voices")
        assert status == 200
    finally:
        conn.close()


def test_rejected_request_with_body_closes_cleanly(server):
    """Un 401 con cuerpo: el servidor cierra esa conexión. No queda basura en el socket
    y una conexión nueva responde bien."""
    conn = _conn(server)
    try:
        status, _ = _same_conn_request(conn, "POST", "/api/config", body='{"a": 1}', token="mal")
        assert status == 401
    finally:
        conn.close()
    status, _ = _json(server, "GET", "/api/status")
    assert status == 200


def test_head_on_events_returns_headers_without_stream(server):
    conn = _conn(server)
    try:
        conn.request("HEAD", "/api/events?token=" + server.token)
        resp = conn.getresponse()
        assert resp.status == 200
        assert resp.getheader("Content-Type").startswith("text/event-stream")
        resp.read()
    finally:
        conn.close()


def test_stop_ends_sse_handler_even_with_full_queue(svc):
    srv = ControlServer(svc, port=0, ui_dir=UI_DIR, token="s" * 24)
    srv.start()
    conn, resp = _open_sse(srv, token=srv.token)
    resp.fp.readline()  # conectado; el cliente NO lee después
    deadline = time.time() + 2
    while time.time() < deadline and not svc.subs:
        time.sleep(0.01)
    for i in range(1000):  # llena la cola del cliente (512)
        svc.emit({"type": "partial", "seg_id": str(i), "text_es": "x"})
    start = time.time()
    srv.stop()
    assert time.time() - start < 5, "stop no debe quedar colgado con cola llena"
    assert srv._clients == []
    conn.close()


# --------------------------- errores de conflicto e inválidos, status de start/stop
def test_conflict_error_maps_to_409(server, svc):
    svc.fail_with = ConflictError("No se puede cambiar voice mientras traduce.")
    status, data = _json(server, "POST", "/api/config", {"voice": "af_bella"})
    assert status == 409
    assert data["error"]["code"] == "conflict"
    assert "voice" in data["error"]["message"]


def test_value_error_maps_to_400_invalid(server, svc):
    svc.fail_with = ValueError("monitor_volume debe estar entre 0 y 1.")
    status, data = _json(server, "POST", "/api/config", {"monitor_volume": 9})
    assert status == 400
    assert data == {"error": {"code": "invalid", "message": "monitor_volume debe estar entre 0 y 1."}}


def test_start_and_stop_return_service_status_when_given(svc):
    class StatusService(FakeService):
        def start(self):
            return {"state": "starting"}

        def stop(self):
            return {"state": "stopping"}

    srv = ControlServer(StatusService(), port=0, ui_dir=UI_DIR, token="v" * 24)
    srv.start()
    try:
        assert _json(srv, "POST", "/api/start") == (200, {"state": "starting"})
        assert _json(srv, "POST", "/api/stop") == (200, {"state": "stopping"})
    finally:
        srv.stop()

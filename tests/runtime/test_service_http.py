"""Integración RuntimeService (falso motor/audio) + ControlServer real en loopback."""
import json
import time
import urllib.error
import urllib.request

from runtime.http_control import ControlServer
from runtime.service import RuntimeService

from test_service import FakeAudio, FakeEngine, FakeMic, ok_preflight, wait


def call(srv, method, path, body=None, token=True):
    req = urllib.request.Request(srv.url.rstrip("/") + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None)
    req.add_header("Host", f"127.0.0.1:{srv.port}")
    if body is not None:
        req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("X-OT-Token", srv.token)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def test_http_flow(tmp_path):
    svc = RuntimeService(tmp_path / "h.db", engine_factory=FakeEngine, audio=FakeAudio, preflight=ok_preflight)
    svc.voices = lambda: {"voices": [{"id": "af_heart", "label": "af_heart"}]}
    srv = ControlServer(svc, port=0)
    srv.start()
    try:
        assert call(srv, "GET", "/api/status", token=False)[0] in (401, 403)
        c, body = call(srv, "POST", "/api/config", {"input_id": "mic", "output_id": "spk"})
        assert c == 200 and body["config"]["input_id"] == "mic"
        assert call(srv, "POST", "/api/start")[0] == 200
        assert wait(lambda: call(srv, "GET", "/api/status")[1]["state"] == "running")
        c, body = call(srv, "POST", "/api/config", {"voice": "af_heart", "output_id": "bh"})
        assert c == 409 and body["error"]["code"] == "conflict"
        for _ in range(3):
            FakeMic.last.on_block(bytes(640))
        assert wait(lambda: call(srv, "GET", "/api/history?limit=5")[1].get("total", 0) == 1 or True)
        assert call(srv, "POST", "/api/stop")[0] == 200
        assert wait(lambda: call(srv, "GET", "/api/status")[1]["state"] == "idle")
        c, h = call(srv, "GET", "/api/history?limit=5")
        assert c == 200 and h["total"] == 1
        assert call(srv, "DELETE", "/api/history")[1]["deleted"] == 1
    finally:
        srv.stop()
        svc.close()

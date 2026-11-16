"""Edición Mac App Store (OPENTRADUCTOR_EDITION=store): arranca sin runtime/license.py y /api/license* da 404.

sign_store.sh borra runtime/license.py del bundle de tienda (guía 2.4.5(vi)); estas pruebas lo simulan
bloqueando el import con sys.modules["runtime.license"] = None.
"""

from __future__ import annotations

import ast
import http.client
import json
import signal
import sys
import types
from pathlib import Path

import runtime.__main__ as main_mod
import runtime.http_control as http_mod
import runtime.preflight as preflight_mod

RUNTIME = Path(main_mod.__file__).parent


class _Svc:
    def __init__(self) -> None:
        self.started = 0

    def _preflight(self, cfg):
        return []

    def start(self):
        self.started += 1

    def stop(self):
        pass

    def close(self):
        pass

    def subscribe(self, cb):
        return 1

    def unsubscribe(self, h):
        pass


def _call(srv, method, path):
    c = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=5)
    headers = {"Host": f"127.0.0.1:{srv.port}", "X-OT-Token": srv.token}
    body = None
    if method == "POST":
        body, headers["Content-Type"] = "{}", "application/json"
    c.request(method, path, body=body, headers=headers)
    r = c.getresponse()
    out = json.loads(r.read() or b"null")
    c.close()
    return r.status, out


def test_store_run_without_license_module(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENTRADUCTOR_EDITION", "store")
    monkeypatch.setitem(sys.modules, "runtime.license", None)  # import -> ImportError, como sin el archivo
    monkeypatch.setattr(preflight_mod, "force_offline", lambda: None)
    monkeypatch.setattr(signal, "signal", lambda *a, **k: None)
    cfg = types.SimpleNamespace(asr_backend="parakeet")
    monkeypatch.setattr(main_mod, "build_service", lambda a: (_Svc(), cfg))

    servers: list = []
    real = http_mod.ControlServer

    class Capture(real):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            servers.append(self)

    monkeypatch.setattr(http_mod, "ControlServer", Capture)
    seen: dict = {}

    class Ev:
        def set(self):
            pass

        def wait(self):  # en vez de esperar Ctrl+C, consulta el servidor y vuelve
            srv = servers[0]
            seen["license"] = _call(srv, "GET", "/api/license")
            seen["activate"] = _call(srv, "POST", "/api/license/activate")
            seen["start"] = _call(srv, "POST", "/api/start")

    monkeypatch.setattr(main_mod, "threading", types.SimpleNamespace(Event=Ev))  # solo el Event de cmd_run
    a = types.SimpleNamespace(db=str(tmp_path / "h.db"), port=0, engine_mode="baseline")
    assert main_mod.cmd_run(a) == 0
    assert servers[0]._license is None
    assert seen["license"][0] == 404 and seen["license"][1]["error"]["code"] == "not_found"
    assert seen["activate"][0] == 404
    assert seen["start"][0] == 200  # sin compuerta de licencia


def test_store_server_ignores_license_manager():
    class Lic:
        def allows_use(self):
            return False

    with http_mod.ControlServer(_Svc(), port=0, license=Lic(), store=True) as srv:
        assert _call(srv, "GET", "/api/license")[0] == 404
        assert _call(srv, "POST", "/api/start")[0] == 200  # sin compuerta 402 de licencia en la tienda


def test_only_main_imports_license_and_only_inside_a_function():
    for f in RUNTIME.glob("*.py"):
        if f.name == "license.py":
            continue
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [n.name for n in node.names if node.module in (None, "")]
            elif isinstance(node, ast.Import):
                names = [n.name for n in node.names]
            if any(n in ("license", "runtime.license") for n in names):
                assert f.name == "__main__.py", f"{f.name} importa license"
                assert node not in tree.body, "__main__ importa license a nivel de módulo"

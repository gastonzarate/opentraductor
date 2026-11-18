"""Edición Mac App Store (OPENTRADUCTOR_EDITION=store) y fuera de ella: sin licencia (T-70, repo open source).

En la tienda /api/license da 404; fuera, solo GET con state=disabled (la UI web oculta la tarjeta). /api/start nunca
pide licencia. sign_store.sh corta si el bundle lleva código de licencia (guía 2.4.5(vi)).
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
    a = types.SimpleNamespace(db=str(tmp_path / "h.db"), port=0)
    assert main_mod.cmd_run(a) == 0
    assert seen["license"][0] == 404 and seen["license"][1]["error"]["code"] == "not_found"
    assert seen["activate"][0] == 404
    assert seen["start"][0] == 200  # sin compuerta de licencia


def test_no_license_module_nor_gate():
    assert not (RUNTIME / "license.py").exists()
    for f in RUNTIME.rglob("*.py"):
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").endswith("license") and "license" not in [n.name for n in node.names], f
    with http_mod.ControlServer(_Svc(), port=0) as srv:
        st, body = _call(srv, "GET", "/api/license")
        assert st == 200 and body["state"] == "disabled"
        assert _call(srv, "POST", "/api/license/activate")[0] == 404
        assert _call(srv, "POST", "/api/start")[0] == 200
    with http_mod.ControlServer(_Svc(), port=0, store=True) as srv:
        assert _call(srv, "GET", "/api/license")[0] == 404

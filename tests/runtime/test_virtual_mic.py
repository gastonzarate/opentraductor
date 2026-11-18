"""Pruebas de runtime/virtual_mic.py (detección de BlackHole 2ch) y de la ruta /api/virtual-mic."""

from __future__ import annotations

import http.client
import json

from runtime import virtual_mic
from runtime.http_control import ControlServer


def test_virtual_mic_detects_blackhole_2ch():
    d = {
        "audio_ok": True,
        "inputs": [{"id": "BlackHole 2ch", "name": "BlackHole 2ch"}, {"id": "Mic", "name": "MacBook Pro Microphone"}],
        "outputs": [{"id": "BlackHole 2ch", "name": "BlackHole 2ch"}],
    }
    r = virtual_mic.detect(d)
    assert r["installed"] and r["steps"] == [] and r["output_id"] == "BlackHole 2ch"


def test_virtual_mic_missing_gives_three_steps():
    r = virtual_mic.detect({"audio_ok": True, "inputs": [{"id": "Mic", "name": "Mic"}], "outputs": [{"id": "S", "name": "Speakers"}]})
    assert not r["installed"] and [s["n"] for s in r["steps"]] == [1, 2, 3]
    assert r["steps"][0]["command"] == "brew install blackhole-2ch"


def test_virtual_mic_ignores_blackhole_16ch_and_half_installed():
    only16 = {"inputs": [{"id": "BlackHole 16ch", "name": "BlackHole 16ch"}], "outputs": [{"id": "BlackHole 16ch", "name": "BlackHole 16ch"}]}
    assert not virtual_mic.detect(only16)["installed"]
    half = {"inputs": [], "outputs": [{"id": "BlackHole 2ch", "name": "BlackHole 2ch"}]}
    assert not virtual_mic.detect(half)["installed"]


# ------------------------------------------------------------------ HTTP
class Svc:
    def __init__(self) -> None:
        self.started = 0

    def devices(self):
        return {"audio_ok": True, "inputs": [{"id": "BlackHole 2ch", "name": "BlackHole 2ch"}], "outputs": [{"id": "BlackHole 2ch", "name": "BlackHole 2ch"}]}

    def start(self):
        self.started += 1

    def stop(self):
        pass

    def subscribe(self, cb):
        return 1

    def unsubscribe(self, h):
        pass


def call(srv, method, path, body=None, token=True):
    c = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=5)
    headers = {"Host": f"127.0.0.1:{srv.port}"}
    if token:
        headers["X-OT-Token"] = srv.token
    data = None
    if body is not None:
        data = json.dumps(body)
        headers["Content-Type"] = "application/json"
    c.request(method, path, body=data, headers=headers)
    r = c.getresponse()
    out = json.loads(r.read() or b"null")
    c.close()
    return r.status, out


def test_http_virtual_mic():
    with ControlServer(Svc(), port=0) as srv:
        st, body = call(srv, "GET", "/api/virtual-mic")
        assert st == 200 and body["installed"] is True
        assert call(srv, "POST", "/api/virtual-mic")[0] == 405

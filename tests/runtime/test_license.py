"""Pruebas de runtime/license.py y de las rutas /api/license y /api/virtual-mic.

Sin red: la API de Lemon Squeezy se reemplaza por un ``opener`` falso. Reloj inyectado.
"""
from __future__ import annotations

import http.client
import io
import json
import os
import stat
import urllib.error
import urllib.parse

import pytest

from runtime import license as lic_mod
from runtime import virtual_mic
from runtime.http_control import ControlServer, ServiceError
from runtime.license import LicenseManager, TRIAL_DAYS

DAY = 86400.0
KEY = "38b1460a-5104-4067-a91d-77b872934d51"


class Clock:
    def __init__(self, t: float = 1_800_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class FakeResp:
    def __init__(self, body: dict, status: int = 200) -> None:
        self._raw = json.dumps(body).encode()
        self.status = status

    def read(self, n=-1):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def ok_body(**over):
    body = {
        "activated": True,
        "error": None,
        "license_key": {"status": "active", "key": KEY, "activation_limit": 2, "activation_usage": 1},
        "instance": {"id": "47596ad9-a811-4ebf-ac8a-03fc7b6d2a17", "name": "Mac de prueba"},
        "meta": {"store_id": 1, "product_id": 2},
    }
    body.update(over)
    return body


class Opener:
    def __init__(self, responder) -> None:
        self.responder = responder
        self.requests: list = []

    def __call__(self, req, timeout):
        self.requests.append((req, timeout))
        return self.responder(req)


def make(tmp_path, responder=None, clock=None):
    clock = clock or Clock()
    op = Opener(responder or (lambda req: FakeResp(ok_body())))
    return LicenseManager(tmp_path / "license.json", clock=clock, opener=op, instance_name="Mac de prueba"), clock, op


def test_trial_starts_on_first_use_and_counts_down(tmp_path):
    m, clock, _ = make(tmp_path)
    s = m.status()
    assert (s["state"], s["trial_days_left"], s["can_use"]) == ("trial", TRIAL_DAYS, True)
    clock.t += 3 * DAY + 10
    assert m.status()["trial_days_left"] == TRIAL_DAYS - 3
    clock.t += 4 * DAY
    s = m.status()
    assert (s["state"], s["trial_days_left"], s["can_use"]) == ("expired", 0, False)
    assert not m.allows_use()


def test_trial_persists_across_instances(tmp_path):
    m, clock, _ = make(tmp_path)
    m.status()
    clock.t += 8 * DAY
    m2 = LicenseManager(tmp_path / "license.json", clock=clock, instance_name="x")
    assert m2.status()["state"] == "expired"


def test_clock_rollback_does_not_extend_trial(tmp_path):
    m, clock, _ = make(tmp_path)
    m.status()
    clock.t += 8 * DAY
    assert m.status()["state"] == "expired"
    clock.t -= 7 * DAY  # el usuario atrasa el reloj
    assert m.status()["state"] == "expired"


def test_license_file_permissions(tmp_path):
    m, _, _ = make(tmp_path)
    m.status()
    mode = stat.S_IMODE(os.stat(tmp_path / "license.json").st_mode)
    assert mode == 0o600


def test_activate_sends_only_key_and_instance_name(tmp_path):
    m, clock, op = make(tmp_path)
    m.status()  # la prueba empieza en la primera consulta
    clock.t += 7 * DAY - 1
    assert m.status()["state"] == "trial"
    clock.t += 5
    assert m.status()["state"] == "expired"
    s = m.activate("  " + KEY + " ")
    assert s["state"] == "licensed" and s["key_hint"] == "…4d51" and s["can_use"]
    (req, timeout), = op.requests
    assert req.full_url == "https://api.lemonsqueezy.com/v1/licenses/activate"
    assert req.get_method() == "POST"
    assert urllib.parse.parse_qs(req.data.decode()) == {"license_key": [KEY], "instance_name": ["Mac de prueba"]}
    assert timeout <= 15


def test_after_activation_no_more_network_and_survives_time(tmp_path):
    m, clock, op = make(tmp_path)
    m.activate(KEY)
    n = len(op.requests)
    clock.t += 400 * DAY
    assert m.status()["state"] == "licensed"
    assert m.allows_use()
    assert len(op.requests) == n


def test_second_activation_on_same_mac_is_refused_without_network(tmp_path):
    m, _, op = make(tmp_path)
    m.activate(KEY)
    with pytest.raises(ServiceError) as e:
        m.activate(KEY)
    assert e.value.code == "already_active" and len(op.requests) == 1


@pytest.mark.parametrize("bad", [None, 5, "", "   ", "a" * 201, "clave con espacio", "cla\nve"])
def test_invalid_key_format_never_reaches_network(tmp_path, bad):
    m, _, op = make(tmp_path)
    with pytest.raises(ServiceError) as e:
        m.activate(bad)
    assert e.value.code == "invalid_key" and not op.requests


def http_error(code, body):
    return urllib.error.HTTPError(lic_mod.ACTIVATE_URL, code, "x", {}, io.BytesIO(json.dumps(body).encode()))


def test_rejected_key(tmp_path):
    def resp(req):
        raise http_error(404, {"activated": False, "error": "license_key not found.", "license_key": None})

    m, _, _ = make(tmp_path, resp)
    with pytest.raises(ServiceError) as e:
        m.activate(KEY)
    assert e.value.code == "license_rejected"
    assert m.status()["state"] != "licensed"


def test_activation_limit_reached(tmp_path):
    def resp(req):
        raise http_error(400, {"activated": False, "error": "limit", "license_key": {"activation_limit": 2, "activation_usage": 2, "status": "active"}})

    m, _, _ = make(tmp_path, resp)
    with pytest.raises(ServiceError) as e:
        m.activate(KEY)
    assert e.value.code == "activation_limit" and e.value.status == 409


def test_network_down_keeps_trial_state(tmp_path):
    def resp(req):
        raise urllib.error.URLError("sin red")

    m, _, _ = make(tmp_path, resp)
    with pytest.raises(ServiceError) as e:
        m.activate(KEY)
    assert e.value.code == "network" and e.value.status == 502
    assert m.status()["state"] == "trial"


def test_server_error_and_garbage_are_network_errors(tmp_path):
    def r500(req):
        raise http_error(503, {})

    m, _, _ = make(tmp_path, r500)
    with pytest.raises(ServiceError) as e:
        m.activate(KEY)
    assert e.value.code == "network"

    class Garbage(FakeResp):
        def __init__(self):
            self._raw = b"<html>no json</html>"

    m2, _, _ = make(tmp_path / "b", lambda req: Garbage())
    with pytest.raises(ServiceError) as e2:
        m2.activate(KEY)
    assert e2.value.code == "network"


def test_product_mismatch_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(lic_mod, "EXPECTED_PRODUCT_ID", 999)
    m, _, _ = make(tmp_path)
    with pytest.raises(ServiceError) as e:
        m.activate(KEY)
    assert e.value.code == "license_rejected"
    monkeypatch.setattr(lic_mod, "EXPECTED_PRODUCT_ID", 2)
    monkeypatch.setattr(lic_mod, "EXPECTED_STORE_ID", 1)
    assert m.activate(KEY)["licensed"]


def test_inactive_key_status_is_rejected(tmp_path):
    body = ok_body(license_key={"status": "disabled", "activation_limit": 2, "activation_usage": 1})
    m, _, _ = make(tmp_path, lambda req: FakeResp(body))
    with pytest.raises(ServiceError):
        m.activate(KEY)


# ---------------------------------------------------------------- virtual_mic
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


def test_http_license_gate_and_activation(tmp_path):
    m, clock, _ = make(tmp_path)
    svc = Svc()
    with ControlServer(svc, port=0, license=m) as srv:
        assert call(srv, "GET", "/api/license")[1]["state"] == "trial"
        assert call(srv, "POST", "/api/start")[0] == 200 and svc.started == 1
        clock.t += 8 * DAY
        st, body = call(srv, "POST", "/api/start")
        assert st == 402 and body["error"]["code"] == "license_required" and svc.started == 1
        st, body = call(srv, "POST", "/api/license/activate", {"license_key": KEY})
        assert st == 200 and body["state"] == "licensed"
        assert call(srv, "POST", "/api/start")[0] == 200 and svc.started == 2
        st, body = call(srv, "POST", "/api/license/activate", {"license_key": 5})
        assert st == 409 or st == 400


def test_http_license_requires_token(tmp_path):
    m, _, _ = make(tmp_path)
    with ControlServer(Svc(), port=0, license=m) as srv:
        assert call(srv, "GET", "/api/license", token=False)[0] == 401
        assert call(srv, "POST", "/api/license/activate", {"license_key": KEY}, token=False)[0] == 401


def test_http_without_license_manager_allows_start():
    svc = Svc()
    with ControlServer(svc, port=0) as srv:
        assert call(srv, "GET", "/api/license")[1]["state"] == "disabled"
        assert call(srv, "POST", "/api/start")[0] == 200
        assert call(srv, "POST", "/api/license/activate", {"license_key": KEY})[0] == 404


def test_http_virtual_mic():
    with ControlServer(Svc(), port=0) as srv:
        st, body = call(srv, "GET", "/api/virtual-mic")
        assert st == 200 and body["installed"] is True
        assert call(srv, "POST", "/api/virtual-mic")[0] == 405

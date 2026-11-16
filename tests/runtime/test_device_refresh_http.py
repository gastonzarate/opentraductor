"""POST /api/devices/refresh con un servicio falso: autenticación, método, errores y que GET /api/devices no refresque."""
import pytest
from test_http_control import FakeService, UI_DIR, _json, _req

from runtime.http_control import ControlServer, ServiceError


class RefreshService(FakeService):
    def __init__(self):
        super().__init__()
        self.fail_with = None
        self.refresh_error = None

    def refresh_devices(self):
        self.calls.append(("refresh",))
        if self.refresh_error is not None:
            raise self.refresh_error
        return {"devices": self.devices(), "refresh": {"generation": 1, "added": ["WH-1000XM5"], "removed": []}, "config_missing": []}


@pytest.fixture
def svc():
    return RefreshService()


@pytest.fixture
def server(svc):
    srv = ControlServer(svc, port=0, ui_dir=UI_DIR, token="tok-de-prueba-123")
    srv.start()
    yield srv
    srv.stop()


def test_refresh_requires_token_and_post(server, svc):
    assert _json(server, "POST", "/api/devices/refresh", headers={"X-OT-Token": ""})[0] == 401
    status, data = _json(server, "GET", "/api/devices/refresh")
    assert status == 405 and data["error"]["code"] == "method_not_allowed"
    assert ("refresh",) not in svc.calls


def test_refresh_ok_returns_devices_and_info(server, svc):
    status, data = _json(server, "POST", "/api/devices/refresh")
    assert status == 200 and data["refresh"]["added"] == ["WH-1000XM5"] and data["devices"]["inputs"][0]["id"] == "in1"
    assert svc.calls.count(("refresh",)) == 1


def test_refresh_service_errors_become_json_errors_with_their_status(server, svc):
    svc.refresh_error = ServiceError("bad_state", "No se pueden actualizar los dispositivos con una sesión en curso.", 409)
    status, data = _json(server, "POST", "/api/devices/refresh")
    assert status == 409 and data["error"]["code"] == "bad_state"
    svc.refresh_error = RuntimeError("detalle interno secreto")
    status, data = _json(server, "POST", "/api/devices/refresh")
    assert status == 500 and "secreto" not in str(data)


def test_get_devices_never_refreshes_and_foreign_host_is_rejected(server, svc):
    for _ in range(3):
        assert _json(server, "GET", "/api/devices")[0] == 200
    assert ("refresh",) not in svc.calls
    status, data = _json(server, "POST", "/api/devices/refresh", headers={"Host": "evil.example"})
    assert status == 403 and ("refresh",) not in svc.calls


def test_refresh_with_a_body_does_not_break_the_connection(server, svc):
    status, _hdrs, raw = _req(server, "POST", "/api/devices/refresh", body="{}")
    assert status == 200

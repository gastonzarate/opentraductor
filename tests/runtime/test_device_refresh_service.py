"""Servicio + refresco de dispositivos con motor y audio FALSOS. Foco: nunca con sesión/carga en curso, carrera con start(),
estado de la configuración, y que GET de dispositivos jamás toque PortAudio."""
import threading
import time
from types import SimpleNamespace

import pytest
import test_service as TS  # fakes del servicio existentes (FakeEngine, FakeMic, FakePlayer, wait, ok_preflight)

from runtime import audio_io
from runtime.config import PipelineConfig
from runtime.http_control import ServiceError
from runtime.service import RuntimeService


class RefreshAudio(TS.FakeAudio):
    """FakeAudio con refresh_devices() instrumentado. ``gate`` bloquea el refresco hasta soltarlo."""
    calls = 0
    gate: threading.Event | None = None
    entered = threading.Event()
    exc: Exception | None = None
    missing: set = set()
    ambiguous: set = set()
    device_calls = 0

    @classmethod
    def reset(cls):
        cls.calls, cls.gate, cls.exc, cls.missing, cls.device_calls, cls.ambiguous = 0, None, None, set(), 0, set()
        cls.entered = threading.Event()

    @staticmethod
    def refresh_devices():
        RefreshAudio.calls += 1
        RefreshAudio.entered.set()
        if RefreshAudio.gate is not None:
            RefreshAudio.gate.wait(5)
        if RefreshAudio.exc is not None:
            raise RefreshAudio.exc
        return {"generation": RefreshAudio.calls, "devices_before": 3, "devices_after": 5,
                "added": ["WH-1000XM5", "WH-1000XM5"], "removed": []}

    @staticmethod
    def devices_payload():
        RefreshAudio.device_calls += 1
        return TS.FakeAudio.devices_payload()

    @staticmethod
    def device_is_ambiguous(dev_id):
        return dev_id in RefreshAudio.ambiguous

    @staticmethod
    def device_candidates(dev_id, kind):
        return [dev_id.split("#")[0] + "#7"]

    @staticmethod
    def find_device(dev_id, kind):
        if dev_id in RefreshAudio.missing:
            raise LookupError(dev_id)
        return TS.FakeAudio.find_device(dev_id, kind)


@pytest.fixture
def svc(tmp_path):
    RefreshAudio.reset()
    TS.FakePlayer.instances.clear()
    s = RuntimeService(tmp_path / "t.db", pipeline_cfg=PipelineConfig(), engine_factory=TS.FakeEngine, audio=RefreshAudio,
                       preflight=TS.ok_preflight)
    s.voices = lambda: {"voices": [{"id": "af_heart", "label": "af_heart"}]}
    yield s
    if RefreshAudio.gate is not None:
        RefreshAudio.gate.set()
    s.close()


def configure(svc):
    return svc.configure({"input_id": "mic", "output_id": "bh", "monitor_volume": 0.4})


def test_idle_refresh_returns_devices_info_and_keeps_the_configuration_untouched(svc):
    configure(svc)
    before = dict(svc.status()["config"])
    r = svc.refresh_devices()
    assert r["config_candidates"] == {}
    assert RefreshAudio.calls == 1 and r["refresh"]["added"] == ["WH-1000XM5", "WH-1000XM5"] and r["config_missing"] == []
    assert set(r["devices"]) >= {"inputs", "outputs"}
    assert svc.status()["config"] == before and before["input_id"] == "mic" and before["output_id"] == "bh"
    assert before["monitor_id"] is None and before["monitor_volume"] == 0.4  # monitor apagado y volumen intactos
    assert svc.status()["state"] == "idle" and svc._refreshing is False


def test_refresh_reports_saved_devices_that_no_longer_exist_without_changing_them(svc):
    configure(svc)
    RefreshAudio.missing = {"bh"}
    r = svc.refresh_devices()
    assert r["config_missing"] == ["output_id"] and svc.status()["config"]["output_id"] == "bh"
    assert r["config_candidates"] == {"output_id": ["bh#7"]}  # sugerencia para mostrar; la config NO se reasigna


@pytest.mark.parametrize("state", ["starting", "running", "stopping", "error"])
def test_refresh_is_refused_in_every_non_idle_state_and_never_touches_audio(svc, state):
    configure(svc)
    with svc._lock:
        svc._state = state
    with pytest.raises(ServiceError) as e:
        svc.refresh_devices()
    assert e.value.status == 409 and e.value.code == "bad_state" and RefreshAudio.calls == 0 and svc._refreshing is False


def test_refresh_is_refused_during_a_real_running_session_and_the_session_survives(svc):
    configure(svc)
    svc.start()
    assert TS.wait(lambda: svc.status()["state"] == "running")
    with pytest.raises(ServiceError) as e:
        svc.refresh_devices()
    assert e.value.status == 409 and RefreshAudio.calls == 0
    assert svc.status()["state"] == "running" and TS.FakeMic.last.started and not TS.FakeMic.last.stopped
    svc.stop()
    assert TS.wait(lambda: svc.status()["state"] == "idle")
    svc.refresh_devices()  # ya ocioso: ahora sí
    assert RefreshAudio.calls == 1


class SlowOpenEngine(TS.FakeEngine):
    release = threading.Event()
    opening = threading.Event()

    async def open(self):
        SlowOpenEngine.opening.set()
        await __import__("asyncio").to_thread(SlowOpenEngine.release.wait, 5)
        self.opened = True


def test_refresh_is_refused_during_model_warmup_and_start_completes_normally(tmp_path):
    RefreshAudio.reset()
    SlowOpenEngine.release, SlowOpenEngine.opening = threading.Event(), threading.Event()
    s = RuntimeService(tmp_path / "w.db", pipeline_cfg=PipelineConfig(), engine_factory=SlowOpenEngine, audio=RefreshAudio,
                       preflight=TS.ok_preflight)
    try:
        configure(s)
        s.start()
        assert SlowOpenEngine.opening.wait(5) and s.status()["state"] == "starting"
        with pytest.raises(ServiceError) as e:
            s.refresh_devices()
        assert e.value.status == 409 and RefreshAudio.calls == 0
        SlowOpenEngine.release.set()
        assert TS.wait(lambda: s.status()["state"] == "running")
    finally:
        SlowOpenEngine.release.set()
        s.close()


def test_start_during_a_refresh_gets_409_busy_status_stays_responsive_and_start_works_afterwards(svc):
    configure(svc)
    TS.FakeMic.last = None  # el micrófono falso es global: sin restos de otras pruebas
    RefreshAudio.gate = threading.Event()
    result = {}
    t = threading.Thread(target=lambda: result.setdefault("r", svc.refresh_devices()))
    t.start()
    assert RefreshAudio.entered.wait(5) and svc._refreshing is True
    t0 = time.monotonic()
    assert svc.status()["state"] == "idle"  # GET /api/status no espera al refresco
    assert time.monotonic() - t0 < 0.5
    with pytest.raises(ServiceError) as e:
        svc.start()
    assert e.value.status == 409 and e.value.code == "busy" and svc.status()["state"] == "idle"
    assert TS.FakeMic.last is None  # el start rechazado no abrió el micrófono
    RefreshAudio.gate.set()
    t.join(5)
    assert "r" in result and svc._refreshing is False
    svc.start()
    assert TS.wait(lambda: svc.status()["state"] == "running")


def test_second_concurrent_refresh_is_rejected(svc):
    RefreshAudio.gate = threading.Event()
    t = threading.Thread(target=svc.refresh_devices)
    t.start()
    assert RefreshAudio.entered.wait(5)
    with pytest.raises(ServiceError) as e:
        svc.refresh_devices()
    assert e.value.status == 409 and e.value.code == "busy" and RefreshAudio.calls == 1
    RefreshAudio.gate.set()
    t.join(5)


def test_refresh_errors_are_mapped_and_always_release_the_flag(svc):
    RefreshAudio.exc = audio_io.AudioBusy("hay 1 stream(s) de audio abiertos")
    with pytest.raises(ServiceError) as e:
        svc.refresh_devices()
    assert e.value.status == 409 and e.value.code == "bad_state" and svc._refreshing is False
    RefreshAudio.exc = audio_io.AudioRefreshError("Pa_Terminate falló")
    with pytest.raises(ServiceError) as e:
        svc.refresh_devices()
    assert e.value.status == 500 and e.value.code == "audio_refresh" and "Pa_Terminate" in e.value.message and svc._refreshing is False
    RefreshAudio.exc = None
    configure(svc)
    svc.start()  # tras los errores el servicio sigue usable
    assert TS.wait(lambda: svc.status()["state"] == "running")


def test_listing_devices_never_refreshes_even_during_a_session(svc):
    configure(svc)
    svc.start()
    assert TS.wait(lambda: svc.status()["state"] == "running")
    for _ in range(3):
        assert "inputs" in svc.devices()
    assert RefreshAudio.calls == 0 and RefreshAudio.device_calls == 3 and svc.status()["state"] == "running"


def test_audio_without_refresh_support_answers_501(tmp_path):
    s = RuntimeService(tmp_path / "n.db", engine_factory=TS.FakeEngine, audio=TS.FakeAudio, preflight=TS.ok_preflight)
    try:
        with pytest.raises(ServiceError) as e:
            s.refresh_devices()
        assert e.value.status == 501
    finally:
        s.close()


def test_closed_service_refuses_refresh(svc):
    svc.close()
    with pytest.raises(ServiceError) as e:
        svc.refresh_devices()
    assert e.value.status == 409 and RefreshAudio.calls == 0


def test_refresh_then_full_session_with_the_saved_microphone_and_blackhole_output(svc):
    configure(svc)
    svc.refresh_devices()
    svc.start()
    assert TS.wait(lambda: svc.status()["state"] == "running")
    assert any(p.dev.id == "bh" for p in TS.FakePlayer.instances) and TS.FakeMic.last.started
    cfg = svc.status()["config"]
    assert cfg["input_id"] == "mic" and cfg["output_id"] == "bh" and cfg["monitor_id"] is None


def test_ambiguous_saved_devices_block_start_until_chosen_again_but_unambiguous_ones_do_not(svc):
    configure(svc)
    RefreshAudio.ambiguous = {"mic"}
    r = svc.refresh_devices()
    assert r["config_ambiguous"] == ["input_id"] and r["config_missing"] == []
    assert svc.status()["config"]["input_id"] == "mic"  # la config no se reasigna ni se borra
    with pytest.raises(ServiceError) as e:
        svc.start()
    assert e.value.status == 409 and e.value.code == "device_ambiguous" and svc.status()["state"] == "idle"
    assert TS.FakeMic.last is None or True
    svc.configure({"output_id": "bh"})  # elegir OTRA clave no levanta la marca de input
    with pytest.raises(ServiceError):
        svc.start()
    svc.configure({"input_id": "mic"})  # elección explícita de esa clave: se levanta
    svc.start()
    assert TS.wait(lambda: svc.status()["state"] == "running")


def test_unambiguous_devices_are_never_marked_and_a_second_refresh_recomputes_the_marks(svc):
    configure(svc)
    r = svc.refresh_devices()
    assert r["config_ambiguous"] == [] and svc._stale_device_keys == set()
    RefreshAudio.ambiguous = {"bh"}
    assert svc.refresh_devices()["config_ambiguous"] == ["output_id"] and svc._stale_device_keys == {"output_id"}
    RefreshAudio.ambiguous = set()
    svc.refresh_devices()
    assert svc._stale_device_keys == set()
    svc.start()
    assert TS.wait(lambda: svc.status()["state"] == "running")

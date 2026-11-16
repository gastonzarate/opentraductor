"""Pruebas del servicio con motor y audio FALSOS: sin modelos, sin micrófono,
sin hardware. Verifican estados, cola, historial solo-texto y errores."""
import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from runtime.config import PipelineConfig
from runtime.http_control import ServiceError
from runtime.service import RuntimeService


class FakeEngine:
    load_ms = {"asr": 10, "mt": 5, "tts": 20}

    def __init__(self, cfg, emit):
        self.cfg, self.emit = cfg, emit
        self.pushed = []
        self.closed = False
        self.opened = False

    async def open(self):
        self.opened = True

    async def start_stream(self, sid):
        self.sid = sid

    async def push(self, sid, pcm, t):
        self.pushed.append((sid, len(pcm), t))
        if len(self.pushed) == 3:  # simula un segmento completo
            await self.emit("endpoint", {"stream_id": sid, "seg_id": 1, "audio_pos_s": t})
            await self.emit("asr_partial", {"stream_id": sid, "seg_id": 1, "text": "hola"})
            await self.emit("asr_final", {"stream_id": sid, "seg_id": 1, "text": "hola mundo"})
            await self.emit("mt_final", {"stream_id": sid, "seg_id": 1, "text": "hello world"})
            await self.emit("tts_chunk", {"stream_id": sid, "seg_id": 1, "pcm": bytes(960), "sample_rate": 24000})

    async def end_stream(self, sid):
        pass

    async def close(self):
        self.closed = True

    def backlog(self):
        return {"asr_backlog_s": 0.0, "dropped_ms": 0, "queue_depth": 0}


class FakePlayer:
    instances = []
    lose_s = 0.0  # pérdida simulada por write (segundos)
    order = []  # orden de apertura entre salida y micrófono

    def __init__(self, dev, volume=1.0, max_queue_s=60.0, latency=None):
        self.dev, self.volume, self.writes, self.queued_s, self.stopped = dev, volume, [], 0.0, False
        self.dropped_s, self.partial_blocks, self.status_events = 0.0, 0, 0
        self.latency, self.latency_s, self.blocksize = latency, 0.020, 480  # valores 'reales' falsos
        FakePlayer.instances.append(self)

    @property
    def dropped_ms(self):
        return self.dropped_s * 1000.0

    def start(self):
        FakePlayer.order.append(("player", time.monotonic()))
    def stop(self): self.stopped = True
    def set_volume(self, v): self.volume = v
    def write(self, pcm, sr):
        self.writes.append((len(pcm), sr))
        self.dropped_s += FakePlayer.lose_s


class FakeMic:
    last = None
    emit_on_start = True  # el micrófono falso entrega un bloque al abrir (como un micro real)

    def __init__(self, dev, on_block, on_status=None, latency=None):
        self.on_block, self.started, self.stopped = on_block, False, False
        self.latency, self.latency_s, self.blocksize = latency, 0.030, 320  # valores 'reales' falsos
        self.on_status = on_status
        FakeMic.last = self

    def start(self):
        self.started = True
        FakePlayer.order.append(("mic", time.monotonic()))
        if FakeMic.emit_on_start:
            self.on_block(bytes(640))
    def stop(self): self.stopped = True


class FakeAudio:
    Player = FakePlayer
    MicCapture = FakeMic

    @staticmethod
    def devices_payload():
        return {"inputs": [{"id": "mic", "name": "Mic", "default": True, "virtual": False}],
                "outputs": [{"id": "spk", "name": "Spk", "default": True, "virtual": False},
                            {"id": "bh", "name": "BlackHole 2ch", "default": False, "virtual": True}]}

    @staticmethod
    def find_device(dev_id, kind):
        if dev_id not in ("mic", "spk", "bh"):
            raise LookupError(dev_id)
        return SimpleNamespace(id=dev_id, index=0)


def ok_preflight(cfg):
    return [SimpleNamespace(name="x", ok=True, detail="ok")]


def bad_preflight(cfg):
    return [SimpleNamespace(name="asr:parakeet", ok=False, detail="falta")]


def wait(cond, t=3.0):
    t0 = time.time()
    while time.time() - t0 < t:
        if cond():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def svc(tmp_path):
    FakePlayer.instances.clear()
    s = RuntimeService(tmp_path / "t.db", pipeline_cfg=PipelineConfig(), engine_factory=FakeEngine,
                       audio=FakeAudio, preflight=ok_preflight)
    s.voices = lambda: {"voices": [{"id": "af_heart", "label": "af_heart"}]}
    yield s
    s.close()


def events_of(svc):
    ev = []
    svc.subscribe(ev.append)
    return ev


def configure_ok(svc):
    return svc.configure({"input_id": "mic", "output_id": "spk"})


def test_start_requires_selection(svc):
    with pytest.raises(ServiceError) as e:
        svc.start()
    assert e.value.status == 400


def test_preflight_blocks_start(tmp_path):
    FakeMic.last = None
    s = RuntimeService(tmp_path / "t.db", engine_factory=FakeEngine, audio=FakeAudio, preflight=bad_preflight)
    try:
        s.configure({"input_id": "mic", "output_id": "spk"})
        with pytest.raises(ServiceError) as e:
            s.start()
        assert e.value.status == 503 and e.value.code == "preflight"
        assert FakeMic.last is None or not FakeMic.last.started
    finally:
        s.close()


def test_full_session_flow_and_history(svc):
    ev = events_of(svc)
    configure_ok(svc)
    svc.start()
    assert wait(lambda: svc.status()["state"] == "running")
    assert FakeMic.last.started  # el micrófono se abrió solo tras start()
    for _ in range(3):
        FakeMic.last.on_block(bytes(640))
    assert wait(lambda: any(e["type"] == "segment" for e in ev))
    seg = next(e for e in ev if e["type"] == "segment")
    assert seg["text_es"] == "hola mundo" and seg["text_en"] == "hello world"
    assert any(e["type"] == "partial" for e in ev)
    assert FakePlayer.instances[0].writes == [(960, 24000)]
    svc.stop()
    assert wait(lambda: svc.status()["state"] == "idle")
    assert FakeMic.last.stopped
    h = svc.history(10, 0)
    assert h["total"] == 1 and h["items"][0]["text_en"] == "hello world"
    assert "hello world" in svc.export_history("txt")
    assert [e["state"] for e in ev if e["type"] == "state"][-1] == "idle"


def test_start_twice_conflicts_and_stop_idempotent(svc):
    configure_ok(svc)
    svc.start()
    with pytest.raises(ServiceError) as e:
        svc.start()
    assert e.value.status == 409
    assert wait(lambda: svc.status()["state"] == "running")
    svc.stop(); svc.stop()
    assert wait(lambda: svc.status()["state"] == "idle")
    svc.stop()  # idempotente en idle


def test_configure_hot_and_cold_fields(svc):
    configure_ok(svc)
    svc.start()
    assert wait(lambda: svc.status()["state"] == "running")
    st = svc.configure({"endpoint_silence_ms": 120})
    assert st["config"]["endpoint_silence_ms"] == 120.0
    with pytest.raises(ServiceError) as e:
        svc.configure({"output_id": "bh"})
    assert e.value.status == 409 and e.value.code == "conflict"
    # mismo valor: no es cambio
    svc.configure({"output_id": "spk"})
    svc.stop()
    assert wait(lambda: svc.status()["state"] == "idle")


def test_configure_validation(svc):
    for bad in ({"nope": 1}, {"monitor_volume": 2}, {"endpoint_silence_ms": 5}, {"voice": "zz"}, {"input_id": ""}):
        with pytest.raises(ServiceError) as e:
            svc.configure(bad)
        assert e.value.status == 400


def test_config_persists_across_reopen(tmp_path):
    a = RuntimeService(tmp_path / "t.db", engine_factory=FakeEngine, audio=FakeAudio, preflight=ok_preflight)
    a.voices = lambda: {"voices": [{"id": "af_heart", "label": "x"}]}
    a.configure({"input_id": "mic", "output_id": "spk", "monitor_volume": 0.3})
    a.close()
    b = RuntimeService(tmp_path / "t.db", engine_factory=FakeEngine, audio=FakeAudio, preflight=ok_preflight)
    try:
        c = b.status()["config"]
        assert c["input_id"] == "mic" and c["monitor_volume"] == 0.3
    finally:
        b.close()


@pytest.mark.parametrize("saved, expected", [(None, 160.0), (250.0, 160.0), (300.0, 300.0), (120.0, 120.0)])
def test_saved_endpoint_legacy_250_uses_new_default(tmp_path, saved, expected):
    # T-58: un 250 guardado (default viejo, se guardaba con cualquier cambio) se toma como "sin valor"
    from runtime.text_store import TextStore
    if saved is not None:
        st = TextStore(tmp_path / "t.db")
        st.set_config({"endpoint_silence_ms": saved})
        st.close()
    pcfg = PipelineConfig()
    s = RuntimeService(tmp_path / "t.db", pipeline_cfg=pcfg, engine_factory=FakeEngine, audio=FakeAudio,
                       preflight=ok_preflight)
    try:
        assert s.status()["config"]["endpoint_silence_ms"] == expected
        assert pcfg.endpoint_silence_ms == expected
    finally:
        s.close()


def test_input_queue_overflow_is_visible_and_counted(svc):
    ev = events_of(svc)
    configure_ok(svc)
    svc.start()
    assert wait(lambda: svc.status()["state"] == "running")
    # engine lento: bloquea el feeder y llena la cola
    gate = threading.Event()

    async def slow_push(*a, **k):
        while not gate.is_set():
            await asyncio.sleep(0.01)

    svc._engine.push = slow_push
    for _ in range(1700):
        FakeMic.last.on_block(bytes(640))
    assert wait(lambda: any(e["type"] == "overflow" for e in ev))
    m = svc.status()["metrics"]
    assert m["overflow_count"] > 0 and m["dropped_ms"] > 0
    gate.set()
    svc.stop()
    assert wait(lambda: svc.status()["state"] == "idle", 8)


def test_open_failure_reports_error_and_recovers(tmp_path):
    FakeMic.last = None

    class Boom(FakeEngine):
        async def open(self):
            raise RuntimeError("sin modelos")

    s = RuntimeService(tmp_path / "t.db", engine_factory=Boom, audio=FakeAudio, preflight=ok_preflight)
    try:
        s.configure({"input_id": "mic", "output_id": "spk"})
        s.start()
        assert wait(lambda: s.status()["state"] == "error")
        st = s.status()
        assert "sin modelos" in st["error"] and st["models"]["asr"]["state"] == "error"
        assert FakeMic.last is None or not FakeMic.last.started
        s.stop()
        assert s.status()["state"] == "idle"
    finally:
        s.close()


def test_missing_device_goes_to_error(svc):
    svc.configure({"input_id": "ghost", "output_id": "spk"})
    svc.start()
    assert wait(lambda: svc.status()["state"] == "error")


def test_no_audio_in_database(tmp_path, svc):
    configure_ok(svc)
    svc.start()
    assert wait(lambda: svc.status()["state"] == "running")
    marker = bytes([0x7A, 0x13] * 320)
    for _ in range(3):
        FakeMic.last.on_block(marker)
    assert wait(lambda: svc.history(5, 0)["total"] >= 0)
    svc.stop()
    assert wait(lambda: svc.status()["state"] == "idle")
    svc._flush_writer()
    raw = b"".join(p.read_bytes() for p in tmp_path.glob("t.db*"))
    assert marker[:64] not in raw  # ni rastro del PCM capturado
    import sqlite3
    cols = {r[1] for r in sqlite3.connect(tmp_path / "t.db").execute("PRAGMA table_info(segments)")}
    assert not ({"pcm", "audio", "wav"} & cols)


def test_delete_history_while_session_open_keeps_saving(svc):
    configure_ok(svc)
    svc.start()
    assert wait(lambda: svc.status()["state"] == "running")
    svc.delete_history(None)
    for _ in range(3):
        FakeMic.last.on_block(bytes(640))
    svc.stop()
    assert wait(lambda: svc.status()["state"] == "idle")
    assert svc.history(10, 0)["total"] == 1


def test_stop_during_engine_load_never_opens_mic(tmp_path):
    """Stop mientras el motor carga: el micrófono no debe abrirse ni llegar a 'running'."""
    FakeMic.last = None
    FakePlayer.instances.clear()
    gate = threading.Event()

    class GatedEngine(FakeEngine):
        async def open(self):
            self.opened = True
            await asyncio.to_thread(gate.wait, 5)

    s = RuntimeService(tmp_path / "t.db", pipeline_cfg=PipelineConfig(), engine_factory=GatedEngine,
                       audio=FakeAudio, preflight=ok_preflight)
    try:
        ev = events_of(s)
        configure_ok(s)
        s.start()
        assert wait(lambda: s.status()["state"] == "starting")
        s.stop()
        gate.set()
        assert wait(lambda: s.status()["state"] == "idle")
        assert FakeMic.last is None or not FakeMic.last.started
        states = [e["state"] for e in ev if e["type"] == "state"]
        assert "running" not in states
        assert states[-1] == "idle"
    finally:
        s.close()


def test_stop_during_load_then_restart_runs_normally(tmp_path):
    """Tras un stop durante la carga, un nuevo start debe funcionar con normalidad."""
    FakeMic.last = None
    gate = threading.Event()

    class GatedOnce(FakeEngine):
        calls = 0

        async def open(self):
            GatedOnce.calls += 1
            if GatedOnce.calls == 1:
                await asyncio.to_thread(gate.wait, 5)

    s = RuntimeService(tmp_path / "t.db", pipeline_cfg=PipelineConfig(), engine_factory=GatedOnce,
                       audio=FakeAudio, preflight=ok_preflight)
    try:
        configure_ok(s)
        s.start()
        assert wait(lambda: s.status()["state"] == "starting")
        s.stop()
        gate.set()
        assert wait(lambda: s.status()["state"] == "idle")
        s.start()
        assert wait(lambda: s.status()["state"] == "running")
        assert FakeMic.last.started
        s.stop()
        assert wait(lambda: s.status()["state"] == "idle")
    finally:
        s.close()


def test_concurrent_start_only_one_wins(svc):
    """Dobles start concurrentes: exactamente uno inicia; el resto recibe 409."""
    configure_ok(svc)
    results = []
    barrier = threading.Barrier(8)

    def go():
        barrier.wait()
        try:
            svc.start()
            results.append("ok")
        except ServiceError as e:
            results.append(e.status)

    ths = [threading.Thread(target=go) for _ in range(8)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    assert results.count("ok") == 1
    assert all(r == 409 for r in results if r != "ok")
    assert wait(lambda: svc.status()["state"] == "running")
    svc.stop()
    assert wait(lambda: svc.status()["state"] == "idle")


def test_stop_immediately_after_start_never_runs(tmp_path):
    """Stop justo tras start, sin esperar 'starting': el start no debe revivir ni abrir mic."""
    FakeMic.last = None
    gate = threading.Event()

    class GatedEngine(FakeEngine):
        async def open(self):
            self.opened = True
            await asyncio.to_thread(gate.wait, 5)

    s = RuntimeService(tmp_path / "t.db", pipeline_cfg=PipelineConfig(), engine_factory=GatedEngine,
                       audio=FakeAudio, preflight=ok_preflight)
    try:
        ev = events_of(s)
        configure_ok(s)
        s.start()
        s.stop()
        gate.set()
        assert wait(lambda: s.status()["state"] == "idle")
        assert FakeMic.last is None or not FakeMic.last.started
        assert "running" not in [e["state"] for e in ev if e["type"] == "state"]
    finally:
        s.close()


def test_playback_loss_is_published_and_counted(tmp_path):
    """Pérdidas de reproducción: evento overflow y métrica visible, sin doble conteo."""
    FakePlayer.instances.clear()
    FakePlayer.lose_s = 0.5
    s = RuntimeService(tmp_path / "t.db", pipeline_cfg=PipelineConfig(), engine_factory=FakeEngine,
                       audio=FakeAudio, preflight=ok_preflight)
    try:
        ev = events_of(s)
        configure_ok(s)
        s.start()
        assert wait(lambda: s.status()["state"] == "running")
        for _ in range(3):
            FakeMic.last.on_block(bytes(640))
        assert wait(lambda: any(e["type"] == "overflow" and e["reason"] == "playback_queue" for e in ev))
        assert s.metrics()["playback_dropped_ms"] == 500
        assert s.metrics()["dropped_ms"] in (None, 0)  # la pérdida de reproducción no cuenta como entrada
    finally:
        FakePlayer.lose_s = 0.0
        s.close()


def test_writer_error_is_published(tmp_path, monkeypatch):
    """Un fallo al guardar texto debe llegar como evento de error, no silenciarse."""
    from runtime.text_store import TextStoreError

    s = RuntimeService(tmp_path / "t.db", pipeline_cfg=PipelineConfig(), engine_factory=FakeEngine,
                       audio=FakeAudio, preflight=ok_preflight)

    def boom(*a, **k):
        raise TextStoreError("disco lleno")

    monkeypatch.setattr(s._store, "add_segment", boom)
    try:
        ev = events_of(s)
        configure_ok(s)
        s.start()
        assert wait(lambda: s.status()["state"] == "running")
        for _ in range(3):
            FakeMic.last.on_block(bytes(640))
        assert wait(lambda: any(e["type"] == "error" and "Historial no guardado" in e["message"] for e in ev))
    finally:
        s.close()


def test_sqlite_error_publishes_and_writer_survives(tmp_path, monkeypatch):
    """Un error de SQLite (disco lleno, base bloqueada) se publica y el escritor sigue vivo."""
    import sqlite3

    s = RuntimeService(tmp_path / "t.db", pipeline_cfg=PipelineConfig(), engine_factory=FakeEngine,
                       audio=FakeAudio, preflight=ok_preflight)
    real_add = s._store.add_segment
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise sqlite3.OperationalError("database is locked")
        return real_add(*a, **k)

    monkeypatch.setattr(s._store, "add_segment", flaky)
    try:
        ev = events_of(s)
        configure_ok(s)
        s.start()
        assert wait(lambda: s.status()["state"] == "running")
        for _ in range(3):
            FakeMic.last.on_block(bytes(640))
        assert wait(lambda: any(e["type"] == "error" and "database is locked" in e["message"] for e in ev))
        assert s._writer.is_alive()
        s._flush_writer(timeout=2)  # el escritor responde: sigue procesando
        assert s._writer.is_alive()
        assert s.history(10, 0)["total"] == 0  # el segmento fallido no aparece como guardado
    finally:
        s.close()


def test_close_during_engine_load_never_opens_mic(tmp_path):
    """close() mientras el motor carga: no debe abrir mic ni dejar un start vivo."""
    FakeMic.last = None
    gate = threading.Event()

    class GatedEngine(FakeEngine):
        async def open(self):
            self.opened = True
            await asyncio.to_thread(gate.wait, 5)

    s = RuntimeService(tmp_path / "t.db", pipeline_cfg=PipelineConfig(), engine_factory=GatedEngine,
                       audio=FakeAudio, preflight=ok_preflight)
    configure_ok(s)
    s.start()
    assert wait(lambda: s.status()["state"] == "starting")
    closer = threading.Thread(target=s.close)
    closer.start()
    time.sleep(0.1)
    gate.set()
    closer.join(timeout=10)
    assert not closer.is_alive()
    assert FakeMic.last is None or not FakeMic.last.started
    with pytest.raises(ServiceError) as e:
        s.start()
    assert e.value.code == "closed"


def test_voice_labels_are_readable_and_ids_unchanged(tmp_path, monkeypatch):
    import runtime.config as cfgmod

    d = tmp_path / "snap" / "voices"
    d.mkdir(parents=True)
    for v in ("af_heart", "am_adam"):
        (d / f"{v}.safetensors").write_bytes(b"x")
    monkeypatch.setattr(cfgmod, "snapshot", lambda *a, **k: tmp_path / "snap")
    s = RuntimeService(tmp_path / "t.db", pipeline_cfg=PipelineConfig(), engine_factory=FakeEngine,
                       audio=FakeAudio, preflight=ok_preflight)
    try:
        voices = {v["id"]: v["label"] for v in RuntimeService.voices(s)["voices"]}
        assert voices == {"af_heart": "Heart · inglés (EE. UU.)", "am_adam": "Adam · inglés (EE. UU.)"}
    finally:
        s.close()


def test_raw_asr_text_reaches_event_and_database(tmp_path):
    """asr_final con text (normalizado) y text_raw: ambos llegan al evento y a SQLite."""

    class RawEngine(FakeEngine):
        async def push(self, sid, pcm, t):
            self.pushed.append(1)
            if len(self.pushed) == 3:
                await self.emit("endpoint", {"stream_id": sid, "seg_id": 1, "audio_pos_s": t})
                await self.emit("asr_final", {"stream_id": sid, "seg_id": 1,
                                              "text": "hola mundo", "text_raw": "hola  mundo!"})
                await self.emit("mt_final", {"stream_id": sid, "seg_id": 1, "text": "hello world"})

    s = RuntimeService(tmp_path / "t.db", pipeline_cfg=PipelineConfig(), engine_factory=RawEngine,
                       audio=FakeAudio, preflight=ok_preflight)
    try:
        ev = events_of(s)
        configure_ok(s)
        s.start()
        assert wait(lambda: s.status()["state"] == "running")
        for _ in range(3):
            FakeMic.last.on_block(bytes(640))
        assert wait(lambda: any(e["type"] == "segment" for e in ev))
        seg = next(e for e in ev if e["type"] == "segment")
        assert seg["text_es"] == "hola mundo" and seg["text_es_raw"] == "hola  mundo!"
        assert wait(lambda: s.history(10, 0)["total"] == 1)
        item = s.history(10, 0)["items"][0]
        assert item["text_es"] == "hola mundo" and item["text_es_raw"] == "hola  mundo!"
        assert "ES (ASR original): hola  mundo!" in s.export_history("txt")
        s.stop()
        assert wait(lambda: s.status()["state"] == "idle")
    finally:
        s.close()


def test_audio_latency_option_reaches_streams_and_defaults_to_none(tmp_path):
    FakePlayer.instances.clear()
    FakeMic.last = None
    s = RuntimeService(tmp_path / "t.db", pipeline_cfg=PipelineConfig(), engine_factory=FakeEngine,
                       audio=FakeAudio, preflight=ok_preflight, audio_latency="low")
    try:
        configure_ok(s)
        s.start()
        assert wait(lambda: s.status()["state"] == "running")
        assert FakePlayer.instances[-1].latency == "low" and FakeMic.last.latency == "low"
        assert s.status()["audio"]["latency_mode"] == "low"
    finally:
        s.close()
    s2 = RuntimeService(tmp_path / "t2.db", pipeline_cfg=PipelineConfig(), engine_factory=FakeEngine,
                        audio=FakeAudio, preflight=ok_preflight)
    try:
        configure_ok(s2)
        s2.start()
        assert wait(lambda: s2.status()["state"] == "running")
        assert FakePlayer.instances[-1].latency is None and FakeMic.last.latency is None
    finally:
        s2.close()


def test_invalid_audio_latency_is_rejected_at_construction(tmp_path):
    with pytest.raises(ValueError):
        RuntimeService(tmp_path / "t.db", pipeline_cfg=PipelineConfig(), engine_factory=FakeEngine,
                       audio=FakeAudio, preflight=ok_preflight, audio_latency="rapido")


def test_status_reports_real_stream_latency_and_blocksize(svc):
    configure_ok(svc)
    svc.start()
    assert wait(lambda: svc.status()["state"] == "running")
    audio = svc.status()["audio"]
    assert audio["output"] == {"stream_reported_latency_ms": 20.0, "blocksize": 480, "blocksize_variable": False}
    assert audio["input"] == {"stream_reported_latency_ms": 30.0, "blocksize": 320, "blocksize_variable": False}
    svc.stop()
    assert wait(lambda: svc.status()["state"] == "idle")


def test_same_device_opens_output_first_with_settle_time(tmp_path):
    FakePlayer.order.clear()
    s = RuntimeService(tmp_path / "t.db", pipeline_cfg=PipelineConfig(), engine_factory=FakeEngine,
                       audio=FakeAudio, preflight=ok_preflight)
    try:
        s.configure({"input_id": "bh", "output_id": "bh"})
        s.start()
        assert wait(lambda: s.status()["state"] == "running")
        kinds = [k for k, _ in FakePlayer.order]
        assert kinds == ["player", "mic"]
        gap = FakePlayer.order[1][1] - FakePlayer.order[0][1]
        assert gap >= 0.25  # SAME_DEVICE_SETTLE_S menos margen
        s.stop()
        assert wait(lambda: s.status()["state"] == "idle")
    finally:
        s.close()


def test_different_devices_do_not_wait(tmp_path):
    FakePlayer.order.clear()
    s = RuntimeService(tmp_path / "t.db", pipeline_cfg=PipelineConfig(), engine_factory=FakeEngine,
                       audio=FakeAudio, preflight=ok_preflight)
    try:
        configure_ok(s)  # mic y spk: dispositivos distintos
        s.start()
        assert wait(lambda: s.status()["state"] == "running")
        assert FakePlayer.order[1][1] - FakePlayer.order[0][1] < 0.25
        s.stop()
        assert wait(lambda: s.status()["state"] == "idle")
    finally:
        s.close()


def test_capture_that_emits_nothing_is_a_visible_error(tmp_path):
    """Micrófono sin callbacks iniciales: error visible, mic cerrado, nunca 'running' mudo."""
    FakeMic.last = None
    FakeMic.emit_on_start = False
    s = RuntimeService(tmp_path / "t.db", pipeline_cfg=PipelineConfig(), engine_factory=FakeEngine,
                       audio=FakeAudio, preflight=ok_preflight, first_block_timeout_s=0.3)
    try:
        ev = events_of(s)
        configure_ok(s)
        s.start()
        assert wait(lambda: s.status()["state"] == "error")
        st = s.status()
        assert "no entregó audio" in st["error"]
        assert "running" not in [e["state"] for e in ev if e["type"] == "state"]
        assert FakeMic.last.stopped is True
    finally:
        FakeMic.emit_on_start = True
        s.close()


def test_audio_latency_is_saved_validated_and_locked_while_running(tmp_path):
    db = tmp_path / "t.db"
    s = RuntimeService(db, pipeline_cfg=PipelineConfig(), engine_factory=FakeEngine,
                       audio=FakeAudio, preflight=ok_preflight)
    try:
        configure_ok(s)
        assert s.configure({"audio_latency": "low"})["config"]["audio_latency"] == "low"
        with pytest.raises(ServiceError) as e:
            s.configure({"audio_latency": "turbo"})
        assert e.value.status == 400
        s.start()
        assert wait(lambda: s.status()["state"] == "running")
        with pytest.raises(ServiceError) as e:
            s.configure({"audio_latency": "high"})
        assert e.value.status == 409
        s.stop()
        assert wait(lambda: s.status()["state"] == "idle")
    finally:
        s.close()
    # guardado en SQLite: otra instancia lo recupera sin flag
    s2 = RuntimeService(db, pipeline_cfg=PipelineConfig(), engine_factory=FakeEngine,
                        audio=FakeAudio, preflight=ok_preflight)
    try:
        assert s2.status()["config"]["audio_latency"] == "low"
    finally:
        s2.close()
    # con flag explícita (constructor) prevalece sobre el guardado
    s3 = RuntimeService(db, pipeline_cfg=PipelineConfig(), engine_factory=FakeEngine,
                        audio=FakeAudio, preflight=ok_preflight, audio_latency="high")
    try:
        assert s3.status()["config"]["audio_latency"] == "high"
    finally:
        s3.close()


def test_stale_callback_after_restart_does_not_mark_new_capture(tmp_path):
    """Un callback tardío de la sesión anterior no debe contar como primer bloque de la nueva."""
    FakeMic.last = None
    FakeMic.emit_on_start = True
    s = RuntimeService(tmp_path / "t.db", pipeline_cfg=PipelineConfig(), engine_factory=FakeEngine,
                       audio=FakeAudio, preflight=ok_preflight, first_block_timeout_s=0.5)
    try:
        configure_ok(s)
        s.start()
        assert wait(lambda: s.status()["state"] == "running")
        stale_on_block = FakeMic.last.on_block  # callback de la sesión 1
        s.stop()
        assert wait(lambda: s.status()["state"] == "idle")
        FakeMic.emit_on_start = False  # la sesión 2 no entrega audio por sí misma
        s.start()
        stale_on_block(bytes(640))  # llega tarde, desde la sesión 1, durante el arranque de la 2
        assert wait(lambda: s.status()["state"] == "error", t=3.0)
        assert "no entregó audio" in s.status()["error"]
    finally:
        FakeMic.emit_on_start = True
        s.close()


def test_session_snapshot_has_latency_mode_and_stream_diagnostics(tmp_path):
    import json

    s = RuntimeService(tmp_path / "t.db", pipeline_cfg=PipelineConfig(), engine_factory=FakeEngine,
                       audio=FakeAudio, preflight=ok_preflight, audio_latency="low")
    try:
        configure_ok(s)
        s.start()
        assert wait(lambda: s.status()["state"] == "running")
        s._flush_writer(timeout=2)
        sid = s._session_id
        row = s._store._conn.execute("SELECT config FROM sessions WHERE id = ?", (sid,)).fetchone()
        snap = json.loads(row["config"])
        assert snap["audio_latency"] == "low"
        assert snap["audio"]["latency_mode"] == "low"
        assert snap["audio"]["output"]["stream_reported_latency_ms"] == 20.0
        assert snap["audio"]["input"]["stream_reported_latency_ms"] == 30.0
        s.stop()
        assert wait(lambda: s.status()["state"] == "idle")
    finally:
        s.close()


def test_blocksize_zero_means_variable_not_zero_samples():
    from runtime.service import _stream_info

    info = _stream_info(0.02, 0)
    assert info["blocksize_variable"] is True and info["blocksize"] == 0
    assert _stream_info(0.02, 480)["blocksize_variable"] is False


def test_stale_callback_during_blocked_start_stream_does_not_enter_new_queue(tmp_path):
    """Callback de la sesión 1 mientras la sesión 2 espera a start_stream: no debe entrar en la cola nueva."""
    FakeMic.last = None
    FakeMic.emit_on_start = False
    gate = threading.Event()
    entered = threading.Event()

    class GatedStart(FakeEngine):
        block_start = False

        async def start_stream(self, sid):
            if GatedStart.block_start:
                entered.set()
                await asyncio.to_thread(gate.wait, 5)
            await super().start_stream(sid)

    s = RuntimeService(tmp_path / "t.db", pipeline_cfg=PipelineConfig(), engine_factory=GatedStart,
                       audio=FakeAudio, preflight=ok_preflight, first_block_timeout_s=0.5)
    try:
        configure_ok(s)
        # sesión 1 normal: un bloque del micrófono 1 la pone en running
        FakeMic.emit_on_start = True
        s.start()
        assert wait(lambda: s.status()["state"] == "running")
        stale_on_block = FakeMic.last.on_block
        stale_on_status = FakeMic.last.on_status
        s.stop()
        assert wait(lambda: s.status()["state"] == "idle")
        pushed_before = len(s._engine.pushed)  # pushes de la sesión 1 (su propio bloque)
        # sesión 2: start_stream bloqueado, el micrófono nuevo no entrega audio
        FakeMic.emit_on_start = False
        GatedStart.block_start = True
        s.start()
        assert entered.wait(3.0), "la sesión 2 debía estar dentro de start_stream"
        stale_on_block(bytes(640))  # callback viejo, con la cola nueva ya instalada
        stale_on_status("output underflow")
        gate.set()
        assert wait(lambda: s.status()["state"] == "error", t=3.0)
        assert len(s._engine.pushed) == pushed_before  # nada del micrófono viejo llegó al motor en la sesión 2
        assert s._m["audio_status"] is None  # ni el estado de PortAudio viejo contamina la sesión nueva
    finally:
        GatedStart.block_start = False
        FakeMic.emit_on_start = True
        s.close()


def test_subtitles_only_runs_without_player_and_persists(tmp_path):
    """Modo «solo subtítulos» (NO_OUTPUT): hay segmentos y texto, no se abre ningún reproductor."""
    from runtime.service import NO_OUTPUT

    FakePlayer.instances.clear()
    s = RuntimeService(tmp_path / "t.db", pipeline_cfg=PipelineConfig(), engine_factory=FakeEngine,
                       audio=FakeAudio, preflight=ok_preflight)
    s.voices = lambda: {"voices": [{"id": "af_heart", "label": "af_heart"}]}
    try:
        ev = events_of(s)
        s.configure({"input_id": "mic", "output_id": NO_OUTPUT})
        s.start()
        assert wait(lambda: s.status()["state"] == "running")
        for _ in range(3):
            FakeMic.last.on_block(bytes(640))
        assert wait(lambda: any(e["type"] == "segment" for e in ev))
        assert not FakePlayer.instances  # ningún Player abierto
        assert "output" not in s.status()["audio"]
        assert s.status()["config"]["output_id"] == NO_OUTPUT
        s.stop()
        assert wait(lambda: s.status()["state"] == "idle")
        assert s.history(10, 0)["total"] == 1
    finally:
        s.close()

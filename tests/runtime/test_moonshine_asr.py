"""Contrato del backend Moonshine con un ``moonshine_voice`` falso (sin modelos ni inferencia)."""
import asyncio
import sys
import types
from dataclasses import dataclass

import numpy as np
import pytest

from runtime import components as K
from runtime import config as C
from runtime.config import PipelineConfig


@dataclass
class Line:
    line_id: int
    text: str
    last_transcription_latency_ms: int = 7


@dataclass
class Ev:
    line: Line


class FakeStream:
    def __init__(self, interval):
        self.interval, self.listeners, self.n, self.closed, self.stopped = interval, [], 0, False, False

    def add_listener(self, fn): self.listeners.append(fn)
    def start(self): pass

    def add_audio(self, data, sr):
        assert isinstance(data, list) and sr == 16000
        self.n += 1
        for fn in self.listeners:
            fn(Ev(Line(1, "hola")))

    def stop(self):
        self.stopped = True
        for fn in self.listeners:
            fn(Ev(Line(1, "hola mundo")))
            fn(Ev(Line(2, "adios")))
            fn(types.SimpleNamespace(line=None))  # Error: se ignora

    def close(self): self.closed = True


class FakeTranscriber:
    last = None

    def __init__(self, model_path, model_arch, update_interval):
        self.args = (model_path, model_arch, update_interval)
        self.streams = []
        FakeTranscriber.last = self

    def create_stream(self, update_interval=None):
        s = FakeStream(update_interval)
        self.streams.append(s)
        return s


@pytest.fixture
def fake_moonshine(monkeypatch):
    import importlib.machinery

    m = types.ModuleType("moonshine_voice")
    m.__spec__ = importlib.machinery.ModuleSpec("moonshine_voice", None)
    m.Transcriber = FakeTranscriber
    m.ModelArch = types.SimpleNamespace(TINY_STREAMING=2, SMALL_STREAMING=4)
    monkeypatch.setitem(sys.modules, "moonshine_voice", m)


def pcm(ms=160):
    return np.zeros(16 * ms, dtype="<i2").tobytes()


def test_streamer_partial_final_and_flush(fake_moonshine):
    a = K.MoonshineStreamer("/x", "SMALL_STREAMING", 0.1)
    assert FakeTranscriber.last.args == ("/x", 4, 0.1)
    assert a.add(pcm()) == "hola"  # start implícito
    st = FakeTranscriber.last.streams[0]
    assert st.interval == 0.1
    assert a.finish() == "hola mundo adios"  # stop() = vaciado; líneas en orden
    assert st.stopped and st.closed
    assert a.finish() == ""  # idempotente
    a.start(); a.add(pcm())
    assert len(FakeTranscriber.last.streams) == 2  # un stream por segmento
    assert a.finish().startswith("hola")


def test_missing_package_is_clear(monkeypatch):
    monkeypatch.setitem(sys.modules, "moonshine_voice", None)
    with pytest.raises(ModuleNotFoundError, match="moonshine-voice"):
        K.MoonshineStreamer("/x", "TINY_STREAMING", 0.1)


def test_config_backend_validation_and_paths(monkeypatch):
    assert PipelineConfig().asr_backend == "parakeet"
    assert PipelineConfig.from_dict({"asr_backend": "moonshine-tiny"}).moonshine_update_interval_s == 0.1
    with pytest.raises(ValueError):
        PipelineConfig(asr_backend="nope")
    monkeypatch.setenv("OPENTRADUCTOR_MOONSHINE_DIR", "/m")
    assert str(C.moonshine_model_dir("moonshine-small")) == f"/m/small-streaming-es/{C.MOONSHINE_REV}"
    assert C.MOONSHINE_MODELS["moonshine-tiny"][1] == "TINY_STREAMING"


def test_preflight_checks_selected_backend(tmp_path, monkeypatch, fake_moonshine):
    from runtime.preflight import run_checks

    monkeypatch.setenv("OPENTRADUCTOR_MOONSHINE_DIR", str(tmp_path))
    names = [c.name for c in run_checks(PipelineConfig(asr_backend="moonshine-tiny"))]
    assert "asr:moonshine-tiny" in names and "asr:parakeet" not in names
    bad = [c for c in run_checks(PipelineConfig(asr_backend="moonshine-tiny")) if c.name == "asr:moonshine-tiny"][0]
    assert not bad.ok and "faltan" in bad.detail
    d = C.moonshine_model_dir("moonshine-tiny"); d.mkdir(parents=True)
    for f in ("encoder.ort", "decoder_kv.ort", "tokenizer.bin", "streaming_config.json"):
        (d / f).write_bytes(b"x")
    ok = [c for c in run_checks(PipelineConfig(asr_backend="moonshine-tiny")) if c.name == "asr:moonshine-tiny"][0]
    assert ok.ok


def test_force_offline_overrides_env(monkeypatch):
    import os
    from runtime.preflight import force_offline

    monkeypatch.setenv("HF_HUB_OFFLINE", "0"); monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    force_offline()
    assert os.environ["HF_HUB_OFFLINE"] == "1" and os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_engine_uses_separate_cpu_worker_for_moonshine():
    from runtime.pipeline import Engine

    async def emit(k, f): pass

    async def go():
        e = Engine(PipelineConfig(asr_backend="moonshine-small"), emit)
        assert e.asr_worker is not e.gpu and e.asr_worker is not e.cpu
        names = {e.asr_worker._ex._thread_name_prefix, e.gpu._ex._thread_name_prefix}
        assert names == {"asr", "mlx"}
        e2 = Engine(PipelineConfig(), emit)
        assert e2.asr_worker is e2.gpu  # Parakeet conserva el hilo MLX
        for x in (e, e2):
            await x.close()

    asyncio.run(go())


def test_close_releases_native_transcriber_and_stream(fake_moonshine, monkeypatch):
    closed = []
    monkeypatch.setattr(FakeTranscriber, "close", lambda self: closed.append("tr"), raising=False)
    a = K.MoonshineStreamer("/x", "TINY_STREAMING", 0.1)
    a.start(); a.add(pcm())
    st = FakeTranscriber.last.streams[0]
    a.close()
    assert st.stopped and st.closed and closed == ["tr"] and a.tr is None
    a.close()  # idempotente: no vuelve a cerrar
    assert closed == ["tr"]


def test_engine_close_closes_asr_on_its_worker():
    from runtime.pipeline import Engine

    seen = []

    class A:
        def close(self):
            import threading
            seen.append(threading.current_thread().name)

    async def emit(k, f): pass

    async def go():
        e = Engine(PipelineConfig(asr_backend="moonshine-tiny"), emit)
        e.asr = A()
        await e.close()

    asyncio.run(go())
    assert len(seen) == 1 and seen[0].startswith("asr")


def test_adapter_name_follows_asr_backend():
    from runtime.adapters.baseline import BaselineAdapter

    assert BaselineAdapter.name == "baseline-parakeet-marian-kokoro"  # histórico, sin cambios
    assert BaselineAdapter.name_for(PipelineConfig()) == "baseline-parakeet-marian-kokoro"
    assert BaselineAdapter.name_for(PipelineConfig(asr_backend="moonshine-tiny")) == "baseline-moonshine-tiny-marian-kokoro"

    async def go(cfg):
        a = BaselineAdapter()

        async def fake_open():  # Engine.open cargaría modelos: se sustituye
            pass

        from runtime import pipeline
        orig = pipeline.Engine.open
        pipeline.Engine.open = lambda self: fake_open()
        try:
            await a.open(cfg)
        finally:
            pipeline.Engine.open = orig
        n = a.name
        await a.engine.close()
        return n

    assert asyncio.run(go({"asr_backend": "moonshine-small"})) == "baseline-moonshine-small-marian-kokoro"
    assert asyncio.run(go({})) == "baseline-parakeet-marian-kokoro"

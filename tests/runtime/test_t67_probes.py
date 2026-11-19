"""T-67 sondas a) (Moonshine con CoreML por env) y d) (Kokoro precalentado al empezar la voz). Componentes falsos."""
import asyncio
import importlib.machinery
import sys
import time
import types

import pytest

from runtime import components as K

from test_speculative import FakeTTS, run_segment


# ------------------------------------------------------------------ a) proveedor ORT de Moonshine
class OptsTranscriber:
    calls = []
    fail_coreml = False

    def __init__(self, model_path, model_arch, update_interval, options=None):
        OptsTranscriber.calls.append(options)
        if options and OptsTranscriber.fail_coreml:
            raise RuntimeError("CoreML execution provider is not in this build")


@pytest.fixture
def fake_ms(monkeypatch, tmp_path):
    m = types.ModuleType("moonshine_voice")
    m.__spec__ = importlib.machinery.ModuleSpec("moonshine_voice", None)
    m.Transcriber = OptsTranscriber
    m.ModelArch = types.SimpleNamespace(TINY_STREAMING=2, SMALL_STREAMING=4)
    monkeypatch.setitem(sys.modules, "moonshine_voice", m)
    monkeypatch.setenv("OPENTRADUCTOR_COREML_CACHE", str(tmp_path / "cml"))
    OptsTranscriber.calls, OptsTranscriber.fail_coreml = [], False
    return tmp_path


def test_cpu_by_default_no_options(fake_ms, monkeypatch):
    monkeypatch.delenv("OPENTRADUCTOR_ASR_ORT_PROVIDER", raising=False)
    a = K.MoonshineStreamer("/x", "SMALL_STREAMING", 0.1)
    assert a.ort_provider == "cpu" and a.provider_error is None
    assert OptsTranscriber.calls == [None]  # camino de hoy: sin options


def test_coreml_env_passes_options(fake_ms, monkeypatch):
    monkeypatch.setenv("OPENTRADUCTOR_ASR_ORT_PROVIDER", "coreml")
    a = K.MoonshineStreamer("/x", "SMALL_STREAMING", 0.1)
    assert a.ort_provider == "coreml"
    assert OptsTranscriber.calls == [{"ort_provider": "coreml", "coreml_cache_dir": str(fake_ms / "cml")}]
    assert (fake_ms / "cml").is_dir()


def test_coreml_failure_falls_back_to_cpu(fake_ms, monkeypatch):
    monkeypatch.setenv("OPENTRADUCTOR_ASR_ORT_PROVIDER", "CoreML")
    OptsTranscriber.fail_coreml = True
    a = K.MoonshineStreamer("/x", "SMALL_STREAMING", 0.1)
    assert a.ort_provider == "cpu" and "not in this build" in a.provider_error
    assert OptsTranscriber.calls[-1] is None


def test_invalid_env_is_cpu(monkeypatch):
    monkeypatch.setenv("OPENTRADUCTOR_ASR_ORT_PROVIDER", "gpu")
    assert K.asr_ort_provider() == "cpu"


# ------------------------------------------------------------------ d) Kokoro precalentado
class WarmTTS(FakeTTS):
    def __init__(self):
        super().__init__()
        self.warms = 0

    def prewarm(self, text="a"):
        self.warms += 1
        time.sleep(0.005)
        return 5.0


def _run(**kw):
    async def go():
        tts = WarmTTS()
        e, ev = await run_segment(partial="hola mundo", final="hola mundo", spec=False, tts=tts, **kw)
        await e.close()
        return e, ev, tts
    return asyncio.run(go())


def test_tts_prewarm_off_by_default(monkeypatch):
    monkeypatch.delenv("OPENTRADUCTOR_TTS_PREWARM", raising=False)
    e, ev, tts = _run()
    assert tts.warms == 0 and "tts" not in e.prewarm_stats


def test_tts_prewarm_env_once_per_segment_output_unchanged(monkeypatch):
    monkeypatch.delenv("OPENTRADUCTOR_TTS_PREWARM", raising=False)
    _, ev_off, _ = _run()
    monkeypatch.setenv("OPENTRADUCTOR_TTS_PREWARM", "1")
    e, ev, tts = _run()
    assert tts.warms == 1 and e.prewarm_stats["tts"]["started"] == 1
    assert e.prewarm_stats["tts"]["cost_ms"] == [5.0]
    chunks = lambda evs: [f["pcm"] for k, f in evs if k == "tts_chunk"]
    assert chunks(ev) == chunks(ev_off)  # no emite nada propio


def test_tts_prewarm_cfg_and_env_off(monkeypatch):
    monkeypatch.setenv("OPENTRADUCTOR_TTS_PREWARM", "0")
    _, _, tts = _run(tts_prewarm_on_speech=True)
    assert tts.warms == 0
    monkeypatch.delenv("OPENTRADUCTOR_TTS_PREWARM")
    _, _, tts = _run(tts_prewarm_on_speech=True)
    assert tts.warms == 1


def test_tts_prewarm_skipped_with_parakeet(monkeypatch):
    monkeypatch.setenv("OPENTRADUCTOR_TTS_PREWARM", "1")
    e, _, tts = _run(backend="parakeet")  # ASR en el hilo MLX: no se le roba GPU
    assert tts.warms == 0 and e.prewarm_stats["tts"]["skipped_hot"] == 1

"""T-67: flag del ASR en proceso aparte y que el pipeline elija la clase según el flag."""
from runtime import asr_process
from runtime.config import PipelineConfig


def test_flag_default_on(monkeypatch):
    monkeypatch.delenv("OPENTRADUCTOR_ASR_PROCESS", raising=False)
    assert PipelineConfig().asr_process is True
    assert asr_process.asr_process_enabled(False) is False
    assert asr_process.asr_process_enabled(True) is True


def test_env_manda(monkeypatch):
    monkeypatch.setenv("OPENTRADUCTOR_ASR_PROCESS", "1")
    assert asr_process.asr_process_enabled(False) is True
    monkeypatch.setenv("OPENTRADUCTOR_ASR_PROCESS", "0")
    assert asr_process.asr_process_enabled(True) is False


def test_make_asr_elige_proceso(monkeypatch):
    from runtime import pipeline as P

    made = {}

    class Fake:
        def __init__(self, *a):
            made["cls"] = type(self).__name__

    monkeypatch.setattr(asr_process, "ProcessMoonshineStreamer", type("ProcessMoonshineStreamer", (Fake,), {}))
    monkeypatch.setattr(P.K, "MoonshineStreamer", type("MoonshineStreamer", (Fake,), {}))
    eng = P.Engine.__new__(P.Engine)
    eng.cfg = PipelineConfig(asr_backend="moonshine-tiny")
    monkeypatch.setenv("OPENTRADUCTOR_ASR_PROCESS", "1")
    eng._make_asr()
    assert made["cls"] == "ProcessMoonshineStreamer"
    monkeypatch.setenv("OPENTRADUCTOR_ASR_PROCESS", "0")
    eng._make_asr()
    assert made["cls"] == "MoonshineStreamer"

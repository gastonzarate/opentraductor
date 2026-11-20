import sys

from runtime import pipeline as P


def test_switch_interval_env(monkeypatch):
    monkeypatch.delenv("OPENTRADUCTOR_SWITCH_INTERVAL_MS", raising=False)
    assert P._switch_interval_s() is None
    monkeypatch.setenv("OPENTRADUCTOR_SWITCH_INTERVAL_MS", "1")
    assert P._switch_interval_s() == 0.001
    for bad in ("0", "51", "abc", "-1", "1.5", ""):
        monkeypatch.setenv("OPENTRADUCTOR_SWITCH_INTERVAL_MS", bad)
        assert P._switch_interval_s() is None


def test_engine_sets_switch_interval(monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "setswitchinterval", calls.append)
    monkeypatch.setenv("OPENTRADUCTOR_SWITCH_INTERVAL_MS", "2")
    eng = P.Engine(P.PipelineConfig(), lambda *a, **k: None)
    try:
        assert calls == [0.002]
    finally:
        for w in {eng.gpu, eng.cpu, eng.asr_worker}:
            w.shutdown()


def test_engine_default_leaves_switch_interval(monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "setswitchinterval", calls.append)
    monkeypatch.delenv("OPENTRADUCTOR_SWITCH_INTERVAL_MS", raising=False)
    eng = P.Engine(P.PipelineConfig(), lambda *a, **k: None)
    try:
        assert calls == []
    finally:
        for w in {eng.gpu, eng.cpu, eng.asr_worker}:
            w.shutdown()

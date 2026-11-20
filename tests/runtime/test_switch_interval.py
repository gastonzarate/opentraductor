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
        assert P._switch_interval_s(1) == 0.001  # env inválido: manda la config
    monkeypatch.setenv("OPENTRADUCTOR_SWITCH_INTERVAL_MS", "off")
    assert P._switch_interval_s(1) is None
    monkeypatch.delenv("OPENTRADUCTOR_SWITCH_INTERVAL_MS")
    assert P._switch_interval_s(3) == 0.003
    assert P._switch_interval_s(None) is None


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


def test_engine_default_sets_1ms(monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "setswitchinterval", calls.append)
    monkeypatch.delenv("OPENTRADUCTOR_SWITCH_INTERVAL_MS", raising=False)
    assert P.PipelineConfig().switch_interval_ms == 1
    eng = P.Engine(P.PipelineConfig(), lambda *a, **k: None)
    try:
        assert calls == [0.001]
    finally:
        for w in {eng.gpu, eng.cpu, eng.asr_worker}:
            w.shutdown()


def test_engine_none_leaves_switch_interval(monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "setswitchinterval", calls.append)
    monkeypatch.delenv("OPENTRADUCTOR_SWITCH_INTERVAL_MS", raising=False)
    eng = P.Engine(P.PipelineConfig(switch_interval_ms=None), lambda *a, **k: None)
    try:
        assert calls == []
    finally:
        for w in {eng.gpu, eng.cpu, eng.asr_worker}:
            w.shutdown()

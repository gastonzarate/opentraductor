"""T-67 fin de frase adaptativo: espera corta si el parcial cierra frase, larga si termina en conector."""
import asyncio

import pytest

from runtime import components as K
from runtime.config import PipelineConfig
from runtime.pipeline import Engine

from test_asr_spec_finish import PatternVAD
from test_speculative import WIN, FakeMT, FakeTTS

# T-67 (19/11): estos tests cubren el camino previo a encender adapt+stream+spec_finish por defecto; se fija
# explícito y cada test enciende lo que mide.
LEGACY_T67 = dict(endpoint_adaptive=False, endpoint_short_ms=96.0, asr_spec_finish_on_pause=False,
                  asr_spec_finish_silence_ms=80.0, mt_stream_tts=False, tts_first_chunk_max_words=8)



@pytest.mark.parametrize("text,kind", [
    ("", "unknown"),
    ("   ", "unknown"),
    ("el granizo destruyó toda la plantación", "complete"),
    ("el granizo destruyó toda la", "continue"),
    ("fuimos al centro y", "continue"),
    ("me dijo que", "continue"),
    ("está bien pero", "continue"),
    ("no sé o sea", "continue"),
    ("es lo que hay, es decir", "continue"),
    ("bueno este", "continue"),
    ("lo hacemos mañana,", "continue"),
    ("¿vamos o no?", "complete"),
    ("creo que si", "complete"),  # "si"/"sí" sin tilde: puede cerrar
    ("ya está.", "complete"),
    ("terminamos y.", "complete"),  # la puntuación final manda
    ("Pero", "continue"),
])
def test_phrase_endpoint_kind(text, kind):
    assert K.phrase_endpoint_kind(text) == kind


class TextASR:
    """Parcial fijo; el final agrega cuántos bytes de voz recibió (para contar segmentos y voz)."""

    def __init__(self, partial):
        self.partial, self.n, self.finishes = partial, 0, []

    def start(self):
        self.n = 0

    def add(self, pcm):
        self.n += len(pcm) - pcm.count(0)
        return self.partial

    def finish(self):
        self.finishes.append(self.n)
        return f"{self.partial} b{self.n}"


async def _run(pattern, partial, **cfg_kw):
    events = []

    async def emit(kind, f):
        events.append((kind, f))

    cfg = PipelineConfig(asr_backend="moonshine-tiny", speculative_mt_tts=False, endpoint_silence_ms=160,
                         asr_chunk_ms=160, min_speech_ms=0, mt_kv_prefill_on_pause=False, **{**LEGACY_T67, **cfg_kw})
    e = Engine(cfg, emit)
    e.asr, e.mt, e.tts = TextASR(partial), FakeMT(), FakeTTS()
    e.vad_proto = PatternVAD(pattern)
    e._out_q = asyncio.Queue(maxsize=8)
    e._out_task = asyncio.ensure_future(e._speaker_loop())
    await e.start_stream("s")
    for i in range(len(pattern) + 20):
        voice = i < len(pattern) and pattern[i]
        await e.push("s", (b"\x01" if voice else b"\x00") * (2 * WIN), 0.0)
        await asyncio.sleep(0.02)
    await e.end_stream("s")
    await e.close()
    return e, events


def _go(pattern, partial, **kw):
    return asyncio.run(_run(pattern, partial, **kw))


def _endpoints(ev):
    return [f for k, f in ev if k == "endpoint"]


# voz, pausa de 6 ventanas (192 ms: > corta 96, > base 160, < larga 480), voz otra vez
PAUSE = [1] * 10 + [0] * 6 + [1] * 6


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv("OPENTRADUCTOR_ENDPOINT_ADAPTIVE", raising=False)
    monkeypatch.delenv("OPENTRADUCTOR_ASR_SPEC_FINISH", raising=False)
    monkeypatch.delenv("OPENTRADUCTOR_ENDPOINT_SHORT_MS", raising=False)


def test_t67_defaults_on():
    """T-67 (19/11): fin de frase adaptativo con espera corta 64 y MT en streaming a Kokoro (1.er fragmento de 4
    palabras), encendidos por defecto tras el A/B t67-asrfin/n2. El finish especulativo del ASR queda apagado."""
    c = PipelineConfig()
    assert c.endpoint_adaptive is True and c.endpoint_short_ms == 64.0 and c.endpoint_long_ms == 480.0
    assert c.mt_stream_tts is True and c.tts_first_chunk_max_words == 2
    assert c.asr_spec_finish_on_pause is False and c.asr_spec_finish_silence_ms == 80.0


def test_default_engine_closes_complete_at_64():
    e, ev = _go([1] * 10, "el granizo destruyó la plantación", **{k: getattr(PipelineConfig(), k) for k in LEGACY_T67})
    eps = _endpoints(ev)
    assert len(eps) == 1 and eps[0]["wait_reason"] == "complete" and eps[0]["wait_ms"] == 64
    assert e.asr.finishes[-1] == 10 * 2 * WIN  # no se pierde voz


def test_off_by_default_is_fixed():
    e, ev = _go(PAUSE, "fuimos al centro y")
    eps = _endpoints(ev)
    assert len(eps) == 2 and all(f["wait_reason"] == "fixed" and f["wait_ms"] == 160 for f in eps)


def test_continue_waits_long_and_keeps_one_segment():
    e, ev = _go(PAUSE, "fuimos al centro y", endpoint_adaptive=True)
    eps = _endpoints(ev)
    assert len(eps) == 1 and eps[0]["wait_reason"] == "continue" and eps[0]["wait_ms"] == 480
    assert e.asr.finishes == [16 * 2 * WIN]  # toda la voz en un solo segmento


def test_complete_closes_short():
    e, ev = _go([1] * 10, "el granizo destruyó la plantación", endpoint_adaptive=True)
    eps = _endpoints(ev)
    assert len(eps) == 1 and eps[0]["wait_reason"] == "complete" and eps[0]["wait_ms"] == 96
    assert e.asr.finishes == [10 * 2 * WIN]  # no se pierde voz al cerrar antes
    # cierra 3 ventanas de silencio (96 ms) después de la voz, no 5 (160 ms)
    assert eps[0]["audio_pos_s"] == pytest.approx(13 * WIN / 16000)


def test_short_never_above_base_nor_long_below():
    e, ev = _go([1] * 10, "la plantación", endpoint_adaptive=True, endpoint_short_ms=320)
    assert _endpoints(ev)[0]["wait_ms"] == 160
    e, ev = _go(PAUSE, "fuimos al centro y", endpoint_adaptive=True, endpoint_long_ms=64)
    assert {f["wait_ms"] for f in _endpoints(ev)} == {160}


def test_env_short_ms_overrides_only_complete(monkeypatch):
    monkeypatch.setenv("OPENTRADUCTOR_ENDPOINT_SHORT_MS", "64")
    _, ev = _go([1] * 10, "el granizo destruyó la plantación", endpoint_adaptive=True)
    eps = _endpoints(ev)
    assert eps[0]["wait_reason"] == "complete" and eps[0]["wait_ms"] == 64
    assert eps[0]["audio_pos_s"] == pytest.approx(12 * WIN / 16000)
    _, ev = _go(PAUSE, "fuimos al centro y", endpoint_adaptive=True)
    assert len(_endpoints(ev)) == 1  # "continue" sigue esperando largo
    _, ev = _go(PAUSE, "fuimos al centro y")
    assert {f["wait_ms"] for f in _endpoints(ev)} == {160}  # sin el flag, nada cambia
    for bad in ("0", "-5", "abc", ""):
        monkeypatch.setenv("OPENTRADUCTOR_ENDPOINT_SHORT_MS", bad)
        _, ev = _go([1] * 10, "el granizo destruyó la plantación", endpoint_adaptive=True)
        assert _endpoints(ev)[0]["wait_ms"] == 96


def test_env_forces_on_and_off(monkeypatch):
    monkeypatch.setenv("OPENTRADUCTOR_ENDPOINT_ADAPTIVE", "1")
    _, ev = _go(PAUSE, "fuimos al centro y")
    assert len(_endpoints(ev)) == 1
    monkeypatch.setenv("OPENTRADUCTOR_ENDPOINT_ADAPTIVE", "0")
    _, ev = _go(PAUSE, "fuimos al centro y", endpoint_adaptive=True)
    assert len(_endpoints(ev)) == 2


def test_adapter_endpoint_lleva_wait_reason_en_unit_reason():
    """T-67: el A/B de bench veía wait_reason=None porque AdapterEvent no tiene ese campo."""
    import asyncio

    from runtime.adapters.baseline import BaselineAdapter

    async def go():
        a = BaselineAdapter()
        await a._emit("endpoint", {"stream_id": "s1", "audio_pos_s": 1.0, "forced": False, "seg_id": 2,
                                   "wait_ms": 96.0, "wait_reason": "complete"})
        await a._emit("asr_final", {"stream_id": "s1", "text": "Hola.", "seg_id": 2})
        return a._q.get_nowait(), a._q.get_nowait()

    ep, fin = asyncio.run(go())
    assert ep.unit_reason == "ep:complete:96"
    assert fin.unit_reason is None


def test_adapter_no_rompe_con_eventos_fuera_del_contrato():
    """spec_result (T-73), simul_clause/_mt (T-67 paso 2) y discarded no tienen EventKind: antes KeyError en _emit
    (dentro del Engine). Ahora no se reenvían al arnés y quedan en describe()['engine_events']."""
    import asyncio

    from runtime.adapters.baseline import BaselineAdapter

    a = BaselineAdapter()

    async def go():
        for k in ("spec_result", "simul_clause", "simul_clause_mt", "discarded"):
            await a._emit(k, {"stream_id": "s", "seg_id": 1, "spec": True, "pcm": b"x"})
        await a._emit("asr_final", {"stream_id": "s", "seg_id": 1, "text": "hola"})

    asyncio.run(go())
    assert a._q.qsize() == 1
    kinds = [e["kind"] for e in a.engine_events]
    assert kinds == ["spec_result", "simul_clause", "simul_clause_mt", "discarded"]
    assert all("pcm" not in e for e in a.engine_events)
    assert a.describe()["engine_events"][0]["spec"] is True

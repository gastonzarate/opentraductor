"""T-58 fin de frase especulativo: finish del ASR adelantado a la pausa VAD de 80 ms (componentes falsos)."""
import asyncio

from runtime.config import PipelineConfig
from runtime.pipeline import Engine

from test_speculative import WIN, FakeMT, FakeTTS

# T-67 (19/11): estos tests cubren el camino previo a encender adapt+stream+spec_finish por defecto; se fija
# explícito y cada test enciende lo que mide.
LEGACY_T67 = dict(endpoint_adaptive=False, endpoint_short_ms=96.0, asr_spec_finish_on_pause=False,
                  asr_spec_finish_silence_ms=80.0, mt_stream_tts=False, tts_first_chunk_max_words=8)



class PatternVAD:
    """Probabilidad por ventana según un patrón de 1 (voz) y 0 (silencio); silencio al terminarse."""

    def __init__(self, pattern):
        self.pattern, self.n = pattern, 0

    def reset(self): pass

    def reset_states(self): pass

    def probs(self, pcm):
        out = []
        for i in range(len(pcm) // 2 // WIN):
            v = self.pattern[self.n] if self.n < len(self.pattern) else 0
            out.append(((i + 1) * WIN, 0.9 if v else 0.0))
            self.n += 1
        return out


class RecASR:
    """El texto es la cantidad de bytes de VOZ (no nulos) recibidos desde el último start(): detecta voz perdida
    o duplicada. El silencio llega en ceros y no cuenta (los cortes de chunk difieren entre caminos)."""

    def __init__(self):
        self.n, self.starts, self.finishes = 0, 0, []

    def start(self):
        self.n, self.starts = 0, self.starts + 1

    def add(self, pcm):
        self.n += len(pcm) - pcm.count(0)
        return f"b{self.n}"

    def finish(self):
        self.finishes.append(self.n)
        return f"b{self.n}"


async def _run(pattern, mt=None, **cfg_kw):
    events = []

    async def emit(kind, f):
        events.append((kind, f))

    cfg = PipelineConfig(asr_backend="moonshine-tiny", speculative_mt_tts=False, endpoint_silence_ms=160,
                         asr_chunk_ms=160, min_speech_ms=0, mt_kv_prefill_on_pause=False, **{**LEGACY_T67, **cfg_kw})
    e = Engine(cfg, emit)
    e.asr, e.mt, e.tts = RecASR(), mt or FakeMT(), FakeTTS()
    e.vad_proto = PatternVAD(pattern)
    e._out_q = asyncio.Queue(maxsize=8)
    e._out_task = asyncio.ensure_future(e._speaker_loop())
    await e.start_stream("s")
    for i in range(len(pattern) + 8):
        voice = i < len(pattern) and pattern[i]
        await e.push("s", (b"\x01" if voice else b"\x00") * (2 * WIN), 0.0)
        await asyncio.sleep(0.02)
    await e.end_stream("s")
    await e.close()
    return e, events


def _go(pattern, **kw):
    return asyncio.run(_run(pattern, **kw))


def _finals(ev):
    return [f["text"] for k, f in ev if k == "asr_final"]


SIMPLE = [1] * 10
# voz, pausa de 4 ventanas (128 ms: >= 80, < endpoint 160), voz otra vez, silencio hasta el endpoint
RESUME = [1] * 10 + [0] * 4 + [1] * 6


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("OPENTRADUCTOR_ASR_SPEC_FINISH", raising=False)
    e, ev = _go(SIMPLE)
    assert e.asr_spec_stats["started"] == 0 and len(e.asr.finishes) == 1


def test_hit_uses_early_finish_once(monkeypatch):
    monkeypatch.delenv("OPENTRADUCTOR_ASR_SPEC_FINISH", raising=False)
    e, ev = _go(SIMPLE, asr_spec_finish_on_pause=True)
    s = e.asr_spec_stats
    assert s["started"] == 1 and s["hit"] == 1 and s["discard_voice"] == 0 and len(s["finish_ms"]) == 1
    assert len(e.asr.finishes) == 1  # no hay un 2.º finish en el endpoint
    assert len(_finals(ev)) == 1
    assert [f["text"] for k, f in ev if k == "mt_final"] == ["EN " + _finals(ev)[0]]


def test_hit_has_all_the_voice(monkeypatch):
    monkeypatch.delenv("OPENTRADUCTOR_ASR_SPEC_FINISH", raising=False)
    off, _ = _go(SIMPLE)
    on, _ = _go(SIMPLE, asr_spec_finish_on_pause=True)
    # el finish adelantado ve exactamente la misma voz que el camino normal
    assert on.asr.finishes[0] == off.asr.finishes[0] == 10 * 2 * WIN


def test_voice_resumes_discards_and_refeeds_whole_segment(monkeypatch):
    monkeypatch.delenv("OPENTRADUCTOR_ASR_SPEC_FINISH", raising=False)
    off, ev_off = _go(RESUME)
    on, ev_on = _go(RESUME, asr_spec_finish_on_pause=True)
    s = on.asr_spec_stats
    assert s["discard_voice"] == 1 and len(s["refeed_ms"]) == 1
    assert s["started"] == 2 and s["hit"] == 1  # la 2.ª pausa (la del endpoint) sí acierta
    assert len(_finals(ev_on)) == len(_finals(ev_off)) == 1
    # ni voz perdida ni duplicada al re-alimentar: misma voz que el camino normal
    assert on.asr.finishes[-1] == off.asr.finishes[-1] == 16 * 2 * WIN


def test_env_forces_on_and_off(monkeypatch):
    monkeypatch.setenv("OPENTRADUCTOR_ASR_SPEC_FINISH", "1")
    e, _ = _go(SIMPLE)
    assert e.asr_spec_stats["hit"] == 1
    monkeypatch.setenv("OPENTRADUCTOR_ASR_SPEC_FINISH", "0")
    e, _ = _go(SIMPLE, asr_spec_finish_on_pause=True)
    assert e.asr_spec_stats["started"] == 0


def test_two_segments_independent(monkeypatch):
    monkeypatch.delenv("OPENTRADUCTOR_ASR_SPEC_FINISH", raising=False)
    off, ev_off = _go(SIMPLE + [0] * 10 + SIMPLE)
    on, ev_on = _go(SIMPLE + [0] * 10 + SIMPLE, asr_spec_finish_on_pause=True)
    assert on.asr_spec_stats["hit"] == 2 and on.asr_spec_stats["discard_voice"] == 0
    assert len(_finals(ev_on)) == len(_finals(ev_off)) == 2


# T-67: pausa de 1 ventana (32 ms) dentro de la frase
SHORT_PAUSE = [1] * 10 + [0] * 1 + [1] * 6


def test_env_ms_moves_the_threshold(monkeypatch):
    monkeypatch.delenv("OPENTRADUCTOR_ASR_SPEC_FINISH", raising=False)
    monkeypatch.delenv("OPENTRADUCTOR_ASR_SPEC_FINISH_MS", raising=False)
    base, ev_base = _go(SHORT_PAUSE, asr_spec_finish_on_pause=True)
    assert base.asr_spec_stats["started"] == 1  # la pausa de 32 ms no llega a 80
    monkeypatch.setenv("OPENTRADUCTOR_ASR_SPEC_FINISH_MS", "32")
    on, ev_on = _go(SHORT_PAUSE, asr_spec_finish_on_pause=True)
    s = on.asr_spec_stats
    assert s["started"] == 2 and s["discard_voice"] == 1 and s["hit"] == 1
    assert on.asr.finishes[-1] == base.asr.finishes[-1] == 16 * 2 * WIN  # misma voz
    assert _finals(ev_on) == _finals(ev_base)


def test_env_ms_invalid_is_ignored(monkeypatch):
    monkeypatch.setenv("OPENTRADUCTOR_ASR_SPEC_FINISH_MS", "x")
    e, _ = _go(SHORT_PAUSE, asr_spec_finish_on_pause=True)
    assert e.asr_spec_stats["started"] == 1


class ContinueASR(RecASR):
    def add(self, pcm):
        super().add(pcm)
        return "vamos y"

    def finish(self):
        super().finish()
        return "vamos y"


def test_adaptive_skips_early_finish_on_continue(monkeypatch):
    monkeypatch.delenv("OPENTRADUCTOR_ASR_SPEC_FINISH", raising=False)
    monkeypatch.setenv("OPENTRADUCTOR_ASR_SPEC_FINISH_MS", "32")
    monkeypatch.setenv("OPENTRADUCTOR_ENDPOINT_ADAPTIVE", "1")

    async def run():
        events = []

        async def emit(kind, f):
            events.append((kind, f))

        cfg = PipelineConfig(asr_backend="moonshine-tiny", speculative_mt_tts=False, endpoint_silence_ms=160,
                             asr_chunk_ms=160, min_speech_ms=0, mt_kv_prefill_on_pause=False,
                             asr_spec_finish_on_pause=True)
        e = Engine(cfg, emit)
        e.asr, e.mt, e.tts = ContinueASR(), FakeMT(), FakeTTS()
        e.vad_proto = PatternVAD(SHORT_PAUSE)
        e._out_q = asyncio.Queue(maxsize=8)
        e._out_task = asyncio.ensure_future(e._speaker_loop())
        await e.start_stream("s")
        for i in range(len(SHORT_PAUSE) + 20):
            voice = i < len(SHORT_PAUSE) and SHORT_PAUSE[i]
            await e.push("s", (b"\x01" if voice else b"\x00") * (2 * WIN), 0.0)
            await asyncio.sleep(0.02)
        await e.end_stream("s")
        await e.close()
        return e, events

    e, ev = asyncio.run(run())
    assert e.asr_spec_stats["started"] == 0  # parcial en conector: nunca cierra el stream antes del endpoint
    assert e.asr_spec_stats["discard_voice"] == 0
    assert len(e.asr.finishes) == 1 and e.asr.finishes[0] == 16 * 2 * WIN

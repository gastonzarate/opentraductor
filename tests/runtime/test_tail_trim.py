"""T-47: recorte del silencio final de cada chunk Kokoro en el pipeline baseline (sin MLX ni modelos)."""
import numpy as np
import pytest

from runtime import __main__ as cli
from runtime import components as K
from runtime.config import PipelineConfig

SR = 24000


def _pcm(x) -> bytes:
    return np.asarray(x, dtype="<i2").tobytes()


def _tone(ms: float, amp: int = 8000) -> np.ndarray:
    n = int(SR * ms / 1000)
    return (amp * np.sin(2 * np.pi * 220 * np.arange(n) / SR)).astype(np.int16)


def _silence(ms: float, amp: int = 0) -> np.ndarray:
    n = int(SR * ms / 1000)
    rng = np.random.default_rng(0)
    return (rng.standard_normal(n) * amp).astype(np.int16) if amp else np.zeros(n, np.int16)


def test_trims_tail_keeping_80ms_and_exact_prefix():
    voice, tail = _tone(300), _silence(900)
    pcm = _pcm(np.concatenate([voice, tail]))
    out = K.trim_tail_pcm(pcm, SR)
    keep = len(out) // 2
    assert pcm.startswith(out)  # prefijo exacto, bit a bit
    # termina en el fin del último bloque sonoro (alineado desde el final) + 80 ms
    assert abs(keep - (voice.size + int(SR * 0.080))) <= int(SR * 0.005)
    assert keep < len(pcm) // 2


def test_internal_silence_untouched():
    x = np.concatenate([_tone(200), _silence(400), _tone(200), _silence(500)])
    out = K.trim_tail_pcm(_pcm(x), SR)
    kept = np.frombuffer(out, dtype="<i2")
    assert kept.size >= int(SR * 0.800)  # la pausa interna de 400 ms sigue entera
    assert np.array_equal(kept, x[: kept.size])


def test_quiet_noise_below_minus60_is_trimmed_but_louder_is_kept():
    quiet = _silence(600, amp=10)  # ~ -70 dBFS
    loud = _silence(600, amp=200)  # ~ -44 dBFS
    v = _tone(200)
    assert len(K.trim_tail_pcm(_pcm(np.concatenate([v, quiet])), SR)) < 2 * (v.size + quiet.size)
    pcm_loud = _pcm(np.concatenate([v, loud]))
    assert K.trim_tail_pcm(pcm_loud, SR) is pcm_loud


@pytest.mark.parametrize("pcm", [b"", _pcm(_silence(500)), b"\x01\x02\x03", _pcm(_tone(100))])
def test_passthrough_cases_return_same_object(pcm):
    # vacío, todo silencio (nunca vaciar), largo impar y sin cola silenciosa: sin cambios
    assert K.trim_tail_pcm(pcm, SR) is pcm


def test_invalid_sample_rate_passthrough():
    pcm = _pcm(np.concatenate([_tone(100), _silence(500)]))
    assert K.trim_tail_pcm(pcm, 0) is pcm


def test_matches_reference_rule_on_random_chunks():
    """Misma decisión que la regla de incremental-tts-tail-v1/trim.py (reimplementada aquí, en s16/32768)."""
    rng = np.random.default_rng(42)
    n = int(SR * 0.005)
    for _ in range(50):
        parts = [_tone(rng.uniform(20, 400), amp=int(rng.uniform(50, 20000)))]
        parts.append((rng.standard_normal(int(SR * rng.uniform(0, 1.2))) * rng.uniform(0, 40)).astype(np.int16))
        x = np.concatenate(parts)[: int(rng.integers(1, 40000))]
        L = x.size
        k, r = divmod(L, n)
        f = x.astype(np.float64)
        db = [20 * np.log10(max(np.sqrt(np.mean(f[r + i * n: r + (i + 1) * n] ** 2)), 1e-300) / 32768) for i in range(k)]
        sonoros = [i for i, d in enumerate(db) if d > -60]
        end = r + (sonoros[-1] + 1) * n if sonoros else 0
        if not end and r and 20 * np.log10(max(np.sqrt(np.mean(f[:r] ** 2)), 1e-300) / 32768) > -60:
            end = r
        expected = L if not end else min(L, end + int(SR * 0.080))
        assert K.tail_keep_samples(x, SR) == expected


def test_kokoro_chunk_applies_flag_only_when_on():
    tts = object.__new__(K.KokoroTTS)
    tts.tail_trimmed_ms_total = 0.0
    pcm = _pcm(np.concatenate([_tone(200), _silence(500)]))
    tts.trim_tail = False
    assert tts._chunk(pcm, "hi").pcm is pcm
    assert tts.tail_trimmed_ms_total == 0.0
    tts.trim_tail = True
    ch = tts._chunk(pcm, "hi")
    assert pcm.startswith(ch.pcm) and len(ch.pcm) < len(pcm)
    assert ch.sample_rate == SR and ch.text == "hi"
    assert tts.tail_trimmed_ms_total == pytest.approx((len(pcm) - len(ch.pcm)) / 2 / SR * 1000)


def test_config_default_off_and_from_dict():
    assert PipelineConfig().tts_trim_tail is False
    assert PipelineConfig.from_dict({"tts_trim_tail": True}).tts_trim_tail is True


def _cfg(tmp_path, *extra):
    a = cli.build_parser().parse_args(["run", "--db", str(tmp_path / "h.sqlite3"), *extra])
    svc, cfg = cli.build_service(a)
    svc.close()
    return cfg


def test_cli_flag_and_env(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENTRADUCTOR_TTS_TRIM_TAIL", raising=False)
    assert _cfg(tmp_path).tts_trim_tail is False
    assert _cfg(tmp_path, "--tts-trim-tail", "on").tts_trim_tail is True
    monkeypatch.setenv("OPENTRADUCTOR_TTS_TRIM_TAIL", "1")
    assert _cfg(tmp_path).tts_trim_tail is True
    assert _cfg(tmp_path, "--tts-trim-tail", "off").tts_trim_tail is False  # el flag explícito revierte


def test_spec_fingerprint_includes_tail_flag():
    from runtime.pipeline import Engine

    async def emit(*_):
        pass

    a = Engine(PipelineConfig(), emit)._spec_fp()
    b = Engine(PipelineConfig(tts_trim_tail=True), emit)._spec_fp()
    assert a != b

"""Pruebas de audio_io con sounddevice FALSO: sin hardware, sin micrófono."""
import numpy as np
import pytest

from runtime import audio_io


class FakeStream:
    instances = []
    fail_start = False

    def __init__(self, **kw):
        self.kw = kw
        self.started = False
        self.closed = False
        FakeStream.instances.append(self)

    def start(self):
        if FakeStream.fail_start:
            raise OSError("arranque fallido")
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True


class FakeSD:
    InputStream = FakeStream
    OutputStream = FakeStream


@pytest.fixture(autouse=True)
def fake_sd(monkeypatch):
    FakeStream.instances.clear()
    FakeStream.fail_start = False
    monkeypatch.setattr(audio_io, "_sd", lambda: FakeSD)
    yield


def dev(default_sr=48000.0, max_in=1, max_out=2):
    return audio_io.Device("mic", "Mic", 0, max_in, max_out, default_sr, True, True)


def test_mic_fallback_sets_rate_before_callback_can_run(monkeypatch):
    """Un callback que llega durante start() debe usar ya la tasa real (48 kHz -> 16 kHz)."""
    blocks = []
    m = audio_io.MicCapture(dev(), blocks.append)
    orig_start = FakeStream.start

    def start_with_early_callback(self):
        if self.kw["samplerate"] == audio_io.TARGET_SR:
            raise OSError("16k no soportado")  # el dispositivo rechaza 16 kHz
        orig_start(self)
        # PortAudio puede llamar al callback en cuanto arranca el stream
        indata = np.zeros((self.kw["blocksize"], 1), dtype=np.float32)
        self.kw["callback"](indata, indata.shape[0], None, None)

    monkeypatch.setattr(FakeStream, "start", start_with_early_callback)
    m.start()
    assert m.sr == 48000
    # 20 ms a 48 kHz remuestreados a 16 kHz = 320 muestras int16 = 640 bytes
    assert len(blocks) == 1 and len(blocks[0]) == 320 * 2


def test_mic_failed_start_closes_stream_and_resets_rate():
    FakeStream.fail_start = True
    m = audio_io.MicCapture(dev(default_sr=16000.0), lambda b: None)
    with pytest.raises(RuntimeError):
        m.start()
    assert m.sr == audio_io.TARGET_SR
    assert FakeStream.instances and all(s.closed for s in FakeStream.instances)
    assert m._stream is None


def test_mic_callback_resamples_with_real_rate():
    out = []
    m = audio_io.MicCapture(dev(default_sr=48000.0), out.append)
    m.start()  # 16 kHz acepta el fake: sr = 16000, sin remuestreo
    assert m.sr == 16000
    m._cb(np.zeros((320, 1), dtype=np.float32), 320, None, None)
    assert len(out) == 1 and len(out[0]) == 320 * 2


def test_player_failed_start_closes_stream():
    FakeStream.fail_start = True
    p = audio_io.Player(dev())
    with pytest.raises(OSError):
        p.start()
    assert FakeStream.instances[-1].closed


def test_player_counts_status_flags_and_drops():
    p = audio_io.Player(dev(default_sr=16000.0), max_queue_s=0.5)
    p._cb(np.zeros((160, 1), dtype=np.float32), 160, None, "output underflow")
    assert p.status_events == 1 and p.last_status == "output underflow"
    p.write((np.zeros(16000, dtype="<i2")).tobytes(), 16000)  # 1 s > 0.5 s de cola
    assert p.dropped_s > 0 and p.dropped_ms == pytest.approx(p.dropped_s * 1000)


def test_latency_mode_maps_to_sounddevice_and_default_is_unchanged():
    assert audio_io.latency_arg("default") is None
    assert audio_io.latency_arg("low") == "low" and audio_io.latency_arg("high") == "high"
    with pytest.raises(ValueError):
        audio_io.latency_arg("turbo")


def test_mic_passes_latency_and_reports_real_values():
    m = audio_io.MicCapture(dev(default_sr=48000.0), lambda b: None, latency="low")
    m.start()
    kw = FakeStream.instances[-1].kw
    assert kw["latency"] == "low"
    default = audio_io.MicCapture(dev(), lambda b: None)
    default.start()
    assert FakeStream.instances[-1].kw["latency"] is None


def test_player_passes_latency_and_reports_real_values():
    p = audio_io.Player(dev(default_sr=16000.0), latency="high")
    p.start()
    assert FakeStream.instances[-1].kw["latency"] == "high"


def test_normal_end_of_utterance_is_a_partial_block_not_a_failure():
    """El final normal de un fragmento rellena el bloque a medias: se cuenta como bloque parcial,
    sin errores de PortAudio, sin descarte y sin tratarlo como corte de reproducción."""
    p = audio_io.Player(dev(default_sr=16000.0))
    p.write(np.zeros(100, dtype="<i2").tobytes(), 16000)  # 100 muestras: fin de fragmento
    out = np.zeros((256, 1), dtype=np.float32)
    p._cb(out, 256, None, None)  # el bloque de 256 se completa con 100 muestras y silencio
    assert p.partial_blocks == 1
    assert p.underruns == 1  # alias obsoleto, mismo valor
    assert p.status_events == 0 and p.dropped_s == 0.0
    # un callback sin nada que reproducir no cuenta como bloque parcial
    p._cb(out, 256, None, None)
    assert p.partial_blocks == 1


def _pcm(x):
    return (np.asarray(x) * 32767).astype("<i2").tobytes()


def test_clear_fades_out_instead_of_cutting_the_wave():
    """clear() no corta la onda: el próximo bloque sale atenuado hasta cero (sin clic)."""
    p = audio_io.Player(dev(default_sr=16000.0))
    p.write(_pcm(np.full(4000, 0.5)), 16000)
    out = np.zeros((160, 1), dtype=np.float32)
    p._cb(out, 160, None, None)  # arranca la reproducción
    p.clear()
    assert p.queued_s == 0
    out = np.zeros((256, 1), dtype=np.float32)
    p._cb(out, 256, None, None)
    n = int(16000 * audio_io.CLEAR_FADE_MS / 1000)
    assert out[0, 0] == pytest.approx(0.5, abs=1e-3)       # sin salto al empezar el fundido
    assert out[n - 1, 0] == pytest.approx(0.0, abs=1e-3)   # termina en cero
    assert np.all(np.diff(out[:n, 0]) <= 1e-6)             # decreciente, sin escalones
    assert np.all(out[n:, 0] == 0)                         # y después silencio
    p._cb(out, 256, None, None)
    assert np.all(out == 0)


def test_clear_on_empty_queue_is_silent():
    p = audio_io.Player(dev(default_sr=16000.0))
    p.clear()
    out = np.ones((128, 1), dtype=np.float32)
    p._cb(out, 128, None, None)
    assert np.all(out == 0)


def test_audio_starting_from_silence_fades_in():
    p = audio_io.Player(dev(default_sr=16000.0))
    p.write(_pcm(np.full(1000, 0.5)), 16000)
    out = np.zeros((256, 1), dtype=np.float32)
    p._cb(out, 256, None, None)
    k = int(16000 * audio_io.EDGE_FADE_MS / 1000)
    assert out[0, 0] == 0.0
    assert np.all(np.diff(out[:k, 0]) > 0)
    assert out[k, 0] == pytest.approx(0.5, abs=1e-3)
    # el bloque siguiente es continuo: no se vuelve a atenuar nada
    out2 = np.zeros((256, 1), dtype=np.float32)
    p._cb(out2, 256, None, None)
    assert np.all(out2[:, 0] == pytest.approx(0.5, abs=1e-3))


def test_queue_running_dry_mid_phrase_fades_out_the_last_samples():
    p = audio_io.Player(dev(default_sr=16000.0))
    p.write(_pcm(np.full(200, 0.5)), 16000)
    out = np.zeros((256, 1), dtype=np.float32)
    p._cb(out, 256, None, None)
    k = int(16000 * audio_io.EDGE_FADE_MS / 1000)
    assert out[199, 0] == pytest.approx(0.0, abs=1e-6)
    assert out[200 - k - 1, 0] == pytest.approx(0.5, abs=1e-3)
    assert np.all(out[200:, 0] == 0)

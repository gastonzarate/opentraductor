"""Refresco de PortAudio (audio_io) con un sounddevice FALSO que imita lo relevante: lista congelada en _initialize, PortAudio terminado
entre _terminate y _initialize, contador _initialized. Sin hardware, sin micrófono, sin audio real."""
import threading
import time
from types import SimpleNamespace

import pytest

from runtime import audio_io as A


def dev(name, i=0, o=0, sr=48000.0):
    return {"name": name, "max_input_channels": i, "max_output_channels": o, "default_samplerate": sr}


BASE = [dev("MacBook Pro Microphone", 1, 0), dev("MacBook Pro Speakers", 0, 2), dev("BlackHole 2ch", 2, 2)]
WH = [dev("WH-1000XM5", 1, 0), dev("WH-1000XM5", 0, 2)]


class FakeStream:
    instances = []
    fail_start = False
    raise_on_stop = False
    close_error = False  # como sounddevice: el error de Pa_CloseStream solo se VE con ignore_errors=False

    def __init__(self, **kw):
        self.kw, self.started, self.closed = kw, False, False
        self.latency = 0.02
        self.blocksize = kw.get("blocksize") or 480
        FakeStream.instances.append(self)

    def start(self):
        if FakeStream.fail_start:
            raise OSError("arranque fallido")
        self.started = True

    def stop(self):
        if FakeStream.raise_on_stop:
            raise OSError("stop falló")
        self.started = False

    def close(self, ignore_errors=True):
        self.close_calls = getattr(self, "close_calls", []) + [ignore_errors]
        if FakeStream.close_error and not ignore_errors:
            raise OSError("Error closing stream (Pa_CloseStream)")
        self.closed = True


class FakeSD:
    """PortAudio falso: ve los dispositivos que había al inicializar; ``system`` es lo que hay realmente."""
    __version__ = "0.5.6"

    def __init__(self, system):
        self.system = list(system)
        self._visible = list(system)
        self._initialized = 1
        self.window = False  # True entre _terminate y _initialize: PortAudio no está inicializado
        self.calls = []
        self.fail_terminate = False
        self.fail_initialize = 0  # cuántas llamadas a _initialize fallan
        self.terminate_delay = 0.0
        self.queries_in_window = 0

        class _Default:
            device = (0, 1)
        self.default = _Default()
        self.InputStream = FakeStream
        self.OutputStream = FakeStream

    def query_devices(self):
        if self.window or self._initialized == 0:
            self.queries_in_window += 1
            raise RuntimeError("PortAudio no inicializado")
        return [dict(d) for d in self._visible]

    def _terminate(self):
        self.calls.append("terminate")
        if self.fail_terminate:
            raise RuntimeError("Pa_Terminate falló")
        self.window = True
        self._initialized -= 1
        if self.terminate_delay:
            time.sleep(self.terminate_delay)

    def _initialize(self):
        self.calls.append("initialize")
        if self.fail_initialize > 0:
            self.fail_initialize -= 1
            raise RuntimeError("Pa_Initialize falló")
        self._visible = list(self.system)
        self._initialized += 1
        self.window = False


@pytest.fixture
def sd(monkeypatch):
    fake = FakeSD(BASE)
    monkeypatch.setattr(A, "_sd", lambda: fake)
    monkeypatch.setattr(A, "_generation", 0)
    monkeypatch.setattr(A, "_broken", False)
    monkeypatch.setattr(A, "_open_streams", set())
    monkeypatch.setattr(A, "_unsafe_streams", set())
    monkeypatch.setattr(A, "time", SimpleNamespace(sleep=lambda s: None))  # sin esperas reales en los reintentos
    FakeStream.instances.clear()
    FakeStream.fail_start = FakeStream.raise_on_stop = FakeStream.close_error = False
    return fake


def names(devs):
    return [d.name for d in devs]


# ---------------------------------------------------------------- el problema y su arreglo
def test_without_refresh_a_device_connected_later_is_invisible_and_refresh_shows_it(sd):
    assert names(A.list_devices()) == [d["name"] for d in BASE]
    sd.system = BASE + WH  # el usuario conecta los auriculares con la app abierta
    assert "WH-1000XM5" not in names(A.list_devices())  # cache de PortAudio: el defecto que se vio
    info = A.refresh_devices()
    devs = A.list_devices()
    assert names(devs).count("WH-1000XM5") == 2
    assert info["added"] == ["WH-1000XM5", "WH-1000XM5"] and info["removed"] == [] and info["generation"] == 1
    assert info["devices_before"] == 3 and info["devices_after"] == 5
    assert sd._initialized == 1 and sd.calls == ["terminate", "initialize"]
    assert {"WH-1000XM5#in", "WH-1000XM5#out"} <= {d.id for d in devs}  # homónimos de distinta dirección: id estable, sin índice
    assert A.find_device("WH-1000XM5#in", "in").max_in > 0 and A.find_device("WH-1000XM5#out", "out").max_out > 0
    assert not any(d.ambiguous for d in devs)


def test_removed_devices_are_reported_and_ids_of_unique_devices_are_stable(sd):
    sd.system = [BASE[0], BASE[2]]
    info = A.refresh_devices()
    assert info["removed"] == ["MacBook Pro Speakers"] and info["added"] == []
    assert A.find_device("BlackHole 2ch", "out").index == 1  # el índice cambió, el id (nombre) no


def test_devices_payload_exposes_generation_and_health_without_breaking_old_keys(sd):
    p = A.devices_payload()
    assert set(p) >= {"inputs", "outputs", "generation", "audio_ok"} and p["generation"] == 0 and p["audio_ok"] is True
    A.refresh_devices()
    assert A.devices_payload()["generation"] == 1


# ---------------------------------------------------------------- nunca con streams abiertos
@pytest.mark.parametrize("make", ["player", "mic"])
def test_refresh_is_refused_while_a_stream_is_open_and_allowed_after_stop(sd, make):
    d = A.find_device("BlackHole 2ch", "out" if make == "player" else "in")
    obj = A.Player(d) if make == "player" else A.MicCapture(d, lambda b: None)
    obj.start()
    assert A.open_stream_count() == 1
    with pytest.raises(A.AudioBusy):
        A.refresh_devices()
    assert sd.calls == [] and A.generation() == 0  # no se tocó PortAudio
    obj.stop()
    assert A.open_stream_count() == 0
    A.refresh_devices()
    assert sd.calls == ["terminate", "initialize"]


def test_stream_that_fails_to_start_is_closed_and_never_counted(sd):
    FakeStream.fail_start = True
    with pytest.raises(OSError):
        A.Player(A.find_device("BlackHole 2ch", "out")).start()
    with pytest.raises(RuntimeError):
        A.MicCapture(A.find_device("BlackHole 2ch", "in"), lambda b: None).start()
    assert A.open_stream_count() == 0 and all(s.closed for s in FakeStream.instances)
    A.refresh_devices()


def test_mic_fallback_sample_rate_counts_only_the_stream_that_stays_open(sd, monkeypatch):
    monkeypatch.setattr(A, "prewarm_resample", lambda: None)
    attempts = []
    orig_start = FakeStream.start

    def start(self):
        attempts.append(self.kw["samplerate"])
        if self.kw["samplerate"] == A.TARGET_SR:
            raise OSError("16 kHz no soportado")
        orig_start(self)

    monkeypatch.setattr(FakeStream, "start", start)
    m = A.MicCapture(A.find_device("BlackHole 2ch", "in"), lambda b: None)
    m.start()
    assert attempts[0] == A.TARGET_SR and len(attempts) == 2 and A.open_stream_count() == 1
    m.stop()
    assert A.open_stream_count() == 0


def test_stop_unregisters_even_if_stream_stop_raises(sd):
    p = A.Player(A.find_device("BlackHole 2ch", "out"))
    p.start()
    FakeStream.raise_on_stop = True
    with pytest.raises(OSError):
        p.stop()
    assert A.open_stream_count() == 0 and FakeStream.instances[-1].closed
    A.refresh_devices()


# ---------------------------------------------------------------- precondiciones de las funciones privadas
def test_foreign_portaudio_references_block_refresh(sd):
    sd._initialized = 2
    with pytest.raises(A.AudioRefreshError, match="2 referencias"):
        A.refresh_devices()
    assert sd.calls == []


@pytest.mark.parametrize("version", ["0.4.0", "0.6.0", None])
def test_unverified_sounddevice_version_never_calls_private_functions(sd, version):
    sd.__version__ = version
    with pytest.raises(A.AudioRefreshError, match="no verificado"):
        A.refresh_devices()
    assert sd.calls == []


def test_missing_private_api_is_refused(sd, monkeypatch):
    class Bare:
        __version__ = "0.5.6"

        def query_devices(self):
            return []
    bare = Bare()
    monkeypatch.setattr(A, "_sd", lambda: bare)
    with pytest.raises(A.AudioRefreshError):
        A.refresh_devices()


# ---------------------------------------------------------------- recuperación
def test_terminate_failure_leaves_everything_unchanged(sd):
    sd.fail_terminate = True
    sd.system = BASE + WH
    with pytest.raises(A.AudioRefreshError, match="Pa_Terminate"):
        A.refresh_devices()
    assert sd.calls == ["terminate"] and sd._initialized == 1 and A.generation() == 0
    assert names(A.list_devices()) == [d["name"] for d in BASE]  # sigue usable con la lista anterior


def test_initialize_is_retried_and_succeeds(sd):
    sd.fail_initialize = 2
    sd.system = BASE + WH
    info = A.refresh_devices()
    assert sd.calls == ["terminate", "initialize", "initialize", "initialize"] and info["generation"] == 1
    assert "WH-1000XM5" in names(A.list_devices())


def test_initialize_failure_marks_audio_broken_and_a_later_refresh_recovers(sd):
    sd.fail_initialize = 3
    with pytest.raises(A.AudioRefreshError, match="no pudo reinicializarse"):
        A.refresh_devices()
    assert A.list_devices() == [] and A.devices_payload()["audio_ok"] is False and sd._initialized == 0
    sd.fail_initialize = 0
    sd.system = BASE + WH
    info = A.refresh_devices()  # en estado roto no vuelve a llamar a _terminate: solo reinicializa
    assert sd.calls[-1] == "initialize" and sd.calls.count("terminate") == 1
    assert A.devices_payload()["audio_ok"] is True and info["devices_before"] == 0 and "WH-1000XM5" in names(A.list_devices())


# ---------------------------------------------------------------- concurrencia con la enumeración
def test_enumeration_never_sees_portaudio_in_the_terminated_window(sd):
    sd.terminate_delay = 0.01
    stop = threading.Event()
    empties, errors = [], []

    def lister():
        while not stop.is_set():
            try:
                if not A.list_devices():
                    empties.append(1)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

    threads = [threading.Thread(target=lister) for _ in range(4)]
    [t.start() for t in threads]
    for _ in range(15):
        A.refresh_devices()
    stop.set()
    [t.join(5) for t in threads]
    assert sd.queries_in_window == 0 and not empties and not errors
    assert A.generation() == 15


def test_stream_opening_waits_for_a_running_refresh_and_uses_a_fresh_device(sd):
    sd.terminate_delay = 0.2
    d_old = A.find_device("BlackHole 2ch", "out")
    t = threading.Thread(target=A.refresh_devices)
    t.start()
    time.sleep(0.05)  # el refresco está dentro de la ventana
    t0 = time.monotonic()
    with pytest.raises(A.StaleDevice):  # espera al lock y entonces ve que su dispositivo es de la generación anterior
        A.Player(d_old).start()
    assert time.monotonic() - t0 > 0.05
    t.join(5)
    assert FakeStream.instances == []  # no llegó a abrir ningún stream con un índice viejo
    A.Player(A.find_device("BlackHole 2ch", "out")).start()


def test_stale_device_is_rejected_by_player_and_mic_with_a_clear_error_and_nothing_opened(sd):
    d_out, d_in = A.find_device("BlackHole 2ch", "out"), A.find_device("BlackHole 2ch", "in")
    A.refresh_devices()
    with pytest.raises(A.StaleDevice, match="volvé a elegirlos"):
        A.Player(d_out).start()
    with pytest.raises(A.StaleDevice):
        A.MicCapture(d_in, lambda b: None).start()  # no se disfraza de «No se pudo abrir el micrófono»
    assert FakeStream.instances == [] and A.open_stream_count() == 0


def test_fake_devices_without_generation_are_not_rejected(sd):
    from types import SimpleNamespace
    p = A.Player(SimpleNamespace(index=0, max_out=2, default_sr=48000.0))
    p.start()
    assert A.open_stream_count() == 1
    p.stop()


# ---------------------------------------------------------------- ids guardados: NUNCA se reasignan solos
def test_find_device_is_strict_a_changed_numbering_is_not_silently_reassigned(sd):
    sd.system = [dev("Dup", 1, 0), dev("Dup", 1, 0), dev("Solo", 1, 0)]
    A.refresh_devices()
    assert A.find_device("Dup#1", "in").index == 1 and A.device_is_ambiguous("Dup#1")
    sd.system = [dev("Zeta", 0, 2), dev("Dup", 1, 0), dev("Solo", 1, 0)]  # se reenumera: «Dup#1» ya no existe, hay UN solo Dup
    A.refresh_devices()
    ids = {d.id for d in A.list_devices()}
    assert "Dup" in ids and "Dup#1" not in ids
    with pytest.raises(LookupError):
        A.find_device("Dup#1", "in")  # antes se habría resuelto sola al único Dup: ahora no
    with pytest.raises(LookupError):
        A.find_device("Solo#2", "in")
    assert A.find_device("Solo", "in").index == 2


def test_candidates_are_informative_only_and_respect_direction(sd):
    sd.system = BASE + WH
    A.refresh_devices()
    assert A.device_candidates("WH-1000XM5#9", "in") == ["WH-1000XM5#in"]
    assert A.device_candidates("WH-1000XM5#9", "out") == ["WH-1000XM5#out"]
    assert A.device_candidates("WH-1000XM5#in", "in") == []  # el propio id no es candidato de sí mismo
    assert A.device_candidates("No existe", "in") == []
    with pytest.raises(LookupError):
        A.find_device("WH-1000XM5#9", "in")  # tener candidatos no cambia que la resolución sea estricta


def test_a_stale_generation_device_is_rejected_even_if_the_same_id_exists_again_after_refresh(sd):
    old = A.find_device("BlackHole 2ch", "out")
    A.refresh_devices()
    again = A.find_device("BlackHole 2ch", "out")
    assert again.id == old.id and again.gen == old.gen + 1
    with pytest.raises(A.StaleDevice):
        A.Player(old).start()
    A.Player(again).start()


# ---------------------------------------------------------------- ids estables por dirección y ambigüedad
def test_sony_style_input_output_pair_keeps_stable_ids_when_devices_are_reordered(sd):
    sd.system = BASE + WH
    A.refresh_devices()
    first = {d.id: d.index for d in A.list_devices()}
    sd.system = WH[::-1] + BASE  # reenumeración: cambian TODOS los índices
    A.refresh_devices()
    second = {d.id: d.index for d in A.list_devices()}
    assert first["WH-1000XM5#in"] != second["WH-1000XM5#in"]
    assert A.find_device("WH-1000XM5#in", "in").max_in > 0 and A.find_device("WH-1000XM5#out", "out").max_out > 0
    assert not any(d.ambiguous for d in A.list_devices())


def test_same_name_same_direction_devices_are_marked_ambiguous_and_exposed(sd):
    sd.system = BASE + [dev("USB Mic", 1, 0), dev("USB Mic", 1, 0)]
    A.refresh_devices()
    amb = {d.id for d in A.list_devices() if d.ambiguous}
    assert amb == {"USB Mic#3", "USB Mic#4"}
    p = A.devices_payload()
    assert {i["id"] for i in p["inputs"] if i["ambiguous"]} == amb and all(o["ambiguous"] is False for o in p["outputs"])
    assert A.device_is_ambiguous("USB Mic#3") and not A.device_is_ambiguous("BlackHole 2ch") and not A.device_is_ambiguous("No existe")


# ---------------------------------------------------------------- cierre incierto: nunca se pierde la prueba de stream abierto
def test_failed_close_keeps_the_stream_registered_and_blocks_refresh(sd, monkeypatch):
    p = A.Player(A.find_device("BlackHole 2ch", "out"))
    p.start()
    stream = FakeStream.instances[-1]

    def bad_close():
        raise OSError("close falló")

    monkeypatch.setattr(stream, "close", bad_close)
    with pytest.raises(OSError, match="close falló"):
        p.stop()
    assert A.open_stream_count() == 1 and A.unsafe_stream_count() == 1  # no se «perdió» el stream
    with pytest.raises(A.AudioBusy, match="cierre falló"):
        A.refresh_devices()
    assert sd.calls == [] and A.generation() == 0


def test_stream_that_fails_to_start_and_fails_to_close_stays_registered(sd, monkeypatch):
    FakeStream.fail_start = True
    monkeypatch.setattr(FakeStream, "close", lambda self: (_ for _ in ()).throw(OSError("close falló")))
    with pytest.raises(OSError):
        A.Player(A.find_device("BlackHole 2ch", "out")).start()
    assert A.open_stream_count() == 1 and A.unsafe_stream_count() == 1
    with pytest.raises(A.AudioBusy):
        A.refresh_devices()


def test_stream_is_registered_from_construction_and_a_constructor_failure_registers_nothing(sd, monkeypatch):
    seen = []
    orig_start = FakeStream.start

    def start(self):
        seen.append(A.open_stream_count())  # durante start() el stream YA cuenta como abierto
        orig_start(self)

    monkeypatch.setattr(FakeStream, "start", start)
    A.Player(A.find_device("BlackHole 2ch", "out")).start()
    assert seen == [1]

    def boom(**kw):
        raise OSError("no se pudo construir")

    sd.OutputStream = boom
    with pytest.raises(OSError):
        A.Player(A.find_device("BlackHole 2ch", "out")).start()
    assert A.open_stream_count() == 1  # solo el anterior


def test_successful_close_after_a_previous_failure_state_is_independent_per_stream(sd, monkeypatch):
    p1 = A.Player(A.find_device("BlackHole 2ch", "out"))
    p1.start()
    s1 = FakeStream.instances[-1]
    p2 = A.MicCapture(A.find_device("BlackHole 2ch", "in"), lambda b: None)
    p2.start()
    monkeypatch.setattr(s1, "close", lambda: (_ for _ in ()).throw(OSError("close falló")))
    with pytest.raises(OSError):
        p1.stop()
    p2.stop()  # el otro stream sí cierra bien
    assert A.open_stream_count() == 1 and A.unsafe_stream_count() == 1


def test_close_runs_without_holding_the_audio_lock_so_a_callback_thread_cannot_deadlock(sd, monkeypatch):
    p = A.Player(A.find_device("BlackHole 2ch", "out"))
    p.start()
    stream = FakeStream.instances[-1]
    result = {}

    def close():
        got = []
        t = threading.Thread(target=lambda: got.append(A._pa_lock.acquire(timeout=2) and (A._pa_lock.release() or True)))
        t.start()  # otro hilo (como el de callbacks) toma el lock mientras close() está en curso
        t.join(3)
        result["acquired"] = bool(got and got[0])
        stream.closed = True

    monkeypatch.setattr(stream, "close", close)
    p.stop()
    assert result["acquired"] is True and A.open_stream_count() == 0


def test_audio_callbacks_never_touch_the_audio_lock():
    import ast
    import inspect
    for fn in (A.Player._cb, A.MicCapture._cb):
        src = ast.parse(inspect.getsource(fn).replace("    def", "def", 1) if inspect.getsource(fn).startswith("    def") else inspect.getsource(fn))
        names = {n.id for n in ast.walk(src) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(src) if isinstance(n, ast.Attribute)}
        assert not names & {"_pa_lock", "refresh_devices", "list_devices", "open_stream_count"}, fn


# ---------------------------------------------------------------- sounddevice oculta errores de close() por defecto
def test_close_is_requested_with_ignore_errors_false_so_portaudio_errors_are_not_swallowed(sd):
    p = A.Player(A.find_device("BlackHole 2ch", "out"))
    p.start()
    p.stop()
    assert FakeStream.instances[-1].close_calls == [False]  # el valor por defecto (True) ocultaría el fallo y daría un falso «cierre confirmado»
    m = A.MicCapture(A.find_device("BlackHole 2ch", "in"), lambda b: None)
    m.start()
    m.stop()
    assert FakeStream.instances[-1].close_calls == [False]


def test_a_portaudio_close_error_that_only_shows_with_ignore_errors_false_blocks_refresh(sd):
    p = A.Player(A.find_device("BlackHole 2ch", "out"))
    p.start()
    FakeStream.close_error = True
    with pytest.raises(OSError, match="Pa_CloseStream"):
        p.stop()
    assert A.open_stream_count() == 1 and A.unsafe_stream_count() == 1
    with pytest.raises(A.AudioBusy, match="cierre falló"):
        A.refresh_devices()


def test_failed_start_with_a_portaudio_close_error_is_also_unsafe(sd):
    FakeStream.fail_start = True
    FakeStream.close_error = True
    with pytest.raises(OSError):
        A.Player(A.find_device("BlackHole 2ch", "out")).start()
    assert A.open_stream_count() == 1 and A.unsafe_stream_count() == 1
    with pytest.raises(A.AudioBusy):
        A.refresh_devices()


def test_streams_whose_close_takes_no_parameter_are_still_closed(sd, monkeypatch):
    seen = []
    p = A.Player(A.find_device("BlackHole 2ch", "out"))
    p.start()
    s = FakeStream.instances[-1]
    monkeypatch.setattr(s, "close", lambda: seen.append("close()"))  # doble antiguo sin ignore_errors
    p.stop()
    assert seen == ["close()"] and A.open_stream_count() == 0 and A.unsafe_stream_count() == 0


def test_close_kwargs_follows_the_real_sounddevice_signature():
    import sounddevice as real_sd
    assert A._close_kwargs(object.__new__(real_sd.OutputStream)) == {"ignore_errors": False}  # firma real, sin abrir ningún stream
    assert A._close_kwargs(object.__new__(real_sd.InputStream)) == {"ignore_errors": False}

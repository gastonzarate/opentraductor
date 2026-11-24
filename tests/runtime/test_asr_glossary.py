"""T-96: glosario del reconocedor (set_keyterms) y corrección post-ASR, con un Transcriber falso."""
import time

from runtime import asr_glossary as G


def test_default_file_loads_terms_without_commas():
    terms = G.load_terms()
    assert len(terms) >= 30
    assert "tests de aceptación" in terms and "despliegue continuo" in terms
    assert all("," not in t for t in terms)


def test_fix_replaces_close_ngrams_and_keeps_punctuation():
    terms = ("despliegue continuo", "tests de aceptación", "Kanban")
    assert G.glossary_fix("hacemos despliege continuo.", terms) == "hacemos despliegue continuo."
    assert G.glossary_fix("¿los test de aceptacion?", terms) == "¿los tests de aceptación?"
    assert G.glossary_fix("usamos kanban", terms) == "usamos kanban"  # igual sin acentos/mayúsculas: no toca


def test_fix_leaves_ordinary_spanish_alone():
    s = "el día de hoy vamos a hablar de sueldos, alquileres y de cuánto se paga en negro"
    assert G.glossary_fix(s) == s


def test_fix_is_cheap_on_long_partials():
    s = " ".join(["vamos a ver cómo hacemos las pruebas del equipo"] * 10)  # ~90 palabras
    t0 = time.perf_counter()
    for _ in range(20):
        G.glossary_fix(s)
    assert (time.perf_counter() - t0) / 20 < 0.01


class KT:
    def __init__(self):
        self.terms = None

    def set_keyterms(self, terms):
        self.terms = terms


def test_apply_keyterms_flag(monkeypatch):
    tr = KT()
    assert G.apply_keyterms(tr) == len(G.load_terms()) and tr.terms
    monkeypatch.setenv("OPENTRADUCTOR_ASR_KEYTERMS", "0")
    tr2 = KT()
    assert G.apply_keyterms(tr2) == 0 and tr2.terms is None


def test_apply_keyterms_without_support_or_error():
    assert G.apply_keyterms(object()) == 0

    class Bad:
        def set_keyterms(self, terms):
            raise RuntimeError("no streaming")

    assert G.apply_keyterms(Bad()) == 0


def test_custom_file(tmp_path, monkeypatch):
    f = tmp_path / "kt.txt"
    f.write_text("# comentario\nKubernetes\ncon, coma\n\nKubernetes\n", encoding="utf-8")
    monkeypatch.setenv("OPENTRADUCTOR_ASR_KEYTERMS_FILE", str(f))
    assert G.load_terms() == ("Kubernetes",)


def test_streamer_uses_keyterms_and_fix(monkeypatch):
    import sys
    import types
    import importlib.machinery
    import numpy as np
    from runtime import components as K

    class Stream:
        def __init__(self):
            self.fns = []

        def add_listener(self, fn): self.fns.append(fn)
        def start(self): pass

        def add_audio(self, data, sr):
            for fn in self.fns:
                fn(types.SimpleNamespace(line=types.SimpleNamespace(line_id=1, text="hacemos despliege continuo",
                                                                    last_transcription_latency_ms=1)))

        def stop(self): pass
        def close(self): pass

    class Tr(KT):
        def __init__(self, **kw):
            super().__init__()
            Tr.last = self

        def create_stream(self, update_interval=None): return Stream()
        def close(self): pass

    m = types.ModuleType("moonshine_voice")
    m.__spec__ = importlib.machinery.ModuleSpec("moonshine_voice", None)
    m.Transcriber = Tr
    m.ModelArch = types.SimpleNamespace(TINY_STREAMING=2)
    monkeypatch.setitem(sys.modules, "moonshine_voice", m)
    monkeypatch.setenv("OPENTRADUCTOR_ASR_ORT_PROVIDER", "cpu")
    pcm = np.zeros(1600, dtype="<i2").tobytes()

    a = K.MoonshineStreamer("/x", "TINY_STREAMING", 0.1)
    assert a.keyterms_n > 0 and Tr.last.terms
    assert a.add(pcm) == "hacemos despliegue continuo"

    monkeypatch.setenv("OPENTRADUCTOR_ASR_KEYTERMS", "0")
    monkeypatch.setenv("OPENTRADUCTOR_ASR_GLOSSARY_FIX", "0")
    b = K.MoonshineStreamer("/x", "TINY_STREAMING", 0.1)
    assert b.keyterms_n == 0 and Tr.last.terms is None
    assert b.add(pcm) == "hacemos despliege continuo"

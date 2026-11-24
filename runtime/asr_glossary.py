"""T-96: sesgo de vocabulario del reconocedor (Moonshine) con el glosario de la app y corrección post-ASR.

Medido en T-94 (artifacts/t94-asr/INFORME.md, charla 1, 47 términos, nivel ASR): moonshine-tiny 23/47 = 49 %;
con ``Transcriber.set_keyterms`` 29/47 = 62 %; con la corrección por glosario encima 31/47 = 66 %. Latencia de
stop() igual (124 vs 136 ms p50), charla 2 sin términos metidos (25/35 igual) y 0 líneas en silencio/ruido.

- ``OPENTRADUCTOR_ASR_KEYTERMS=0`` apaga el sesgo (por defecto encendido).
- ``OPENTRADUCTOR_ASR_GLOSSARY_FIX=0`` apaga la corrección post-ASR (por defecto encendida).
- ``OPENTRADUCTOR_ASR_KEYTERMS_FILE=<ruta>`` cambia el glosario (un término por línea, sin comas).

Solo lee un archivo del paquete (runtime/data/asr_keyterms.txt): sin red ni código descargado.
"""
from __future__ import annotations

import difflib
import os
import unicodedata
from functools import lru_cache
from pathlib import Path

DEFAULT_FILE = Path(__file__).resolve().parent / "data" / "asr_keyterms.txt"
FIX_MIN_RATIO = 0.88


def _flag(name: str) -> bool:
    return os.environ.get(name, "1") != "0"


def keyterms_enabled() -> bool:
    return _flag("OPENTRADUCTOR_ASR_KEYTERMS")


def fix_enabled() -> bool:
    return _flag("OPENTRADUCTOR_ASR_GLOSSARY_FIX")


@lru_cache(maxsize=4)
def _load(path: str) -> tuple[str, ...]:
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return ()
    out = []
    for ln in lines:
        t = ln.strip()
        if t and not t.startswith("#") and "," not in t and t not in out:
            out.append(t)
    return tuple(out)


def load_terms() -> tuple[str, ...]:
    return _load(os.environ.get("OPENTRADUCTOR_ASR_KEYTERMS_FILE") or str(DEFAULT_FILE))


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


@lru_cache(maxsize=4)
def _index(terms: tuple[str, ...]):
    """Por cantidad de palabras: [(término, normalizado)]."""
    idx: dict[int, list[tuple[str, str]]] = {}
    for t in terms:
        idx.setdefault(len(t.split()), []).append((t, _norm(t)))
    return idx


_PUNCT = ".,;:¿?¡!"


@lru_cache(maxsize=8192)
def _match(gn: str, n: int, terms: tuple[str, ...]) -> str | None:
    """Primer término de ``n`` palabras con similitud >= 0,88 con el n-grama normalizado ``gn`` (o None). Con
    memo: en el habla los n-gramas se repiten y cada parcial vuelve a pasar por las mismas palabras."""
    for t, tn in _index(terms).get(n, ()):
        if gn == tn:
            continue
        la, lb = len(gn), len(tn)
        if 2 * min(la, lb) / (la + lb) < FIX_MIN_RATIO:  # cota del ratio por largo
            continue
        sm = difflib.SequenceMatcher(None, gn, tn)
        if sm.quick_ratio() >= FIX_MIN_RATIO and sm.ratio() >= FIX_MIN_RATIO:
            return t
    return None


def glossary_fix(text: str, terms: tuple[str, ...] | None = None) -> str:
    """Reemplaza n-gramas (3, 2, 1 palabras) por el término del glosario con igual número de palabras si la
    similitud sin acentos es >= 0,88 (misma regla que scripts/eval/t94_score.py, la que se midió). Conserva la
    puntuación pegada al final del n-grama. Filtro por largo antes de difflib: el ratio no puede pasar
    2*min/(a+b), así que casi ningún par llega a SequenceMatcher y el costo por parcial es bajo."""
    if terms is None:
        terms = load_terms()
    if not text or not terms:
        return text
    idx = _index(terms)  # solo para saber qué largos de n-grama existen
    words = text.split()
    wn = [_norm(w) for w in words]  # _norm es por carácter: normalizar palabra a palabra = normalizar el n-grama
    out: list[str] = []
    i = 0
    while i < len(words):
        done = False
        for n in (3, 2, 1):
            cands = idx.get(n)
            if not cands or i + n > len(words):
                continue
            gn = " ".join(wn[i:i + n]).strip(_PUNCT)
            if len(gn) < 4:
                continue
            t = _match(gn, n, terms)
            if t is not None:
                raw = " ".join(words[i:i + n])
                head = raw[:len(raw) - len(raw.lstrip(_PUNCT))]
                tail = raw[len(raw.rstrip(_PUNCT)):]
                out.append(head + t + tail)
                i += n
                done = True
                break
        if not done:
            out.append(words[i])
            i += 1
    return " ".join(out)


def apply_keyterms(transcriber) -> int:
    """Pasa el glosario a ``set_keyterms`` si está encendido y el Transcriber lo admite. Devuelve cuántos términos
    quedaron activos (0 si apagado, sin soporte o error: el reconocedor sigue igual que sin sesgo)."""
    if not keyterms_enabled():
        return 0
    terms = load_terms()
    fn = getattr(transcriber, "set_keyterms", None)
    if not terms or fn is None:
        return 0
    try:
        fn(list(terms))
    except Exception:  # noqa: BLE001  (p. ej. modelo no streaming): seguir sin sesgo
        return 0
    return len(terms)

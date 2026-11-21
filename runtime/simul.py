"""T-67 paso 2: traducción simultánea por cláusulas (piezas puras, sin estado del pipeline).

Política *local agreement*: una palabra del parcial del ASR se confirma cuando dos hipótesis parciales seguidas
coinciden hasta ella. Sobre el prefijo confirmado se busca un límite de cláusula (coma, "pero", "porque",
"entonces", "o sea"...). La cláusula anterior al límite ya no puede cambiar y se manda a traducir mientras la
persona sigue hablando. El resto de la frase se traduce en el endpoint, como hoy.

Conectado en pipeline._asr_loop detrás de ``mt_simul_clauses`` (env OPENTRADUCTOR_MT_SIMUL).
"""
from __future__ import annotations

_STRIP = ".,;:!?¿¡…\"'()«»-—"

# Palabras que abren cláusula: se corta ANTES de ellas.
CLAUSE_CUES = frozenset({"pero", "porque", "entonces", "aunque", "sino", "mientras", "cuando", "pues"})
# Pares que abren cláusula: se corta antes del primero.
CLAUSE_CUE_PAIRS = frozenset({("o", "sea"), ("así", "que"), ("asi", "que"), ("es", "decir"), ("ya", "que")})


def _norm(w: str) -> str:
    return w.strip(_STRIP).lower()


def agreed_prefix(prev: str, cur: str) -> list[str]:
    """Palabras de ``cur`` cuyo prefijo coincide con ``prev`` (comparación sin puntuación ni mayúsculas).
    Se toma la versión de ``cur`` de cada palabra (la más nueva)."""
    a, b = prev.split(), cur.split()
    n = 0
    while n < len(a) and n < len(b) and _norm(a[n]) == _norm(b[n]):
        n += 1
    return b[:n]


def next_clause_end(words: list[str], start: int, min_words: int = 5,
                    extra_cues: frozenset[str] = frozenset()) -> int | None:
    """Fin (exclusivo) de la próxima cláusula cerrada dentro de ``words[start:]``, o None.

    Límites: palabra que termina en ``, ; :`` (corte después) o una palabra/par de ``CLAUSE_CUES`` /
    ``CLAUSE_CUE_PAIRS`` (corte antes). La cláusula debe tener al menos ``min_words`` palabras, para no mandar
    a traducir fragmentos que el traductor no puede resolver solos. El límite debe estar confirmado entero: para
    un par, las dos palabras tienen que estar en ``words``. ``extra_cues`` suma palabras que abren cláusula
    (p. ej. "y", "que"; config ``mt_simul_extra_cues``)."""
    for i in range(start, len(words)):
        w = _norm(words[i])
        if i - start >= min_words:
            if w in CLAUSE_CUES or w in extra_cues:
                return i
            if i + 1 < len(words) and (w, _norm(words[i + 1])) in CLAUSE_CUE_PAIRS:
                return i
        if i + 1 - start >= min_words and words[i][-1:] in ",;:" and i + 1 < len(words):
            return i + 1
    return None


# Primeras palabras de una pieza que, si siguen a una pieza sin punto final, van en minúscula en el subtítulo
# ("..., We'll do" -> "..., we'll do"). Solo palabras que nunca son nombre propio; "I" y nombres quedan igual.
_LOWER_OK = frozenset("""
a an the and but or so because if then when while although though since unless until as at by for from in
into of on to with without after before about we we're we'll we've we'd you you're you'll you've it it's
it'll they they're they'll he he's she she's this that that's these those there there's here what which who
how why where my our your their his her its me us them let let's just also still even only not no yes well
look listen okay ok please do does did is are was were will would can could should have has had
what's where's who's how's half about around like maybe otherwise meanwhile instead plus once every each some any
""".split())


def join_pieces(pieces: list[tuple[str, str]]) -> str:
    """Une las traducciones de las piezas de una frase (``[(es, en), ...]``, en orden) para el subtítulo.

    Marian traduce cada cláusula como si fuera una oración: cierra con punto y empieza con mayúscula. Si la
    cláusula española no terminaba en punto, ese punto se cambia por la puntuación española (``, ; :``) o se
    quita, y la pieza siguiente empieza en minúscula si su 1.ª palabra está en ``_LOWER_OK``. Solo texto: el
    audio de cada pieza ya se habló con su propio texto."""
    out: list[str] = []
    pieces = [(es, (en or "").strip()) for es, en in pieces if en and en.strip()]
    n = len(pieces)
    for i, (es, en) in enumerate(pieces):
        if out and out[-1][-1:] not in ".!?…":
            first, _, tail = en.partition(" ")
            if first[:1].isupper() and first.strip(_STRIP).lower() in _LOWER_OK:
                en = first[:1].lower() + first[1:] + (" " + tail if tail else "")
        if i < n - 1 and en.endswith(".") and not en.endswith(".."):
            last = (es or "").rstrip()[-1:]
            if last in ",;:":
                en = en[:-1] + last
            elif last not in ".!?…":
                en = en[:-1]
        out.append(en)
    return " ".join(out)


class ClauseCommitter:
    """Estado por segmento: último parcial, cuántas palabras ya se confirmaron como cláusulas."""

    def __init__(self, min_words: int = 5, extra_cues: frozenset[str] | set[str] | tuple = ()) -> None:
        self.min_words = min_words
        self.extra_cues = frozenset(_norm(w.strip()) for w in extra_cues if _norm(w.strip()))
        self.prev = ""
        self.committed = 0  # palabras ya entregadas como cláusulas
        self.words: list[str] = []  # esas palabras, tal como se mandaron a traducir

    def update(self, partial: str) -> list[str]:
        """Nuevo parcial del ASR -> cláusulas nuevas confirmadas (texto), en orden. Vacío si no hay.
        Si el parcial nuevo contradice lo ya confirmado, no confirma nada más (el endpoint lo resuelve)."""
        stable = agreed_prefix(self.prev, partial)
        self.prev = partial
        if [_norm(w) for w in stable[: self.committed]] != [_norm(w) for w in self.words]:
            return []
        out = []
        while True:
            end = next_clause_end(stable, self.committed, self.min_words, self.extra_cues)
            if end is None:
                return out
            out.append(" ".join(stable[self.committed:end]))
            self.words = stable[:end]
            self.committed = end

    def rest(self, final: str) -> str | None:
        """Texto del final del ASR que falta traducir (después de lo confirmado). Si el final ya no empieza con
        lo confirmado (el ASR reescribió), devuelve None para que el llamador traduzca la frase entera."""
        words = final.split()
        if self.committed == 0:
            return final
        if [_norm(w) for w in words[: self.committed]] != [_norm(w) for w in self.words]:
            return None
        return " ".join(words[self.committed:])

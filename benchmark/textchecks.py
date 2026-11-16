"""Chequeos de SUPERFICIE para números, negaciones y términos técnicos.

No miden exactitud semántica: son señales heurísticas que marcan casos
sospechosos para revisión humana, con ejemplos. Una frase sin marca puede seguir
siendo incorrecta, y una marcada puede estar bien (p. ej. «no» → «don't think»).
Números escritos con palabras («treinta», «mil») no se verifican.
"""
from __future__ import annotations

import re
from typing import Sequence

NEG_ES = {"no", "nunca", "jamás", "jamas", "ni", "tampoco", "sin", "nadie", "nada", "ningún", "ningun", "ninguna", "ninguno"}
NEG_EN = {"no", "not", "never", "neither", "nor", "without", "nobody", "nothing", "none", "cannot", "nowhere"}
NEG_EN_RE = re.compile(r"\b\w+n't\b", re.IGNORECASE)
# Términos técnicos en inglés que suelen conservarse tal cual en reuniones de software.
DEFAULT_TECH_TERMS = (
    "deploy", "api", "pr", "pull request", "commit", "merge", "branch", "bug", "backlog", "sprint", "pipeline",
    "endpoint", "framework", "release", "rollback", "staging", "frontend", "backend", "cloud", "docker",
    "kubernetes", "github", "ci/cd", "devops", "token", "script", "feature", "hotfix", "server", "dashboard",
)
_NUM = re.compile(r"\d+(?:[.,]\d+)*")


def number_tokens(text: str) -> list[str]:
    """Cifras sin separadores ('1,2' y '1.2' -> '12'; '1.200' -> '1200'). Se compara
    solo la secuencia de dígitos: no detecta un decimal mal colocado (1,2 vs 12)."""
    return [re.sub(r"[.,]", "", m) for m in _NUM.findall(text)]


def _words(text: str) -> list[str]:
    return re.findall(r"[\w'/]+", text.lower())


def check_numbers(src: str, hyp: str) -> dict | None:
    s = number_tokens(src)
    if not s:
        return None
    h = set(number_tokens(hyp))
    missing = [x for x in s if x not in h]
    return {"source_numbers": s, "missing_in_output": missing, "flag": bool(missing)}


def check_negation(src: str, hyp: str) -> dict | None:
    sw = _words(src)
    src_neg = [w for w in sw if w in NEG_ES]
    if not src_neg:
        return None
    hw = _words(hyp)
    hyp_neg = [w for w in hw if w in NEG_EN] + NEG_EN_RE.findall(hyp)
    return {"source_negations": src_neg, "output_negations": hyp_neg, "flag": not hyp_neg}


def check_terms(src: str, hyp: str, terms: Sequence[str] = DEFAULT_TECH_TERMS) -> dict | None:
    sl, hl = src.lower(), hyp.lower()
    present = [t for t in terms if re.search(rf"(?<!\w){re.escape(t)}(?!\w)", sl)]
    if not present:
        return None
    lost = [t for t in present if not re.search(rf"(?<!\w){re.escape(t)}(?!\w)", hl)]
    return {"source_terms": present, "missing_in_output": lost, "flag": bool(lost)}


def run_checks(rows: Sequence[dict], *, max_examples: int = 5) -> dict:
    """rows: {"item_id", "src", "hyp", "tags"}. Devuelve conteos y ejemplos por chequeo."""
    out: dict = {
        "label": "señales de superficie para revisión humana; NO miden exactitud semántica",
        "limitations": [
            "números en palabras no se verifican; solo se compara la secuencia de dígitos (un decimal mal colocado no se detecta)",
            "una negación puede expresarse sin token de negación o con otro (p. ej. 'don't think')",
            "un término técnico puede traducirse legítimamente",
        ],
        "checks": {},
    }
    for name, fn in (("numbers", check_numbers), ("negation", check_negation), ("technical_terms", check_terms)):
        applicable, flagged, examples = 0, 0, []
        for r in rows:
            res = fn(r["src"], r["hyp"] or "")
            if res is None:
                continue
            applicable += 1
            if res["flag"]:
                flagged += 1
                if len(examples) < max_examples:
                    examples.append({"item_id": r["item_id"], "src": r["src"], "hyp": r["hyp"], **res})
        out["checks"][name] = {"n_applicable": applicable, "n_flagged": flagged, "flagged_examples": examples}
    return out

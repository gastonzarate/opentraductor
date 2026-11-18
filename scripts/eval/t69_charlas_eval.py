#!/usr/bin/env python
"""T-69 (@bench): % de términos bien traducidos en charlas con contexto (scripts/eval/sets/charlas_ctx_ar.jsonl).

Cada charla se traduce turno a turno con Qwen, como en el pipeline: los últimos N turnos (es, en propio) van de contexto.
Brazos = valores de N (0 = sin contexto). La MT se arma con PipelineConfig() del commit medido (como la app); el entorno manda:
OPENTRADUCTOR_MT_GLOSSARY=0/1, OPENTRADUCTOR_MT_SENSE=0/1, OPENTRADUCTOR_MT_SESSION_KV=0/1.
Un término acierta si ``must`` aparece y ``bad`` no. Mide también la MT por turno (sin prefill KV: orientativo, no es end_to_first).

Uso:
  .venv-inc-llm/bin/python scripts/eval/t69_charlas_eval.py --out artifacts/t69-charlas/base [--ctx 0,6] [--set ...]
  .venv-inc-llm/bin/python scripts/eval/t69_charlas_eval.py --simul --out artifacts/t69-charlas/simul   # T-67 paso 2
  OPENTRADUCTOR_MT_GLOSSARY=1 .venv-inc-llm/bin/python scripts/eval/t69_charlas_eval.py --out artifacts/t69-charlas/gloss
"""
import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.environ.get("T69_CODE_ROOT", str(ROOT)))  # T69_CODE_ROOT: carpeta con el commit a medir (git archive)
DEFAULT_MODEL = ROOT / "artifacts/analysis/llm-mp4-a/models/qwen3-4b-instruct-2507-4bit"


def term_ok(en: str, term: dict) -> bool:
    return bool(re.search(term["must"], en, re.I)) and not (term["bad"] and re.search(term["bad"], en, re.I))


def norm_charla(ch: dict) -> dict:
    """Acepta el banco v1 (turnos/terms must,bad) y el set del 17/11 (sentences/terms ok,bad, 7 charlas)."""
    if "turnos" in ch:
        return ch
    return {"id": ch["talk_id"], "turnos": [
        {"es": x["es"], "terms": [{"t": t["es"], "must": t["ok"], "bad": t.get("bad") or ""} for t in x["terms"]]}
        for x in ch["sentences"]]}


def remember(q, h: list, es: str, en: str, n: int) -> None:
    """Igual que Engine._mt_remember (pipeline.py): tope por tokens (OPENTRADUCTOR_MT_CONTEXT_TOKENS) con recorte en
    bloque hasta cap/2; con session_kv recorte en bloque por frases; si no, las últimas n. n = mt_context_turns."""
    h.append((es.strip(), en.strip()))
    v = os.environ.get("OPENTRADUCTOR_MT_CONTEXT_TOKENS")
    cap = int(v) if v and v.isdigit() else 0
    if cap > 0 and hasattr(q, "turn_tokens"):
        sizes = [q.turn_tokens(a, b) for a, b in h]
        if sum(sizes) > cap:
            tot, k = sum(sizes), 0
            while k < len(h) - 1 and tot > cap // 2:
                tot -= sizes[k]
                k += 1
            del h[:k]
        if len(h) > n:
            del h[: len(h) - max(1, n // 2)]
    elif getattr(q, "session_kv", False):
        if len(h) > n:
            del h[: len(h) - max(1, n // 2)]
    else:
        del h[:-n]


def simul_split(es: str, min_words: int | None = None, cues: str | None = None) -> list[str]:
    """Piezas en que el Engine con MT_SIMUL=1 traduce ``es``: cláusulas confirmadas por local agreement sobre
    parciales que crecen de a una palabra (el ASR real puede cortar menos), y el resto al fin de frase.
    Reglas como Engine._new_clauser: OPENTRADUCTOR_MT_SIMUL_MIN_WORDS / _CUES o los defaults de PipelineConfig."""
    from runtime.config import PipelineConfig  # noqa: E402
    from runtime.simul import ClauseCommitter  # noqa: E402
    cfg = PipelineConfig()
    if min_words is None:
        mw = os.environ.get("OPENTRADUCTOR_MT_SIMUL_MIN_WORDS", "")
        min_words = int(mw) if mw.isdigit() and int(mw) > 0 else cfg.mt_simul_min_words
    if cues is None:
        cues = os.environ.get("OPENTRADUCTOR_MT_SIMUL_CUES", cfg.mt_simul_extra_cues)
    c = ClauseCommitter(min_words=min_words, extra_cues=cues.split(","))
    w = es.split()
    pieces: list[str] = []
    for k in range(1, len(w) + 1):
        pieces += c.update(" ".join(w[:k]))
    rest = c.rest(es)
    if rest is None:
        rest = " ".join(w[c.committed:])
    return pieces + ([rest] if rest.strip() else [])


def simul_translate(q, es: str, hist: list, n: int) -> tuple[str, int]:
    """Como Engine._simul_speak: cada pieza se traduce con el contexto de ese momento y se recuerda como turno."""
    parts = []
    for piece in simul_split(es):
        en = q.translate(piece, tuple(hist) if n > 0 else ())
        if n > 0:
            remember(q, hist, piece, en, n)
        parts.append((piece, en))
    from runtime.simul import join_pieces  # noqa: E402
    return join_pieces(parts), len(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("OPENTRADUCTOR_QWEN_DIR", str(DEFAULT_MODEL)))
    ap.add_argument("--set", default=str(ROOT / "scripts/eval/sets/charlas_ctx_ar.jsonl"))
    ap.add_argument("--ctx", default="0,6", help="turnos de contexto por brazo, separados por coma (0 = sin contexto)")
    ap.add_argument("--first-token", action="store_true",
                    help="usa translate_stream (mismo texto que translate) y mide ms hasta el 1.er token de la MT")
    ap.add_argument("--prefill", action="store_true",
                    help="q.prefill() antes de cada turno (camino del pipeline; con SESSION_KV=1 es lo que lo hace valer)")
    ap.add_argument("--simul", action="store_true",
                    help="T-67 paso 2: parte cada turno como el Engine con MT_SIMUL=1 (parciales crecientes -> "
                         "ClauseCommitter), traduce cada cláusula y el resto por separado y une el inglés")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from runtime import components as K  # noqa: E402
    from runtime.config import PipelineConfig  # noqa: E402

    charlas = [norm_charla(json.loads(line)) for line in open(a.set) if line.strip()]
    # 18/11 (T-74): la MT se arma con la PipelineConfig del commit medido, igual que Engine (pipeline.py), y el env
    # (OPENTRADUCTOR_MT_*) sigue mandando dentro de QwenMLX. Antes se usaban los defaults del constructor (todo apagado),
    # y un brazo "sin flags" medía Qwen sin glosario aunque la config lo encendiera (D1 de 95bea57: 66/96 en vez de 91/96).
    cfg = PipelineConfig()
    q = K.QwenMLX(a.model, normalize=cfg.mt_normalize, sense_prompt=cfg.mt_sense_prompt,
                  sense_glossary=cfg.mt_sense_glossary, session_kv=cfg.mt_session_kv, draft_dir=cfg.mt_draft_dir)
    q.translate("hola")  # calentamiento
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"flags": {k: v for k, v in os.environ.items() if k.startswith("OPENTRADUCTOR_MT_")}, "prefill": a.prefill,
               "efectivo": {"sense": q.system != K.QWEN_SYSTEM, "glossary": q.glossary, "session_kv": q.session_kv,
                            "ext": os.environ.get("OPENTRADUCTOR_GLOSSARY_EXT", "1") != "0",
                            "dom": os.environ.get("OPENTRADUCTOR_GLOSSARY_DOM", "1") != "0"},
               "set": a.set, "arms": {}}
    with open(out / "rows.jsonl", "w") as f:
        for n in [int(x) for x in a.ctx.split(",")]:
            arm = f"ctx{n}"
            hits = total = 0
            ms: list[float] = []
            ft: list[float] = []
            n_pieces: list[int] = []
            by_charla: dict[str, list[int]] = {}
            for ch in charlas:
                hist: list[tuple[str, str]] = []
                h = t_ = 0
                q._sess = None  # charla nueva = sesión nueva
                for i, turn in enumerate(ch["turnos"]):
                    context = tuple(hist) if n > 0 else ()
                    if a.prefill:  # como el pipeline: el parcial (sin su última palabra) se carga antes del fin de frase
                        q.prefill(turn["es"], context)
                    t0 = time.perf_counter()
                    if a.first_token:
                        first: list[float] = []
                        en = q.translate_stream(turn["es"], context,
                                                lambda _p: first.append(time.perf_counter()) if not first else None)
                        if first:
                            ft.append((first[0] - t0) * 1000)
                    elif a.simul:
                        en, pieces = simul_translate(q, turn["es"], hist, n)
                    else:
                        en = q.translate(turn["es"], context)
                    ms.append((time.perf_counter() - t0) * 1000)
                    if a.simul:
                        n_pieces.append(pieces)
                    elif n > 0:
                        remember(q, hist, turn["es"], en, n)
                    res = [term_ok(en, t) for t in turn["terms"]]
                    h += sum(res)
                    t_ += len(res)
                    f.write(json.dumps({"arm": arm, "charla": ch["id"], "turno": i, "es": turn["es"], "en": en,
                                        "terms": [{"t": t["t"], "ok": r} for t, r in zip(turn["terms"], res)],
                                        "ms": round(ms[-1], 1)}, ensure_ascii=False) + "\n")
                by_charla[ch["id"]] = [h, t_]
                hits += h
                total += t_
            summary["arms"][arm] = {"terminos_ok": hits, "terminos": total, "pct": round(100 * hits / total, 1),
                                    "por_charla": {k: f"{v[0]}/{v[1]}" for k, v in by_charla.items()},
                                    "mt_p50_ms": round(statistics.median(ms), 1),
                                    "mt_p95_ms": round(sorted(ms)[int(0.95 * (len(ms) - 1))], 1), "turnos": len(ms),
                                    "loadavg_fin": os.getloadavg()[0]}
            if n_pieces:
                summary["arms"][arm]["simul_turnos_cortados"] = sum(1 for x in n_pieces if x > 1)
                summary["arms"][arm]["simul_piezas"] = sum(n_pieces)
            if ft:
                summary["arms"][arm]["mt_1er_token_p50_ms"] = round(statistics.median(ft), 1)
                summary["arms"][arm]["mt_1er_token_p95_ms"] = round(sorted(ft)[int(0.95 * (len(ft) - 1))], 1)
            print(arm, json.dumps(summary["arms"][arm], ensure_ascii=False), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

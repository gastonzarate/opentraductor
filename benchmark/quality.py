"""Calidad de traducción y texto realmente audible.

Reglas:
* Nunca se inventa una puntuación. Si falta librería, modelo o referencia, el
  resultado es ``{"status": "unavailable", "reason": ...}``.
* Se puntúa por separado (a) el texto PLANEADO por el candidato
  (``planned_text_en``) y (b) el texto AUDIBLE: lo que un ASR independiente
  reconoce en el PCM realmente emitido.
* BLEU/chrF/WER son proxies automáticos. NO son MOS humano ni sustituyen la
  escucha ciega (ver ``listening.py``). La similitud de F0 palabra a palabra
  entre idiomas no se calcula: no es comparable.
* Los ítems sin referencia se cuentan y se informan; no se excluyen en silencio.
"""
from __future__ import annotations

import re
import wave
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from . import audio as A
from .metrics import SegmentResult
from .replay import StreamTrace

# ASR independiente: (pcm_s16le, sample_rate) -> texto. Debe ser un sistema
# DISTINTO del candidato evaluado. No se provee uno por defecto.
AudibleAsr = Callable[[bytes, int], str]


def unavailable(reason: str) -> dict:
    return {"status": "unavailable", "reason": reason}


def translation_scores(
    hypotheses: Sequence[str | None],
    references: Sequence[Sequence[str] | None],
) -> dict[str, Any]:
    """BLEU y chrF corpus-level (sacrebleu) sobre ítems con hipótesis y referencia.

    Un ítem sin hipótesis (silencio/fallo) cuenta como hipótesis vacía: penaliza.
    Un ítem sin referencia se excluye del cómputo y se informa su cantidad.
    """
    n_total = len(hypotheses)
    idx = [i for i, r in enumerate(references) if r]
    n_missing_ref = n_total - len(idx)
    if not idx:
        return unavailable("sin referencias reference_en")
    try:
        import sacrebleu  # type: ignore
    except ImportError:
        return unavailable("sacrebleu no instalado")
    hyps = [(hypotheses[i] or "") for i in idx]
    n_empty = sum(1 for h in hyps if not h)
    nref = min(len(references[i]) for i in idx)  # type: ignore[arg-type]
    ref_streams = [[references[i][k] for i in idx] for k in range(nref)]  # type: ignore[index]
    bleu_m, chrf_m = sacrebleu.metrics.BLEU(), sacrebleu.metrics.CHRF()
    bleu = bleu_m.corpus_score(hyps, ref_streams)
    chrf = chrf_m.corpus_score(hyps, ref_streams)
    return {
        "status": "ok",
        "n_total": n_total,
        "n_scored": len(idx),
        "n_missing_reference": n_missing_ref,
        "n_empty_hypothesis": n_empty,
        "n_references_per_item_used": nref,
        "bleu": bleu.score,
        "bleu_signature": str(bleu_m.get_signature()),
        "chrf": chrf.score,
        "chrf_signature": str(chrf_m.get_signature()),
        "caveat": "Proxy automático; no es MOS humano ni evaluación independiente neural.",
    }


def neural_metric_status() -> dict:
    """COMET/MetricX no se ejecutan sin modelo local descargado y verificado."""
    return unavailable(
        "métrica neural (COMET/MetricX) no configurada: requiere modelo local "
        "descargado y licencia revisada; sin ella no se reporta puntuación"
    )


def wer(reference: str, hypothesis: str) -> dict:
    try:
        import jiwer  # type: ignore
    except ImportError:
        return unavailable("jiwer no instalado")
    if not reference.strip():
        return unavailable("referencia vacía")
    return {"status": "ok", "wer": jiwer.wer(reference.lower(), hypothesis.lower())}


def _planned_text(trace: StreamTrace) -> str | None:
    from .metrics import commit_texts

    texts, _ = commit_texts([e for e in trace.events if e.kind == "tts_chunk"], per_event=False)
    return " ".join(texts) if texts else None


_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_text(t: str) -> str:
    """minúsculas, sin puntuación, espacios colapsados (acentos se conservan)."""
    return " ".join(_PUNCT.sub(" ", t.lower()).split())


def asr_error_rates(
    rows: Sequence[tuple[str | None, str | None, str | None]],
) -> dict[str, Any]:
    """WER/CER del ASR (español) contra la transcripción de referencia, por
    procedencia de la transcripción; separado de la calidad de traducción.

    rows: (hipótesis ASR o None, transcripción, procedencia). Hipótesis ausente
    = vacía (penaliza). Nivel corpus (suma de errores / palabras de referencia).
    """
    try:
        import jiwer  # type: ignore
    except ImportError:
        return unavailable("jiwer no instalado")
    out: dict[str, Any] = {}
    for prov in ("human", "generated", "unknown"):
        sub = [(h or "", r) for h, r, p in rows if (p or "unknown") == prov and r and normalize_text(r)]
        if not sub:
            continue
        refs = [normalize_text(r) for _, r in sub]
        hyps = [normalize_text(h) for h, _ in sub]
        out[prov] = {
            "status": "ok",
            "n": len(sub),
            "n_empty_hypothesis": sum(1 for h in hyps if not h),
            "wer": jiwer.wer(refs, hyps),
            "cer": jiwer.cer(refs, hyps),
            "normalization": "minúsculas, sin puntuación, espacios colapsados",
        }
    out["items_without_transcript"] = sum(1 for _, r, _p in rows if not r)
    if not any(k in out for k in ("human", "generated", "unknown")):
        out["status"] = "unavailable"
        out["reason"] = "ningún ítem con transcripción de referencia"
    return out


def audible_text_report(
    traces: Sequence[StreamTrace],
    asr: AudibleAsr | None,
    *,
    min_word_ratio: float = 0.8,
) -> dict:
    """Texto planeado vs. audible por stream (un fragmento por stream).

    Requiere ``keep_pcm=True`` y un ASR independiente del candidato. El ASR
    tiene sesgo propio (errores del ASR se confunden con errores de TTS, y un
    ASR puede "corregir" habla poco clara): el resultado es evidencia de
    inteligibilidad, no verdad de referencia.
    """
    if asr is None:
        return unavailable("sin ASR independiente configurado para texto audible")
    items: dict[str, dict] = {}
    for t in traces:
        if t.continuous:
            return unavailable(
                "texto audible por fragmento en stream continuo requiere source_span_s (no implementado en v0)"
            )
        tts = [e for e in t.events if e.kind == "tts_chunk" and e.pcm]
        pcm = b"".join(e.pcm for e in tts)
        sr = next((e.sample_rate for e in tts if e.sample_rate), A.TARGET_SR)
        planned = _planned_text(t)
        row: dict[str, Any] = {"item_id": t.segments[0].item_id, "planned_text": planned,
                               "audible_text": None, "attributed_pcm_s": 0.0,
                               "omission_suspected": None, "planned_vs_audible_wer": unavailable("sin PCM retenido")}
        if pcm:
            audible = asr(pcm, sr)
            row.update(audible_text=audible, attributed_pcm_s=len(pcm) / 2 / sr)
            if planned:
                row["omission_suspected"] = len(audible.split()) < min_word_ratio * len(planned.split())
                row["planned_vs_audible_wer"] = wer(planned, audible)
            else:
                row["planned_vs_audible_wer"] = unavailable("sin texto planeado")
        items[t.stream_id] = row
    return {
        "status": "ok",
        "items": items,
        "n": len(items),
        "n_omission_suspected": sum(1 for i in items.values() if i["omission_suspected"]),
        "caveat": "Depende del ASR independiente usado; error del ASR se confunde con error de TTS.",
    }


def _by_provenance(
    rows: Sequence[tuple[str | None, Sequence[str] | None, str | None]],
) -> dict:
    """rows: (hipótesis, referencias, procedencia de referencia). Puntúa por
    procedencia; humanas y generadas nunca se mezclan."""
    out: dict[str, Any] = {}
    for prov in ("human", "generated", "unknown"):
        sub = [(h, r) for h, r, p in rows if (p or "unknown") == prov and r]
        if sub:
            out[prov] = translation_scores([h for h, _ in sub], [r for _, r in sub])
    n_no_ref = sum(1 for _, r, _p in rows if not r)
    out["items_without_reference"] = n_no_ref
    if not any(k in out for k in ("human", "generated", "unknown")):
        out["status"] = "unavailable"
        out["reason"] = "ningún ítem con referencia reference_en"
    return out


def evaluate_run(
    results: Sequence[SegmentResult],
    traces: Sequence[StreamTrace],
    refs: dict[str, tuple[Sequence[str] | None, str | None]],
    asr: AudibleAsr | None,
    transcripts: dict[str, tuple[str | None, str | None]] | None = None,
) -> dict:
    """Calidad de la corrida. ``refs``: item_id -> (referencias, procedencia).

    Se puntúa el texto planeado y, si hay ASR independiente, el texto audible.
    Todas las repeticiones atribuibles cuentan como filas; fallos/silencios =
    hipótesis vacía. ``no_correspondence`` se excluye porque no equivale a una
    omisión del candidato.
    """
    event_presence = {
        "tts_chunk": sum(1 for t in traces for e in t.events if e.kind == "tts_chunk"),
        "tts_chunk_with_planned_text": sum(
            1 for t in traces for e in t.events
            if e.kind == "tts_chunk" and bool(e.planned_text_en)
        ),
        "mt_partial": sum(1 for t in traces for e in t.events if e.kind == "mt_partial"),
        "mt_final": sum(1 for t in traces for e in t.events if e.kind == "mt_final"),
        "asr_final": sum(1 for t in traces for e in t.events if e.kind == "asr_final"),
    }
    attributable_results = [r for r in results if r.outcome != "no_correspondence"]
    planned_rows = [
        (r.planned_text_en, *refs.get(r.item_id, (None, None))) for r in attributable_results
    ]
    if any(r.planned_text_en for r in attributable_results):
        planned = _by_provenance(planned_rows)
    elif event_presence["tts_chunk_with_planned_text"]:
        planned = unavailable(
            f"el candidato SÍ emitió {event_presence['tts_chunk']} tts_chunk "
            f"({event_presence['tts_chunk_with_planned_text']} con planned_text_en), pero el texto no se "
            "atribuye a ítems sin correspondencia fuente→salida; no significa que falte TTS ni texto planeado"
        )
    elif event_presence["tts_chunk"]:
        planned = unavailable(
            f"el candidato SÍ emitió {event_presence['tts_chunk']} tts_chunk, pero ninguno incluyó "
            "planned_text_en; falta texto planeado observable y no es un problema de correspondencia fuente→salida"
        )
    else:
        planned = unavailable("el candidato no emitió tts_chunk: no hay texto planeado para TTS")
    out: dict[str, Any] = {
        "event_presence": event_presence,
        "planned_text": planned,
        "neural_metric": neural_metric_status(),
        "rows_scored": len(planned_rows),
        "rows_no_correspondence_excluded": len(results) - len(attributable_results),
        "caveats": [
            "BLEU/chrF son proxies automáticos; no sustituyen evaluación humana.",
            "Ningún puntaje automático se presenta como MOS.",
            "Referencias humanas y generadas se puntúan por separado.",
        ],
    }
    # Texto de la MT (mt_final) por separado del texto planeado para TTS.
    mt_rows = [(r.mt_text_en, *refs.get(r.item_id, (None, None))) for r in attributable_results]
    if any(r.mt_text_en for r in attributable_results):
        out["mt_text"] = _by_provenance(mt_rows)
    else:
        n_mt = event_presence["mt_final"] + event_presence["mt_partial"]
        if n_mt:
            out["mt_text"] = unavailable(
                f"el candidato SÍ emitió {event_presence['mt_final']} mt_final y "
                f"{event_presence['mt_partial']} mt_partial, pero no se atribuyen a ítems "
                "(sin correspondencia fuente→salida por ítem): sin calidad de MT por ítem. "
                "No significa que falte la MT"
            )
        else:
            out["mt_text"] = unavailable("el candidato no emitió mt_final ni mt_partial")
    # Chequeos de superficie (números/negaciones/términos técnicos) en la 1.ª repetición.
    if transcripts is not None:
        from . import textchecks

        first_rep = {}
        for r in results:
            if r.stream_id.startswith("r0-") and r.item_id not in first_rep:
                first_rep[r.item_id] = r
        no_correspondence = [r for r in first_rep.values() if r.outcome == "no_correspondence"]
        rows = [{"item_id": i, "src": (transcripts.get(i) or (None,))[0] or "",
                 "hyp": r.mt_text_en or r.planned_text_en or "", "tags": r.tags}
                for i, r in first_rep.items()
                if r.outcome != "no_correspondence" and (transcripts.get(i) or (None,))[0]]
        if no_correspondence:
            out["caveats"].append(
                f"Chequeos de superficie excluyeron {len(no_correspondence)} ítems sin correspondencia "
                "fuente→salida; no se tratan como hipótesis vacías ni omisiones."
            )
        if rows:
            blk = textchecks.run_checks(rows)
            tags = sorted({t for r in rows for t in r["tags"]})
            blk["by_phenomenon"] = {
                t: {k: {"n_applicable": v["n_applicable"], "n_flagged": v["n_flagged"]}
                    for k, v in textchecks.run_checks([r for r in rows if t in r["tags"]])["checks"].items()}
                for t in tags
            }
            out["text_checks"] = blk
    # ASR del candidato (es) vs transcripción humana: separado de la traducción.
    if transcripts is not None:
        emitted = [r for r in results if r.asr_status == "ok"]
        if not emitted:
            if event_presence["asr_final"]:
                out["candidate_asr_es"] = unavailable(
                    f"el candidato SÍ emitió {event_presence['asr_final']} asr_final, pero no se atribuyen "
                    "a ítems sin correspondencia fuente→salida; no significa que falte ASR"
                )
            else:
                out["candidate_asr_es"] = unavailable(
                    "el candidato no emitió asr_final (¿modelo end-to-end?)"
                )
        else:
            arows = [(r.asr_text_es, *transcripts.get(r.item_id, (None, None)))
                     for r in results if r.asr_status != "no_correspondence"]
            blk = asr_error_rates(arows)
            blk["n_no_correspondence_excluded"] = sum(1 for r in results if r.asr_status == "no_correspondence")
            out["candidate_asr_es"] = blk
    aud = audible_text_report(traces, asr)
    out["audible_text_report"] = aud
    if aud.get("status") == "ok":
        rows = []
        for r in results:
            it = aud["items"].get(r.stream_id)
            ref, prov = refs.get(r.item_id, (None, None))
            rows.append((it["audible_text"] if it else None, ref, prov))
        out["audible_text"] = _by_provenance(rows)
    else:
        out["audible_text"] = aud
    return out

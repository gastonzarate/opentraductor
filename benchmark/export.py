"""Export de audio de salida de candidatos para escucha ciega y proxies.

SOLO audio sintético producido por candidatos sobre audio público del
benchmark. Nunca audio de conversaciones. Cada WAV se verifica al escribirlo
(relectura + sha256) y el índice registra procedencia y resultado.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from . import audio as A
from .replay import StreamTrace

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(s: str) -> str:
    return _SAFE.sub("_", s)[:80]


def export_stream_audio(
    trace: StreamTrace,
    out_dir: str | Path,
    *,
    system: str,
    outcome_by_item: dict[str, str],
    run_label: str,
) -> dict | None:
    """Escribe la salida TTS de un stream como WAV nativo. Devuelve el registro
    de índice, o ``None`` si no hay PCM no silencioso que exportar."""
    tts = [e for e in trace.events if e.kind == "tts_chunk" and e.pcm]
    if not tts:
        return None
    srs = {e.sample_rate or A.TARGET_SR for e in tts}
    if len(srs) != 1:
        return {"stream_id": trace.stream_id, "error": f"frecuencias mezcladas {sorted(srs)}", "exported": False}
    sr = srs.pop()
    pcm = b"".join(e.pcm for e in sorted(tts, key=lambda e: e.recv_ns))
    if all(e.first_nonsilent_offset_s is None for e in tts):
        return None  # salida totalmente silenciosa: nada que escuchar
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    item_ids = [s.item_id for s in trace.segments]
    from .metrics import commit_texts

    ptexts, amb = commit_texts([e for e in trace.events if e.kind == "tts_chunk"], per_event=False)
    mtexts, _ = commit_texts([e for e in trace.events if e.kind == "mt_final"], per_event=True)
    fname = f"{safe_name(system)}__{safe_name(trace.stream_id)}.wav"
    path = out / fname
    A.write_wav_pcm16(path, pcm, sr)
    # verificación anti-corrupción: releer y comparar
    back, sr_back = A.read_wav_pcm16_mono(path)
    ok = back == pcm and sr_back == sr
    rec = {
        "system": system,
        "run_label": run_label,
        "stream_id": trace.stream_id,
        "item_ids": item_ids,
        "outcomes": {i: outcome_by_item.get(i) for i in item_ids},
        "continuous": trace.continuous,
        "planned_text_en": " ".join(ptexts) if ptexts else None,
        "planned_text_ambiguous": amb,
        "mt_text_en": " ".join(mtexts) if mtexts else None,
        "wav": fname,
        "sample_rate": sr,
        "duration_s": len(pcm) / 2 / sr,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "roundtrip_ok": ok,
        "exported": ok,
        "scope": "benchmark_only_synthetic_output",
    }
    return rec


def write_index(records: list[dict], out_dir: str | Path) -> Path:
    p = Path(out_dir) / "export_index.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p


def read_index(path: str | Path) -> list[dict]:
    base = Path(path).parent
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        r["_dir"] = str(base)
    return rows


def verify_index(rows: list[dict]) -> list[str]:
    """Devuelve problemas (archivo ausente o hash distinto)."""
    problems = []
    for r in rows:
        if not r.get("exported"):
            problems.append(f"{r.get('stream_id')}: no exportado")
            continue
        p = Path(r["_dir"]) / r["wav"]
        if not p.exists():
            problems.append(f"{r['wav']}: ausente")
        elif hashlib.sha256(p.read_bytes()).hexdigest() != r["sha256"]:
            problems.append(f"{r['wav']}: sha256 distinto (corrupto)")
    return problems

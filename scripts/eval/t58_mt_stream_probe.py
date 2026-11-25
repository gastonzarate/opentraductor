"""T-58 sonda de mt_stream_first_chunk: ¿intercalar Kokoro entre pasos del decode de Qwen cambia el texto o
atrasa el resto?

Qwen + Kokoro cargados en un solo proceso (mismo hilo, como el hilo GPU del pipeline). Por cada frase ES:
  normal  translate() -> synth_clauses(plan) : 1.er audio = MT + TTS(plan[0])
  stream  translate_stream(on_text) con Kokoro del 1.er fragmento apenas first_clause_final lo fija, después
          synth_clauses(plan[1:], first=False)
Mide desde el inicio de la MT: first_ms (1.er chunk listo), second_ms (2.º chunk listo), mt_ms (fin del decode,
en stream incluye el Kokoro intercalado) y mt_pure_ms (decode sin Kokoro). Verifica texto EN, cortes y PCM
idénticos entre los brazos. El orden normal/stream se alterna por repetición.

Frases: text_es de data/manifests/t58_asr_final_texts.jsonl + text_meetings_ar_dev.jsonl y es de
scripts/eval/sets/mt_sentido_ar.jsonl (LIMIT=n recorta).

Uso (sin otra medición corriendo):
  OPENTRADUCTOR_MLX_CACHE_MB=256 REPS=3 .venv/bin/python scripts/eval/t58_mt_stream_probe.py OUT.json
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime import components as K  # noqa: E402
from runtime.config import PipelineConfig  # noqa: E402
from runtime.pipeline import _apply_mlx_cache_limit  # noqa: E402

REPS = int(os.environ.get("REPS", "3"))
LIMIT = int(os.environ.get("LIMIT", "0"))
QDIR = os.environ.get("OPENTRADUCTOR_QWEN_DIR", str(ROOT / "artifacts/analysis/llm-mp4-a/models/qwen3-4b-instruct-2507-4bit"))


def texts() -> list[str]:
    out = []
    for p, k in (("data/manifests/t58_asr_final_texts.jsonl", "text_es"),
                 ("data/manifests/text_meetings_ar_dev.jsonl", "text_es"),
                 ("scripts/eval/sets/mt_sentido_ar.jsonl", "es")):
        for line in (ROOT / p).read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line)[k])
    return out[:LIMIT] if LIMIT else out


def run_normal(mt, tts, t):
    t0 = time.perf_counter()
    en = mt.translate(t)
    mt_ms = (time.perf_counter() - t0) * 1000
    plan = tts.plan(en) if en else []
    chunks, marks = [], []
    for ch in tts.synth_clauses(plan, True):
        chunks.append(ch)
        marks.append((time.perf_counter() - t0) * 1000)
    return {"en": en, "plan": plan, "pcm": [c.pcm for c in chunks], "mt_ms": mt_ms, "mt_pure_ms": mt_ms,
            "first_ms": marks[0] if marks else None, "second_ms": marks[1] if len(marks) > 1 else None,
            "head_early": False}


def run_stream(mt, tts, t):
    t0 = time.perf_counter()
    st = {"head": None, "tts_ms": 0.0}
    chunks, marks = [], []

    def on_text(prefix):
        if st["head"] is not None:
            return
        h = tts.first_clause_final(prefix)
        if h is None:
            return
        st["head"] = h
        a = time.perf_counter()
        for ch in tts.synth_clauses([h], True):
            chunks.append(ch)
            marks.append((time.perf_counter() - t0) * 1000)
        st["tts_ms"] += (time.perf_counter() - a) * 1000

    en = mt.translate_stream(t, on_text=on_text)
    mt_ms = (time.perf_counter() - t0) * 1000
    plan = tts.plan(en) if en else []
    rest = plan if st["head"] is None else plan[1:]
    for ch in tts.synth_clauses(rest, st["head"] is None):
        chunks.append(ch)
        marks.append((time.perf_counter() - t0) * 1000)
    return {"en": en, "plan": plan, "pcm": [c.pcm for c in chunks], "mt_ms": mt_ms,
            "mt_pure_ms": mt_ms - st["tts_ms"], "first_ms": marks[0] if marks else None,
            "second_ms": marks[1] if len(marks) > 1 else None, "head_early": st["head"] is not None,
            "head_ok": st["head"] is None or plan[:1] == [st["head"]]}


def p50(v):
    v = [x for x in v if x is not None]
    return round(statistics.median(v), 1) if v else None


def main(out: str) -> None:
    cfg = PipelineConfig()
    cap = _apply_mlx_cache_limit(cfg)
    tts = K.KokoroTTS(cfg.voice, cfg.speed, cfg.tts_trim_lead_margin_ms, cfg.tts_first_chunk_max_words,
                      cfg.tts_bind_conjunctions, cfg.tts_trim_tail)
    mt = K.QwenMLX(Path(QDIR), normalize=cfg.mt_normalize)
    ts = texts()
    for t in ts[:3]:  # warmup
        run_normal(mt, tts, t)
        run_stream(mt, tts, t)
    rows = []
    for r in range(REPS):
        for i, t in enumerate(ts):
            arms = ("normal", "stream") if (r + i) % 2 == 0 else ("stream", "normal")
            res = {a: (run_normal if a == "normal" else run_stream)(mt, tts, t) for a in arms}
            n, s = res["normal"], res["stream"]
            rows.append({
                "rep": r, "i": i, "es": t, "en": n["en"], "words_en": len(n["en"].split()),
                "same_text": n["en"] == s["en"], "same_plan": n["plan"] == s["plan"], "same_pcm": n["pcm"] == s["pcm"],
                "head_early": s["head_early"], "head_ok": s.get("head_ok", True),
                **{f"{a}_{k}": round(res[a][k], 1) if res[a][k] is not None else None
                   for a in ("normal", "stream") for k in ("first_ms", "second_ms", "mt_ms", "mt_pure_ms")},
            })
    early = [x for x in rows if x["head_early"]]
    summary = {
        "n": len(rows), "head_early": len(early),
        "same_text": sum(x["same_text"] for x in rows), "same_plan": sum(x["same_plan"] for x in rows),
        "same_pcm": sum(x["same_pcm"] for x in rows), "head_mismatch": sum(not x["head_ok"] for x in rows),
        "first_gain_ms_p50_all": p50([x["normal_first_ms"] - x["stream_first_ms"] for x in rows
                                      if x["normal_first_ms"] and x["stream_first_ms"]]),
        "first_gain_ms_p50_early": p50([x["normal_first_ms"] - x["stream_first_ms"] for x in early]),
        "second_delta_ms_p50_early": p50([x["stream_second_ms"] - x["normal_second_ms"] for x in early
                                          if x["stream_second_ms"] and x["normal_second_ms"]]),
        "mt_pure_delta_ms_p50": p50([x["stream_mt_pure_ms"] - x["normal_mt_pure_ms"] for x in rows]),
        "normal_first_ms_p50": p50([x["normal_first_ms"] for x in rows]),
        "stream_first_ms_p50": p50([x["stream_first_ms"] for x in rows]),
    }
    doc = {"cache_limit_bytes": cap, "reps": REPS, "qwen_dir": QDIR, "first_chunk_max_words": cfg.tts_first_chunk_max_words,
           "load": os.getloadavg(), "summary": summary, "rows": rows}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])

"""Cobertura por ítem de una corrida continua: asr_final atribuidos a cada ítem por posición de endpoint.

Uso: .venv/bin/python scripts/eval/continuous_coverage.py RUN_DIR [MANIFEST]
Atribución: un commit pertenece al último ítem cuyo start_s <= audio_pos_s de su endpoint (posiciones del stream).
Informa por ítem: nº de segmentos, texto ASR concatenado (crudo si el adaptador lo registró), WER vs transcript_es,
palabras de referencia no cubiertas y duplicados EXACTOS (mismo texto normalizado en >1 segmento del ítem).
Más de un segmento por ítem NO prueba omisión, duplicado ni sobre-segmentación incorrecta: puede ser fragmentación
legítima; se informa aparte de los duplicados exactos. Diagnóstico sobre límites estimados, no medida de calidad.
"""
import json
import re
import sys
from pathlib import Path

import jiwer

_PUNCT = re.compile(r"[^\w\sáéíóúüñ]", re.I)


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", _PUNCT.sub(" ", (t or "").lower())).strip()


def main(run_dir: str, manifest: str | None) -> None:
    d = Path(run_dir)
    run = json.loads((d / "run.json").read_text(encoding="utf-8"))
    mpath = Path(manifest) if manifest else d / "manifest.jsonl"
    ref = {json.loads(l)["item_id"]: json.loads(l).get("transcript_es") for l in mpath.read_text(encoding="utf-8").splitlines()}
    raw = {(e["stream_id"], e["commit_id"]): e for e in run["meta"].get("adapter_metadata", {}).get("asr_text_log", [])}
    for line in (d / "traces.jsonl").read_text(encoding="utf-8").splitlines():
        t = json.loads(line)
        if t.get("warmup") or not t.get("continuous"):
            continue
        segs = sorted(t["segments"], key=lambda s: s["start_s"] or 0)
        items = {s["item_id"]: [] for s in segs}
        endpoints = {e["commit_id"]: e["audio_pos_s"] for e in t["events"] if e["kind"] == "endpoint" and e.get("commit_id")}
        unattributed = []
        for e in t["events"]:
            if e["kind"] != "asr_final":
                continue
            pos = endpoints.get(e.get("commit_id"))
            owner = None
            if pos is not None:
                for s in segs:
                    if s["start_s"] is not None and s["start_s"] <= pos:
                        owner = s["item_id"]
            (items[owner] if owner else unattributed).append(e)
        print(f"== stream {t['stream_id']}: {sum(len(v) for v in items.values())} asr_final en {len(segs)} ítems, sin atribuir {len(unattributed)}")
        for iid, evs in items.items():
            texts = [(raw.get((t["stream_id"], e.get("commit_id")), {}).get("asr_raw") or e.get("text") or "") for e in evs]
            hyp = norm(" ".join(texts)); r = norm(ref.get(iid) or "")
            norms = [norm(x) for x in texts if norm(x)]
            dup = len(norms) - len(set(norms))
            wer = jiwer.wer(r, hyp) if r and hyp else None
            miss = ""
            if r:
                miss = jiwer.process_words(r, hyp).deletions if hyp else len(r.split())
            print(f"  {iid}: segmentos={len(evs)} duplicados_exactos={dup} WER={wer if wer is None else round(wer,3)} palabras_ref_borradas={miss}")
            print(f"     ASR: {' | '.join(texts)}")
        for e in unattributed:
            print("  SIN ATRIBUIR:", e.get("text"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)

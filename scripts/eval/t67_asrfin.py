#!/usr/bin/env python3
"""[bench] T-67: resumen por brazo del finish especulativo del ASR (T-58) y del ASR final.

Uso: t67_asrfin.py <dir latency de brazo> [...]   (cada dir con run.json y stages.md al lado)
Muestra de run.json: meta.adapter_metadata.asr_spec_finish (started/hit/miss/discard_voice, finish_ms,
refeed_ms) y WER del ASR (quality.candidate_asr_es.human). De stages.md toma la fila 'todas'
(etapa ASR final p50). Solo medicion.
"""
import json, pathlib, re, statistics, sys


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))] if v else float("nan")


for d in map(pathlib.Path, sys.argv[1:]):
    arm = d.parent.name
    try:
        j = json.loads((d / "run.json").read_text())
    except Exception as e:
        print(f"{arm}: sin run.json ({e})"); continue
    s = j.get("meta", {}).get("adapter_metadata", {}).get("asr_spec_finish", {})
    fin, ref = s.get("finish_ms", []), s.get("refeed_ms", [])
    wer = j.get("quality", {}).get("candidate_asr_es", {}).get("human", {}).get("wer")
    st = (d.parent / "stages.md")
    row = None
    if st.exists():
        m = re.search(r"^\| todas \|(.*)$", st.read_text(), re.M)
        row = m[1].strip() if m else None
    print(f"{arm}: enabled={s.get('enabled')} started={s.get('started')} hit={s.get('hit')} miss={s.get('miss', 0)} "
          f"discard_voice={s.get('discard_voice')} finish_ms p50={pct(fin, .5):.0f} n={len(fin)} "
          f"refeed_ms p50={pct(ref, .5):.0f} n={len(ref)} WER_asr={wer if wer is None else round(wer, 4)}")
    if row: print(f"    stages 'todas': {row}")

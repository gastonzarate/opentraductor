#!/usr/bin/env python3
"""[bench] Duración total del audio TTS por frase (suma pcm_samples/sample_rate de los tts_chunk), mediana por clip y brazo.
Complementa tts_gaps/simul_lag (cuándo llega el audio) con cuánto dura. Uso: t67_duracion.py DIR_LATENCY [DIR_LATENCY...] (cada uno con traces.jsonl; brazo = nombre del padre)."""
import json, statistics as st, sys
from collections import defaultdict
from pathlib import Path
arms = []
res = {}
for d in sys.argv[1:]:
    p = Path(d); arm = p.parent.name if p.name == "latency" else p.name
    per = defaultdict(list)
    for line in (p / "traces.jsonl").read_text().splitlines():
        t = json.loads(line)
        if t["stream_id"].startswith("warm"): continue
        ch = [e for e in t["events"] if e.get("kind") == "tts_chunk"]
        if not ch: continue
        seg = (t.get("segments") or [{}])[0].get("item_id", "?")
        per[seg].append(sum(e["pcm_samples"] / e["sample_rate"] for e in ch))
    res[arm] = {k: st.median(v) for k, v in per.items()}; arms.append(arm)
clips = sorted({c for a in arms for c in res[a]})
print("Duración total del audio TTS por frase (s, mediana de las reps por clip).")
print("| clip | " + " | ".join(arms) + " |"); print("|---|" + "---|" * len(arms))
for c in clips: print(f"| {c} | " + " | ".join(f"{res[a].get(c, float('nan')):.2f}" for a in arms) + " |")
print("| **media** | " + " | ".join(f"{st.mean(res[a].values()):.2f}" if res[a] else "-" for a in arms) + " |")

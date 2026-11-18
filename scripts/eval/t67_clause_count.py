"""T-67 paso 2: piezas por frase solo con texto (sin ASR ni MT) para varias reglas de corte (min_words, cues).
Clips: asr_text_es de artifacts/t73-spec/off/latency/run.json. Charlas: scripts/eval/sets/charlas_ctx_ar.jsonl.
Uso: python3 scripts/eval/t67_clause_count.py"""
import json, sys, glob
sys.path.insert(0, ".")
from scripts.eval.t69_charlas_eval import simul_split, norm_charla
clips = []
for f in glob.glob("artifacts/t73-spec/off/latency/run.json"):
    for s in json.load(open(f)).get("segment_results", []):
        if s.get("outcome") == "ok" and s.get("asr_text_es"): clips.append(s["asr_text_es"])
uniq = sorted(set(clips))
turns = [t["es"] for l in open("scripts/eval/sets/charlas_ctx_ar.jsonl") if l.strip() for t in norm_charla(json.loads(l))["turnos"]]
print("clips frases", len(clips), "distintas", len(uniq), "| charlas turnos", len(turns))
for mw, cues in [(5,""),(3,""),(5,"y,que"),(3,"y"),(3,"que"),(3,"y,que"),(2,"y,que")]:
    r = []
    for name, xs in (("clips", clips), ("charlas", turns)):
        p = [simul_split(x, mw, cues) for x in xs]
        r.append(f"{name} {sum(map(len,p))/len(p):.2f} pz/frase, cortadas {sum(len(q)>1 for q in p)}/{len(p)}")
    print(f"min={mw} cues={cues or '-':6} | " + " | ".join(r))
print("\nclips distintas con min=3 y,que:")
for x in uniq: print(" ", " ‖ ".join(simul_split(x, 3, "y,que")))

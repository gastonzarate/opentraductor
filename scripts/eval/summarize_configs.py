"""Resumen por config de corridas de compare_configs.sh: latencia fin→primer audio por clip, ASR final y desglose."""
import json, sys, glob, statistics as st
from pathlib import Path
out = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/runtime-opt")
for d in sorted(p for p in out.iterdir() if (p / "traces.jsonl").exists()):
    rows = {}
    for l in open(d / "traces.jsonl"):
        t = json.loads(l)
        if t.get("warmup") or t.get("error"):
            continue
        sid = t["stream_id"]; ev = t["events"]
        ep = next((e for e in ev if e["kind"] == "endpoint"), None)
        af = next((e for e in ev if e["kind"] == "asr_final"), None)
        mt = next((e for e in ev if e["kind"] == "mt_final"), None)
        tts = next((e for e in ev if e["kind"] == "tts_chunk"), None)
        seg = (t.get("segments") or [{}])[0]
        rows.setdefault(sid.split("#")[0].split("::")[0][-24:], []).append(
            dict(ep=ep and ep["recv_ns"], af=af and af["recv_ns"], mt=mt and mt["recv_ns"], tts=tts and tts["recv_ns"],
                 text=af and af["text"], seg=seg))
    print("==", d.name)
    for k, v in rows.items():
        d1 = lambda a, b: [round((r[b] - r[a]) / 1e6) for r in v if r[a] and r[b]]
        print(f"  {k}: ep→asr {d1('ep','af')} asr→mt {d1('af','mt')} mt→tts {d1('mt','tts')} | {v[0]['text']}")

"""[bench] T-62: set sintetico (macOS say Paulina es_MX) con laburar/pila/fiaca/guita + 2 controles, para A/B end_to_first de mt_sense_glossary.
Uso: .venv-inc-llm/bin/python scripts/eval/t62_make_idiom_set.py  -> data/audio/t62/*.wav + artifacts/t58-glossE2E/manifest.jsonl
Limite: TTS, no habla humana; sirve para comparar off/on, no como latencia absoluta percibida."""
import json, subprocess, hashlib, soundfile as sf
base = json.loads(open('artifacts/benchmark/offline-review/manifest.jsonl').readline())
P = [("laburar", "Ya no quiero laburar.", "I no longer want to work."),
     ("laburar2", "Mañana tengo que laburar todo el día.", "Tomorrow I have to work all day."),
     ("pila", "El equipo está con toda la pila.", "The team is full of energy."),
     ("fiaca", "Hoy me da fiaca salir de casa.", "I can't be bothered to leave the house today."),
     ("guita", "Si quiere más, que ponga más guita.", "If he wants more, he should pay more."),
     ("guita2", "No tengo guita para pagar el alquiler.", "I have no money to pay the rent."),
     ("ctrl1", "Mañana llegamos a las tres de la tarde.", "We arrive at three tomorrow afternoon."),
     ("ctrl2", "Necesito revisar el informe antes de la reunión.", "I need to review the report before the meeting.")]
out = []
for k, es, en in P:
    a = f"data/audio/t62/{k}.wav"; t = f"/tmp/{k}.aiff"
    subprocess.run(["say", "-v", "Paulina", "-o", t, es], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", t, a], check=True)
    d = sf.info(a).duration
    r = dict(base)
    r.update(item_id=f"t62-{k}", recording_id=f"t62-{k}", source_id="t62-say", audio_path=f"t62/{k}.wav",
             sha256=hashlib.sha256(open(a, 'rb').read()).hexdigest(), duration_s=round(d, 2), transcript_es=es,
             transcript_es_provenance="generated", reference_en=[en], reference_en_provenance="generated",
             utterance_start_s=0.0, utterance_end_s=round(d, 2), source_url="macOS say -v Paulina", license="n/a (sintetico)",
             license_url="", attribution="macOS say Paulina es_MX (TTS, no es habla humana)")
    for x in ("fleurs_id", "source_audio_sha256"): r.pop(x, None)
    out.append(r)
open('artifacts/t58-glossE2E/manifest.jsonl', 'w').write("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n")
print([(r['item_id'], r['duration_s']) for r in out])

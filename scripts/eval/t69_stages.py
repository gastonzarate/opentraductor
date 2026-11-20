#!/usr/bin/env python3
"""T-69 (@bench): desglose de latencia por etapa que sirve con MT+TTS en streaming (defaults de T-67).

t58_stage_breakdown.py supone que el 1er tts_chunk llega DESPUES de mt_final; con mt_stream_tts llega antes
(la etapa TTS da negativa). Aqui las etapas se definen sobre eventos que existen en los dos modos.
Origen = fin de habla (t0_ns + segments[0].end_s, el mismo de end_to_first_ms), recv_ns de cada evento.
  espera       = endpoint   - fin de habla     (silencio que el endpointer necesita)
  asr_final    = asr_final  - endpoint         (finish() de Moonshine)
  a_1er_audio  = 1er tts_chunk - asr_final     (1er fragmento de MT + Kokoro hasta el primer audio, juntos)
  total        = 1er tts_chunk - fin de habla  (= end_to_first_ms; espera+asr_final+a_1er_audio)
  info: mt_fin = mt_final - asr_final (la MT entera, ya fuera del camino critico si hay streaming)
        stream = 1 si el 1er audio sale antes que mt_final
No cubre: salida al dispositivo (scripts/eval/device_loopback.py, necesita lease) ni el 1er token de MT por separado
(no hay evento; sale de t69_charlas_eval.py --first-token, mt_1er_token_*).
Uso: python3 scripts/eval/t69_stages.py DIR [DIR ...]   DIR = carpeta con traces.jsonl o con subcarpetas rN/latency (se juntan)
"""
import json, statistics as st, sys, glob, os
from collections import defaultdict


def find(d):
    d = d.rstrip('/')
    if os.path.exists(d + '/traces.jsonl'):
        return [d + '/traces.jsonl']
    rondas = sorted(glob.glob(d + '/r*/latency/traces.jsonl'))  # intercalado: new/rN/latency; new/latency es la copia junta
    return rondas or sorted(glob.glob(d + '/**/traces.jsonl', recursive=True))


def load(dirs):
    rows = []
    for d in dirs:
        for f in find(d):
            for l in open(f):
                t = json.loads(l)
                if t.get('warmup') or t.get('error') or not t.get('segments'):
                    continue
                end = t['t0_ns'] + t['segments'][0]['end_s'] * 1e9
                ev = defaultdict(list)
                for e in t['events']:
                    ev[e['kind']].append(e)
                if not all(ev[k] for k in ('endpoint', 'asr_final', 'tts_chunk')):
                    continue
                ep, af = ev['endpoint'][0], ev['asr_final'][0]
                tc = min(e['recv_ns'] for e in ev['tts_chunk'])
                mf = ev['mt_final'][0]['recv_ns'] if ev['mt_final'] else None
                rows.append(dict(
                    frase=t['stream_id'].split('-', 2)[-1][:24],
                    espera=(ep['recv_ns'] - end) / 1e6,
                    asr_final=(af['recv_ns'] - ep['recv_ns']) / 1e6,
                    a_1er_audio=(tc - af['recv_ns']) / 1e6,
                    total=(tc - end) / 1e6,
                    mt_fin=(mf - af['recv_ns']) / 1e6 if mf else float('nan'),
                    stream=1 if (mf and tc < mf) else 0))
    return rows


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, round(q * (len(v) - 1)))]


def main(dirs):
    rows = load(dirs)
    if not rows:
        sys.exit('sin trazas validas en ' + ' '.join(dirs))
    neg = [r for r in rows if r['espera'] < 0]
    rows = [r for r in rows if r['espera'] >= 0]  # endpoint ANTES del fin del clip: el clip se parte (corte), no es espera
    ks = ['espera', 'asr_final', 'a_1er_audio', 'total']
    print(f'n={len(rows)} trazas ({" ".join(dirs)}); fuera {len(neg)} con endpoint antes del fin del clip (clip partido)')
    print('| etapa | p50 | p95 | min-max |')
    print('|---|---|---|---|')
    for k in ks + ['mt_fin']:
        v = [r[k] for r in rows if r[k] == r[k]]
        if v:
            print(f'| {k} | {st.median(v):.0f} | {pct(v, .95):.0f} | {min(v):.0f}-{max(v):.0f} |')
    print(f'1er audio antes de mt_final (streaming): {sum(r["stream"] for r in rows)}/{len(rows)}')
    print('max |espera+asr_final+a_1er_audio - total| = %.2f ms' % max(abs(r['espera'] + r['asr_final'] + r['a_1er_audio'] - r['total']) for r in rows))
    g = defaultdict(list)
    for r in rows:
        g[r['frase']].append(r)
    print('\npor clip (p50):')
    print('| clip | n | ' + ' | '.join(ks) + ' |')
    print('|---|---|' + '---|' * len(ks))
    for name, v in sorted(g.items()):
        print(f'| {name} | {len(v)} | ' + ' | '.join(f'{st.median(r[k] for r in v):.0f}' for k in ks) + ' |')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])

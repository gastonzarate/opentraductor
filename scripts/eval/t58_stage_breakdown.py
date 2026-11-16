#!/usr/bin/env python3
"""T-58: desglose por etapa de end_to_first_ms a partir de traces.jsonl.

Origen = fin de habla del segmento (t0_ns + segments[0].end_s), el mismo que usa end_to_first_ms.
Etapas (recv_ns de cada evento):
  espera   = endpoint        - fin de habla   (silencio que el endpointer necesita + cola de audio)
  asr_fin  = asr_final       - endpoint       (finish() de Moonshine y lo que lo precede)
  mt       = mt_final        - asr_final      (Qwen en el hilo GPU)
  tts      = 1er tts_chunk   - mt_final       (Kokoro hasta primer audio)
La suma es end_to_first_ms (se comprueba, max |dif| se imprime).
Uso: python3 scripts/eval/t58_stage_breakdown.py artifacts/<dir>/latency/store [--md]"""
import json, statistics as st, sys
from collections import defaultdict

b = sys.argv[1].rstrip('/') + '/'
rows = []
for l in open(b + 'traces.jsonl'):
    t = json.loads(l)
    if t.get('warmup') or t.get('error') or not t.get('segments'):
        continue
    end_ns = t['t0_ns'] + t['segments'][0]['end_s'] * 1e9
    ev = defaultdict(list)
    for e in t['events']:
        ev[e['kind']].append(e)
    if not all(ev[k] for k in ('endpoint', 'asr_final', 'mt_final', 'tts_chunk')):
        continue
    ep, af, mf = ev['endpoint'][0], ev['asr_final'][0], ev['mt_final'][0]
    tc = min(e['recv_ns'] for e in ev['tts_chunk'])
    r = dict(
        frase=t['stream_id'].split('-', 2)[2][:24],
        espera=(ep['recv_ns'] - end_ns) / 1e6,
        silencio_audio=(ep['audio_pos_s'] - t['segments'][0]['end_s']) * 1e3 if ep.get('audio_pos_s') else float('nan'),
        asr_fin=(af['recv_ns'] - ep['recv_ns']) / 1e6,
        mt=(mf['recv_ns'] - af['recv_ns']) / 1e6,
        tts=(tc - mf['recv_ns']) / 1e6,
        total=(tc - end_ns) / 1e6,
    )
    rows.append(r)

keys = ['espera', 'asr_fin', 'mt', 'tts', 'total']
p = lambda v, q: sorted(v)[min(len(v) - 1, round(q * (len(v) - 1)))]
print(f'n={len(rows)}  ({b})')
print('| frase | n | espera fin de habla | ASR final | MT | Kokoro 1er audio | total |')
print('|---|---|---|---|---|---|---|')
g = defaultdict(list)
for r in rows:
    g[r['frase']].append(r)
for name, v in list(g.items()) + [('todas', rows)]:
    print(f'| {name} | {len(v)} | ' + ' | '.join(f'{st.median(r[k] for r in v):.0f}' for k in keys) + ' |')
print('p95 todas: ' + ' · '.join(f'{k} {p([r[k] for r in rows], .95):.0f}' for k in keys))
print('min-max todas: ' + ' · '.join(f'{k} {min(r[k] for r in rows):.0f}-{max(r[k] for r in rows):.0f}' for k in keys))
sa = [r['silencio_audio'] for r in rows if r['silencio_audio'] == r['silencio_audio']]
if sa:
    print(f'audio_pos(endpoint) - fin de habla: p50 {st.median(sa):.0f} ms (min {min(sa):.0f}, max {max(sa):.0f})')
print('max |suma etapas - total| = %.2f ms' % max(abs(r['espera'] + r['asr_fin'] + r['mt'] + r['tts'] - r['total']) for r in rows))

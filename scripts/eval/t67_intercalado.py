#!/usr/bin/env python3
"""T-67: resume la tanda de t67_intercalado.sh (brazos intercalados por ronda).
Uso: python3 scripts/eval/t67_intercalado.py OUTDIR [--ref off]
Por brazo: p50/p95 de end_to_first (total) y de cada etapa, diferencia contra --ref (p50 agrupado y mediana de las
diferencias por ronda, que cancela mejor la carga que pega a todos), y carga 1 min de las corridas.
Además junta las trazas de cada brazo en OUT/<brazo>/latency/ para los t67_*.py (cortes, huecos TTS, duración).
Escribe OUT/RESUMEN.md."""
import csv, json, pathlib, shutil, statistics as st, sys
from collections import defaultdict

out = pathlib.Path(sys.argv[1]); ref = sys.argv[sys.argv.index('--ref') + 1] if '--ref' in sys.argv else 'off'
KEYS = ['espera', 'asr_fin', 'mt', 'tts', 'total']
pct = lambda v, q: sorted(v)[min(len(v) - 1, round(q * (len(v) - 1)))]


def rows_of(path):
    res = []
    for l in open(path):
        t = json.loads(l)
        if t.get('warmup') or t.get('error') or not t.get('segments') or str(t.get('stream_id', '')).startswith('warmup'):
            continue
        end_ns = t['t0_ns'] + t['segments'][0]['end_s'] * 1e9
        ev = defaultdict(list)
        for e in t['events']:
            ev[e['kind']].append(e)
        if not all(ev[k] for k in ('endpoint', 'asr_final', 'mt_final', 'tts_chunk')):
            continue
        ep, af, mf = ev['endpoint'][0], ev['asr_final'][0], ev['mt_final'][0]
        tc = min(e['recv_ns'] for e in ev['tts_chunk'])
        res.append(dict(espera=(ep['recv_ns'] - end_ns) / 1e6, asr_fin=(af['recv_ns'] - ep['recv_ns']) / 1e6,
                        mt=(mf['recv_ns'] - af['recv_ns']) / 1e6, tts=(tc - mf['recv_ns']) / 1e6, total=(tc - end_ns) / 1e6))
    return res


arms = [d.name for d in sorted(out.iterdir(), key=lambda p: p.stat().st_mtime) if d.is_dir() and list(d.glob('r*/latency/traces.jsonl'))]
if ref in arms:
    arms.remove(ref); arms.insert(0, ref)
data = {}      # arma -> ronda -> filas
for a in arms:
    data[a] = {}
    merged = out / a / 'latency'; merged.mkdir(exist_ok=True)
    with open(merged / 'traces.jsonl', 'w') as mf:
        for rd in sorted((out / a).glob('r*/latency'), key=lambda p: int(p.parent.name[1:])):
            tp = rd / 'traces.jsonl'
            if not tp.exists():
                continue
            data[a][int(rd.parent.name[1:])] = rows_of(tp)
            mf.write(tp.read_text())
            if not (merged / 'run.json').exists() and (rd / 'run.json').exists():
                shutil.copy(rd / 'run.json', merged / 'run.json')

loads = defaultdict(list)
lp = out / 'loads.csv'
if lp.exists():
    for r in csv.DictReader(open(lp)):
        loads[r['brazo']].append((int(r['ronda']), float(r['load1_antes']), float(r['load1_despues']), r['rc']))

L = []
sha = (out / 'sha.txt').read_text().strip() if (out / 'sha.txt').exists() else '?'
L.append(f'# T-67 intercalado · sha {sha} · referencia: {ref}\n')
L.append('| brazo | n | rondas | total p50 | total p95 | Δp50 vs ref | Δp50 pareado | Δp95 vs ref | espera | ASR fin | MT | TTS 1er audio | load1 med/máx |')
L.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
allref = [r for rr in data.get(ref, {}).values() for r in rr]
for a in arms:
    rows = [r for rr in data[a].values() for r in rr]
    if not rows:
        continue
    t = [r['total'] for r in rows]
    p50, p95 = st.median(t), pct(t, .95)
    if a != ref and allref:
        rp50, rp95 = st.median(r['total'] for r in allref), pct([r['total'] for r in allref], .95)
        pair = [st.median(r['total'] for r in data[a][k]) - st.median(r['total'] for r in data[ref][k])
                for k in data[a] if data[a][k] and data[ref].get(k)]
        d50, d95 = f'{p50 - rp50:+.0f}', f'{p95 - rp95:+.0f}'
        dp = f'{st.median(pair):+.0f} (n={len(pair)})' if pair else '-'
    else:
        d50 = d95 = dp = '-'
    ls = loads.get(a, [])
    lv = [x for _, b, c, _ in ls for x in (b, c)]
    lt = f'{st.median(lv):.1f}/{max(lv):.1f}' if lv else '-'
    stg = ' | '.join(f'{st.median(r[k] for r in rows):.0f}' for k in ('espera', 'asr_fin', 'mt', 'tts'))
    L.append(f'| {a} | {len(rows)} | {len(data[a])} | {p50:.0f} | {p95:.0f} | {d50} | {dp} | {d95} | {stg} | {lt} |')
L.append('\nTiempos en ms desde fin de habla. «Δp50 pareado» = mediana de (p50 del brazo − p50 de la ref) dentro de cada ronda.')
L.append('TTS 1er audio negativo = el audio sale antes de que termine la MT (streaming).\n')
L.append('## Por ronda (p50 total, ms · load1 antes→después)\n')
rounds = sorted({k for a in arms for k in data[a]})
L.append('| ronda | ' + ' | '.join(arms) + ' |')
L.append('|---|' + '---|' * len(arms))
for k in rounds:
    cells = []
    for a in arms:
        rr = data[a].get(k)
        ld = next((f'{b:.1f}→{c:.1f}' for kk, b, c, _ in loads.get(a, []) if kk == k), '-')
        cells.append(f'{st.median(r["total"] for r in rr):.0f} · {ld}' if rr else '-')
    L.append(f'| {k} | ' + ' | '.join(cells) + ' |')
(out / 'RESUMEN.md').write_text('\n'.join(L) + '\n')
print('\n'.join(L))

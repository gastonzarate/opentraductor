#!/usr/bin/env python3
"""T-58: por frase, gap recv_ns asr_final->mt_final vs mt_ms y mt_compute_ms (run.json meta.adapter_metadata.mt_timing_log).
Uso: python3 scripts/eval/t58_mt_timing_table.py artifacts/<dir>/latency/store"""
import json, statistics as st, sys
from collections import defaultdict
b = sys.argv[1].rstrip('/') + '/'
tl = {x['commit_id']: x for x in json.load(open(b + 'run.json'))['meta']['adapter_metadata']['mt_timing_log']}
g = defaultdict(list)
for l in open(b + 'traces.jsonl'):
    t = json.loads(l)
    if t.get('warmup'): continue
    by = defaultdict(dict)
    for e in t['events']:
        if e['kind'] in ('asr_final', 'mt_final'): by[e.get('commit_id')][e['kind']] = e
    for c, x in by.items():
        if len(x) == 2 and c in tl:
            gap = (x['mt_final']['recv_ns'] - x['asr_final']['recv_ns']) / 1e6
            g[c.split(':')[0].split('-', 2)[2]].append((len(x['asr_final']['text'].split()), gap, tl[c]['mt_ms'], tl[c]['mt_compute_ms']))
allr = [r for v in g.values() for r in v]
print('| frase | pal. | n | gap p50 | mt_ms p50 | compute p50 | compute min-max |\n|---|---|---|---|---|---|---|')
for k, v in g.items():
    print(f'| {k[:32]} | {v[0][0]} | {len(v)} | {st.median(r[1] for r in v):.0f} | {st.median(r[2] for r in v):.0f} | {st.median(r[3] for r in v):.0f} | {min(r[3] for r in v):.0f}-{max(r[3] for r in v):.0f} |')
print(f'| todas | | {len(allr)} | {st.median(r[1] for r in allr):.0f} | {st.median(r[2] for r in allr):.0f} | {st.median(r[3] for r in allr):.0f} | {min(r[3] for r in allr):.0f}-{max(r[3] for r in allr):.0f} |')
print(f'max |gap-mt_ms| = {max(abs(r[1]-r[2]) for r in allr):.1f} ms · max |mt_ms-compute| = {max(abs(r[2]-r[3]) for r in allr):.1f} ms')

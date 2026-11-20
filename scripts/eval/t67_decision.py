#!/usr/bin/env python3
"""@bench T-67: regla de decision de la tanda final (pedido @lead 20/11).
Uso: python3 scripts/eval/t67_decision.py OUTDIR [--ref new] [--min-delta 20] [--min-rondas 8]
Por brazo != ref: en cada ronda, p50 total del brazo - p50 total de la ref (misma ronda, mismos clips). Cuenta las rondas con
diferencia <= -min-delta. Gana si cuenta >= min-rondas (de las rondas completas) y ademas 0 cortes falsos y 0 huecos TTS
(los lee de cuts.txt y tts_gaps.txt de OUTDIR; si faltan, lo dice). Si ninguno cumple, queda la ref."""
import re, statistics as st, sys, pathlib, importlib.util

out = pathlib.Path(sys.argv[1])
arg = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
ref, mind, minr = arg('--ref', 'new'), arg('--min-delta', 20), arg('--min-rondas', 8)
spec = importlib.util.spec_from_file_location('ti', pathlib.Path(__file__).with_name('t67_intercalado.py'))
# t67_intercalado.py corre al importar (usa sys.argv): reutilizamos solo rows_of leyendo el codigo, sin ejecutarlo.
src = pathlib.Path(__file__).with_name('t67_intercalado.py').read_text()
ns = {}
exec('import json, statistics as st\nfrom collections import defaultdict\n' + src[src.index('def rows_of'):src.index('arms = [d.name')], ns)
rows_of = ns['rows_of']
data = {}
for a in sorted(d.name for d in out.iterdir() if d.is_dir()):
    rr = {int(p.parent.parent.name[1:]): rows_of(p) for p in (out / a).glob('r*/latency/traces.jsonl')}
    if rr:
        data[a] = {k: v for k, v in rr.items() if v}
if ref not in data:
    sys.exit(f'sin ref {ref}')
nr = {a: len(v) for a, v in data.items()}
L = [f'# T-67 final · regla: Δp50 por ronda ≤ -{mind} ms vs {ref} en ≥{minr} rondas, 0 cortes falsos, 0 huecos TTS\n',
     '| brazo | rondas | rondas a favor (≤-%d) | rondas a favor (<0) | Δp50 por ronda mediana | p50 total | p95 total | n |' % mind, '|---|---|---|---|---|---|---|---|']
res = {}
for a, d in data.items():
    allr = [r['total'] for rr in d.values() for r in rr]
    p50, p95 = st.median(allr), sorted(allr)[min(len(allr) - 1, round(.95 * (len(allr) - 1)))]
    if a == ref:
        L.append(f'| {a} | {len(d)} | - | - | - | {p50:.0f} | {p95:.0f} | {len(allr)} |'); continue
    diffs = [st.median(r['total'] for r in d[k]) - st.median(r['total'] for r in data[ref][k]) for k in d if k in data[ref]]
    ok = sum(x <= -mind for x in diffs); neg = sum(x < 0 for x in diffs)
    res[a] = (ok, len(diffs))
    L.append(f'| {a} | {len(diffs)} | {ok}/{len(diffs)} | {neg}/{len(diffs)} | {st.median(diffs):+.0f} | {p50:.0f} | {p95:.0f} | {len(allr)} |')
L.append('\nΔ por ronda (ms, brazo - ref): ' + '; '.join(f'{a}: ' + ' '.join(f'{st.median(r["total"] for r in d[k]) - st.median(r["total"] for r in data[ref][k]):+.0f}' for k in sorted(d) if k in data[ref]) for a, d in data.items() if a != ref))
L.append('\n## Cortes falsos y huecos TTS (de cuts.txt y tts_gaps.txt)\n')
for f in ('cuts.txt', 'tts_gaps.txt'):
    p = out / f
    L.append(f'- {f}: ' + ('ver archivo (revisar a mano: cortes falsos y huecos deben ser 0 en el brazo)' if p.exists() else 'FALTA'))
L.append('\n## Veredicto por la regla de latencia (falta confirmar cortes/huecos y términos)\n')
win = [a for a, (ok, n) in res.items() if ok >= minr]
L.append(('Cumplen latencia: ' + ', '.join(win) if win else f'Nadie cumple: queda {ref}.'))
(out / 'DECISION.md').write_text('\n'.join(L) + '\n'); print('\n'.join(L))

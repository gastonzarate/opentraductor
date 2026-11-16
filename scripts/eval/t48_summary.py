"""T-48: tabla ASR por backend a partir de los run.json de `benchmark run`.

Por backend y set: WER (jiwer, minúsculas, sin puntuación), % de streams cuya transcripción sale en inglés,
p50/p95 de end_to_first_ms y outcomes. Para el set corto también WER por largo (1, 2, 3-4 palabras).

"Sale en inglés": más palabras funcionales inglesas que españolas en la transcripción (lista fija abajo).
Es una heurística transparente; los casos dudosos se listan para revisarlos a mano.

Uso: .venv/bin/python scripts/eval/t48_summary.py artifacts/t48/runs > artifacts/t48/RESULTS.md
     (espera artifacts/t48/runs/<set>/<backend>/run.json y <set>/manifest.path)
"""
from __future__ import annotations

import json
import re
import statistics
import sys
import unicodedata
from pathlib import Path

import jiwer

EN = set("""yeah yes yep the a an you i it its is are was what there here go hold on right oh let's lets come back
can see we if want to and of for that this my me your hello hi okay thanks thank perfect sorry well just
how do does not no's call any see look good great fine wait sure""".split())
ES = set("""sí si el la los las un una de del que y en es está están no me te se lo le ya pero para con por
como cómo qué hola dale bueno listo claro gracias pará vos tenés andás mirá esperá todo bien día una
vamos nuevo razón idea toque che estás ves escuchás escucho pantalla cortó vuelvo ahora segundo""".split())


def norm(t: str) -> str:
    t = unicodedata.normalize("NFC", t.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", t)).strip()


def is_english(hyp: str) -> bool | None:
    w = norm(hyp).split()
    if not w:
        return None
    en = sum(x in EN for x in w)
    es = sum(x in ES for x in w) + sum(any(c in x for c in "áéíóúñ") for x in w)
    return en > es


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def load_refs(run_dir: Path) -> dict[str, tuple[str, int]]:
    """item_id -> (referencia, n_palabras) desde el manifest copiado en el run o el de origen."""
    link = run_dir.parent / "manifest.path"  # lo escribe t48_asr_compare.sh
    man = Path(link.read_text().strip()) if link.is_file() else Path("/nonexistent")
    for cand in (run_dir / "manifest_subset.jsonl", man):
        if cand.is_file():
            out = {}
            for line in cand.read_text().splitlines():
                if line.strip():
                    d = json.loads(line)
                    out[d["item_id"]] = (d.get("transcript_es") or "", len((d.get("transcript_es") or "").split()))
            return out
    return {}


def summarize(run_dir: Path) -> tuple[dict, list[tuple[str, str, str]]]:
    rj = json.loads((run_dir / "run.json").read_text())
    refs = load_refs(run_dir)
    segs = [s for s in rj["segment_results"] if not s["stream_id"].startswith("warmup")]
    lat = [s["end_to_first_ms"] for s in segs if s.get("outcome") == "ok" and s.get("end_to_first_ms") is not None]
    outcomes: dict[str, int] = {}
    for s in segs:
        outcomes[s.get("outcome", "?")] = outcomes.get(s.get("outcome", "?"), 0) + 1
    by_len: dict[str, tuple[list[str], list[str]]] = {}
    R, H, en_flags, english = [], [], [], []
    for s in segs:
        ref, nw = refs.get(s["item_id"], ("", 0))
        hyp = s.get("asr_text_es") or ""
        if not ref:
            continue
        R.append(norm(ref) or "_"); H.append(norm(hyp))
        flag = is_english(hyp)
        en_flags.append(bool(flag))
        if flag:
            english.append((s["item_id"], ref, hyp))
        key = "1" if nw <= 1 else "2" if nw == 2 else "3-4" if nw <= 4 else "5+"
        by_len.setdefault(key, ([], []))
        by_len[key][0].append(norm(ref) or "_"); by_len[key][1].append(norm(hyp))
    row = {
        "n": len(segs), "outcomes": outcomes,
        "wer": jiwer.wer(R, H) if R else None,
        "english_pct": 100 * sum(en_flags) / len(en_flags) if en_flags else None,
        "p50": pct(lat, 0.5), "p95": pct(lat, 0.95),
        "wer_by_len": {k: jiwer.wer(r, h) for k, (r, h) in sorted(by_len.items())},
        "load": tuple(round(((x or {}).get("loadavg_1_5_15") or [float("nan")])[0], 1)
                      for x in ((rj["meta"].get("environment") or {}).get("load_at_start"), rj["meta"].get("load_at_end"))),
        "contention": rj["meta"].get("contention_possible"),
    }
    return row, english


def fmt(x, nd=3):
    return "n/d" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main(root: Path) -> None:
    print("# T-48 · ASR en el pipeline baseline (Marian): parakeet vs moonshine\n")
    for set_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        runs = sorted(p for p in set_dir.iterdir() if (p / "run.json").is_file())
        if not runs:
            continue
        print(f"## {set_dir.name}\n")
        print("| backend | streams | WER | % en inglés | WER 1 pal. | WER 2 pal. | WER 3-4 pal. | WER 5+ | e2f p50 ms | e2f p95 ms | outcomes | load ini/fin | contención |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        bad_all = []
        for r in runs:
            row, bad = summarize(r)
            wl = row["wer_by_len"]
            print(f"| {r.name} | {row['n']} | {fmt(row['wer'])} | {fmt(row['english_pct'], 1)} | {fmt(wl.get('1'))} | "
                  f"{fmt(wl.get('2'))} | {fmt(wl.get('3-4'))} | {fmt(wl.get('5+'))} | {fmt(row['p50'], 0)} | "
                  f"{fmt(row['p95'], 0)} | {row['outcomes']} | {row['load']} | {row['contention']} |")
            bad_all += [(r.name, *b) for b in bad]
        if bad_all:
            print("\nSalieron en inglés:\n")
            for b, it, ref, hyp in bad_all[:60]:
                print(f"- {b} · {it}: «{ref}» → «{hyp}»")
        print()


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/t48/runs"))

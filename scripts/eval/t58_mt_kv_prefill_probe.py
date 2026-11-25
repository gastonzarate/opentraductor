"""T-58: ¿cuánto ahorra hacer el prefill de Qwen con el prefijo estable del parcial durante la espera de endpoint?

Idea: mientras corre el endpoint (160 ms) + ASR finish (~143 ms), cargar en la caché KV el prompt fijo
más el parcial SIN su última palabra (la última suele cambiar: "planta" -> "plantación"). Con el final, solo se
procesan los tokens que faltan y el decode. El prefill también despierta la GPU, así que reemplaza al prewarm.

Brazos, cada repetición después de IDLE_S de GPU ociosa (como en el pipeline):
  prewarm  forward de 1 token sobre prompt("hola") y después translate(final) completo (c1f180d)
  kv       prefill del prefijo estable en una caché nueva; con el final: LCP de tokens, recorte de la caché
           y generate solo con la cola. Mide desde que llega el final.
Reporta p50 por frase, costo del prefill y si la salida es idéntica a translate(final) (greedy).

Uso (con la Mac libre, sin otras mediciones corriendo):
  OPENTRADUCTOR_QWEN_DIR=artifacts/analysis/llm-mp4-a/models/qwen3-4b-instruct-2507-4bit \
  .venv-inc-llm/bin/python scripts/eval/t58_mt_kv_prefill_probe.py artifacts/t58-kv-prefill/probe.json
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# (último parcial, final) de los traces de la base tuned (artifacts/t58-tuned-base/latency/store)
PAIRS = [
    ("el granizo destruyó toda la planta", "el granizo destruyó toda la plantación de lechuga"),
    ("averiguame precios cuando es temporada baja cuando es temporada alta y opciones de cobsiones",
     "averiguame precios cuando es temporada baja cuando es temporada alta y opciones de colectivo"),
    ("dele al lugar toda la dignidad la solemnidad y el respeto que amerita no bromee acerca del holocausto",
     "dele al lugar toda la dignidad la solemnidad y el respeto que amerita no bromee acerca del holocausto o el nazismo"),
]
IDLE_S = float(os.environ.get("IDLE_S", "2.0"))
REPS = int(os.environ.get("REPS", "9"))


def main(out: str) -> None:
    import mlx.core as mx
    from mlx_lm import generate
    from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

    from runtime import components as K

    cap = os.environ.get("OPENTRADUCTOR_MLX_CACHE_MB", "256")
    mx.set_cache_limit(int(float(cap) * 1024 * 1024))
    q = K.QwenMLX(os.environ["OPENTRADUCTOR_QWEN_DIR"])
    tok = q.tok

    def enc(s: str) -> list[int]:
        return list(tok.encode(s, add_special_tokens=False))

    def prefix_tokens(stable: str) -> list[int]:
        p = q.prompt(stable)
        return enc(p[: p.index(stable) + len(stable)])

    def kv_translate(cache, cached: list[int], final: str) -> tuple[str, int]:
        full = enc(q.prompt(final))
        n = 0
        while n < min(len(cached), len(full) - 1) and cached[n] == full[n]:
            n += 1
        if len(cached) > n:
            trim_prompt_cache(cache, len(cached) - n)
        txt = generate(q.model, tok, prompt=full[n:], prompt_cache=cache,
                       max_tokens=3 * len(final.split()) + 16, verbose=False)
        return txt.strip().split("\n")[0].strip(), len(full) - n

    for p, f in PAIRS:  # calentamiento
        q.translate(f)
    res = {"idle_s": IDLE_S, "reps": REPS, "pairs": []}
    for partial, final in PAIRS:
        ref = q.translate(final)
        stable = " ".join(partial.split()[:-1])
        row = {"final": final, "stable": stable, "ref": ref, "prewarm_ms": [], "kv_ms": [], "prefill_ms": [],
               "kv_same": 0, "tail_tokens": None, "prefix_tokens": None}
        for _ in range(REPS):
            time.sleep(IDLE_S)
            q.prewarm()
            t = time.perf_counter()
            q.translate(final)
            row["prewarm_ms"].append((time.perf_counter() - t) * 1000)

            time.sleep(IDLE_S)
            t = time.perf_counter()
            cache = make_prompt_cache(q.model)
            cached = prefix_tokens(stable)
            q.model(mx.array(cached)[None], cache=cache)
            mx.eval([c.state for c in cache])
            row["prefill_ms"].append((time.perf_counter() - t) * 1000)
            t = time.perf_counter()
            out_kv, tail = kv_translate(cache, cached, final)
            row["kv_ms"].append((time.perf_counter() - t) * 1000)
            row["kv_same"] += int(out_kv == ref)
            row["tail_tokens"], row["prefix_tokens"] = tail, len(cached)
            row["kv_out"] = out_kv
        for k in ("prewarm_ms", "kv_ms", "prefill_ms"):
            row[k + "_p50"] = round(statistics.median(row[k]), 1)
        res["pairs"].append(row)
        print(f"{final[:28]!r:32} prewarm {row['prewarm_ms_p50']:6.1f}  kv {row['kv_ms_p50']:6.1f}  "
              f"prefill {row['prefill_ms_p50']:6.1f}  igual {row['kv_same']}/{REPS}  "
              f"tokens prefijo {row['prefix_tokens']} cola {row['tail_tokens']}", flush=True)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])

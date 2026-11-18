"""T-68 sonda: Qwen 4B con contexto largo (caché KV de sesión, tope por tokens) con y sin borrador 0.6B.

Simula una charla: por frase, prefill() con el parcial y translate_stream() con la frase completa, como el pipeline.
Mide ms del translate (cola + decode) y el 1.er token, y compara la salida con/sin borrador (greedy: debería coincidir).

    OPENTRADUCTOR_QWEN_DIR=... .venv/bin/python scripts/eval/t68_ctx_spec_probe.py --draft DIR --cap 2000 --out X.json
"""
import argparse
import json
import statistics
import time

from runtime import components as K

TALK = [
    "Bueno, arrancamos con la daily, che.",
    "Ayer estuve laburando en el tema de los pagos con tarjeta.",
    "El endpoint de cobros todavía devuelve un error quinientos cuando la tarjeta está vencida.",
    "Hay que ver si el proveedor nos cobra la comisión igual o no.",
    "Si sale más de diez lucas por mes, no nos conviene.",
    "Martín dijo que lo podemos patear para el sprint que viene.",
    "Yo prefiero no patearlo, porque el cliente ya se quejó dos veces.",
    "La otra opción es poner un reintento con espera exponencial.",
    "Eso lo hago yo, me lleva un par de horas, no más.",
    "¿Alguien revisó el pull request de la migración de la base?",
    "Lo miré a la mañana, está bárbaro, solo dejé dos comentarios.",
    "Uno era sobre el índice de la tabla de usuarios, que no hace falta.",
    "Y el otro sobre los tests, que tardan una banda en correr.",
    "Dale, lo arreglo hoy y lo mergeo a la tarde.",
    "Después tenemos que hablar de la guita del presupuesto para servidores.",
    "Este mes gastamos el doble por el quilombo del jueves.",
    "Se cayó el balanceador y escaló todo solo, sin límite.",
    "Hay que poner un tope, si no, nos fundimos.",
    "Ok, lo anoto como acción para Lucía.",
    "Bueno, nada más, gracias a todos, nos vemos mañana.",
]


def run(draft, cap, reps):
    q = K.QwenMLX(K_dir, session_kv=True, sense_glossary=True, draft_dir=draft)
    q.prewarm()
    rows = []
    for rep in range(reps):
        q._sess, hist, sizes = None, [], []
        for es in TALK:
            q.prefill(es, tuple(hist))
            first = []
            t0 = time.perf_counter()
            en = q.translate_stream(es, tuple(hist), on_text=lambda s: first or first.append(time.perf_counter()))
            t1 = time.perf_counter()
            rows.append({"rep": rep, "es": es, "en": en, "ms": 1000 * (t1 - t0),
                         "first_ms": 1000 * ((first[0] if first else t1) - t0),
                         "ctx_tokens": sum(sizes), "tail": (q.kv_stats["tail_tokens"] or [None])[-1]})
            if cap < 0:  # sin contexto (línea base)
                continue
            hist.append((es, en))
            sizes.append(q.turn_tokens(es, en))
            if cap and sum(sizes) > cap:
                while len(hist) > 1 and sum(sizes) > cap // 2:
                    hist.pop(0)
                    sizes.pop(0)
    return rows


if __name__ == "__main__":
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", default=None)
    ap.add_argument("--cap", type=int, default=2000)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--only", choices=("base", "draft"), default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    K_dir = os.environ["OPENTRADUCTOR_QWEN_DIR"]
    res = {}
    for name, d in (("base", None), ("draft", a.draft)):
        if (name == "draft" and not d) or (a.only and name != a.only):
            continue
        rows = run(d, a.cap, a.reps)
        warm = [r for r in rows if r["rep"] > 0] or rows
        res[name] = {"rows": rows,
                     "p50_ms": statistics.median(r["ms"] for r in warm),
                     "p50_first_ms": statistics.median(r["first_ms"] for r in warm),
                     "max_ctx_tokens": max(r["ctx_tokens"] for r in rows)}
        print(name, {k: v for k, v in res[name].items() if k != "rows"}, flush=True)
    if "draft" in res and "base" in res:
        b, d = res["base"]["rows"], res["draft"]["rows"]
        res["same_output"] = sum(x["en"] == y["en"] for x, y in zip(b, d))
        res["n"] = len(b)
        print("same_output", res["same_output"], "/", res["n"])
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)

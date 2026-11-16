"""T-58: ¿por qué Qwen 4B tarda ~140 ms más dentro del pipeline que solo?

El brazo bd0f324 mostró que el extra está dentro de ``mt.translate`` (mt_compute_ms ≈ gap). En el
pipeline, entre dos traducciones pasan (1) una síntesis de Kokoro en el mismo hilo MLX y (2) segundos
de GPU ociosa mientras suena el audio. El benchmark de MT sola corre traducción tras traducción.

Mide ``translate`` de los 3 textos asr_final en un solo proceso con Qwen + Kokoro cargados:
  hot        traducción tras traducción (= MT sola)
  kokoro     justo después de sintetizar la traducción anterior con Kokoro
  idle       después de IDLE_S segundos de GPU ociosa, sin Kokoro
  kokoro_idle  Kokoro y después IDLE_S de espera (lo más parecido al pipeline)
  idle_tick  IDLE_S de espera con una op MLX chica cada TICK_S (mantiene la GPU despierta)
  idle_kick  IDLE_S de espera y una op MLX chica KICK_S antes de traducir
  idle_spin  IDLE_S de CPU ocupada en el mismo hilo, sin GPU (¿es la CPU la que se duerme?)
  idle_prewarm  IDLE_S de espera y justo antes una generación corta (PREWARM_TOK tokens) del texto sin
             sus 3 últimas palabras (lo que el pipeline podría correr durante endpoint + ASR finish)
  idle_prewarm_short  como idle_prewarm pero 1 token sobre el prompt de "hola" (prefill mínimo)
  idle_bigtick  como idle_tick con una matmul 2048x2048 (GPU con carga real)
CONDS=hot,idle,... elige condiciones.

Uso (sin la medición de @bench corriendo):
  OPENTRADUCTOR_MLX_CACHE_MB=256 .venv/bin/python scripts/eval/t58_mt_gpu_probe.py OUT.json
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime import components as K  # noqa: E402
from runtime.config import PipelineConfig  # noqa: E402
from runtime.pipeline import _apply_mlx_cache_limit  # noqa: E402

REPS = int(os.environ.get("REPS", "7"))
IDLE_S = float(os.environ.get("IDLE_S", "2.0"))
TICK_S = float(os.environ.get("TICK_S", "0.1"))
PREWARM_TOK = int(os.environ.get("PREWARM_TOK", "4"))
KICK_S = float(os.environ.get("KICK_S", "0.15"))
QDIR = os.environ.get("OPENTRADUCTOR_QWEN_DIR", str(ROOT / "artifacts/analysis/llm-mp4-a/models/qwen3-4b-instruct-2507-4bit"))


def main(out: str) -> None:
    cfg = PipelineConfig()
    cap = _apply_mlx_cache_limit(cfg)
    if os.environ.get("WIRED") == "1":
        # mlx_lm.generate sube el wired limit y lo restaura al salir: entre frases los pesos quedan
        # sin fijar y tras GPU ociosa hay que volver a hacerlos residentes. Fijarlo una vez lo evita.
        import mlx.core as mx0
        mx0.set_wired_limit(mx0.device_info()["max_recommended_working_set_size"])
    texts = [json.loads(line)["text_es"] for line in (ROOT / "data/manifests/t58_asr_final_texts.jsonl").read_text().splitlines() if line.strip()]
    tts = K.KokoroTTS(cfg.voice, cfg.speed, cfg.tts_trim_lead_margin_ms, cfg.tts_first_chunk_max_words,
                      cfg.tts_bind_conjunctions, cfg.tts_trim_tail)
    mt = K.QwenMLX(Path(QDIR), normalize=cfg.mt_normalize)
    for t in texts * 2:  # warmup
        list(tts.synth(mt.translate(t)))

    res: dict[str, list[float]] = {}
    import mlx.core as mx

    def tick(n: int = 256) -> None:
        a = mx.ones((n, n))
        mx.eval(a @ a)

    order = os.environ.get("CONDS", "hot,kokoro,idle,kokoro_idle").split(",")
    for r in range(REPS):
        k = r % len(order)
        for cond in order[k:] + order[:k]:  # rota el orden para no sesgar por deriva térmica
            for t in texts:
                prev = mt.translate(t)  # deja el estado como tras una traducción
                if cond in ("kokoro", "kokoro_idle"):
                    list(tts.synth(prev))
                if cond in ("idle", "kokoro_idle"):
                    time.sleep(IDLE_S)
                elif cond in ("idle_tick", "idle_bigtick"):
                    end = time.perf_counter() + IDLE_S
                    while time.perf_counter() < end:
                        tick(2048 if cond == "idle_bigtick" else 256)
                        time.sleep(TICK_S)
                elif cond == "idle_prewarm":
                    from mlx_lm import generate
                    time.sleep(IDLE_S)
                    p0 = time.perf_counter()
                    generate(mt.model, mt.tok, prompt=mt.prompt(" ".join(t.split()[:-3])), max_tokens=PREWARM_TOK, verbose=False)
                    res.setdefault("prewarm_cost", []).append((time.perf_counter() - p0) * 1000)
                elif cond == "idle_prewarm_short":
                    # forward de 1 token con el prompt de "hola": toca todos los pesos con prefill mínimo
                    from mlx_lm import generate
                    time.sleep(IDLE_S)
                    p0 = time.perf_counter()
                    generate(mt.model, mt.tok, prompt=mt.prompt("hola"), max_tokens=1, verbose=False)
                    res.setdefault("prewarm_short_cost", []).append((time.perf_counter() - p0) * 1000)
                elif cond == "idle_spin":
                    end = time.perf_counter() + IDLE_S
                    while time.perf_counter() < end:
                        pass
                elif cond == "idle_kick":
                    time.sleep(max(0.0, IDLE_S - KICK_S))
                    tick()
                    time.sleep(KICK_S)
                t0 = time.perf_counter()
                mt.translate(t)
                ms = (time.perf_counter() - t0) * 1000
                res.setdefault(cond, []).append(ms)
                res.setdefault(f"{cond}|{t.split()[1]}", []).append(ms)
    summary = {c: {"p50": round(statistics.median(v), 1), "min": round(min(v), 1), "max": round(max(v), 1), "n": len(v)}
               for c, v in res.items()}
    doc = {"cache_limit_bytes": cap, "idle_s": IDLE_S, "reps": REPS, "qwen_dir": QDIR,
           "load": os.getloadavg(), "summary": summary, "raw": res}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(doc, indent=1))
    print(json.dumps({"cache": cap, **summary}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])

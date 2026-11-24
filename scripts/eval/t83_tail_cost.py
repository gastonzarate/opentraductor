"""T-83: costo del ASR (finish) según cuánta cola silenciosa recibe (asr_tail_keep_ms).

Alimenta Moonshine con la voz de un clip en chunks de 160 ms hasta el fin de habla y mide
add(cola de N ms) + finish para cada N, intercalado. Uso:
  .venv-inc-llm/bin/python scripts/eval/t83_tail_cost.py data/audio/t83/bh_1584.wav 6.9 [rondas]
(6.9 = fin de habla en s; sale de la traza del banco o del manifest).
"""
from __future__ import annotations

import random
import statistics
import sys
import time

import numpy as np
import soundfile as sf

from runtime.components import MoonshineStreamer
from runtime.config import MOONSHINE_MODELS, moonshine_model_dir

SR = 16000


def main() -> None:
    path, end_s = sys.argv[1], float(sys.argv[2])
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    x, sr = sf.read(path, dtype="float32")
    assert sr == SR
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    asr = MoonshineStreamer(moonshine_model_dir("moonshine-tiny"), MOONSHINE_MODELS["moonshine-tiny"][1], 0.3)
    end_b = int(end_s * SR) * 2
    ch = int(0.16 * SR) * 2
    keeps = [0, 32, 64, 96]
    res: dict[int, list[float]] = {k: [] for k in keeps}
    txt: dict[int, list[str]] = {k: [] for k in keeps}
    for _ in range(rounds):
        order = keeps[:]
        random.shuffle(order)
        for k in order:
            asr.start()
            for i in range(0, end_b, ch):
                asr.add(pcm[i:min(i + ch, end_b)])
            tail = pcm[end_b:end_b + int(k / 1000 * SR) * 2]
            t0 = time.perf_counter()
            if tail:
                asr.add(tail)
            t = asr.finish()
            res[k].append((time.perf_counter() - t0) * 1000)
            txt[k].append(t.strip().split()[-4:] and " ".join(t.strip().split()[-4:]))
    for k in keeps:
        print(f"keep{k:>3}: p50 {statistics.median(res[k]):6.1f} ms  min {min(res[k]):6.1f}  "
              f"finales {sorted(set(txt[k]))}")


if __name__ == "__main__":
    main()

"""Adaptador de evaluación SOLO TEXTO: MT (Marian CT2) y, opcional, TTS (Kokoro).

Reutiliza ``runtime.components.MarianCT2`` y ``KokoroTTS`` (código del runtime, sin
modificarlo). Entrada: ``text_es`` (sin audio). No mide streaming: la latencia es
petición → primer output (primer ``mt_final`` con ``stages="mt"``, primer PCM no
silencioso con ``stages="mt+tts"``) y se reporta en un grupo propio, nunca mezclada
con la de streaming.

Uso::

    python -m benchmark run --text-manifest data/manifests/text_meetings_ar_dev.jsonl \\
        --adapter benchmark.adapters.text_mt_tts:TextMtTtsAdapter \\
        --adapter-config '{"stages": "mt+tts"}' --split development --out artifacts/benchmark/text-mt-tts

Config: ``stages`` ("mt" | "mt+tts"), ``voice``, ``speed``, ``beam_size``, ``threads``.

Hilos: MLX liga arrays/streams al hilo que los crea; Kokoro se carga y se usa en un
mismo hilo dedicado (como ``runtime.pipeline.ModelWorker``). La MT corre en otro.
No descarga nada: solo disco (modo offline forzado).
"""
from __future__ import annotations

import asyncio
import concurrent.futures as cf
import hashlib
import importlib.metadata as md
from pathlib import Path
from typing import Any, AsyncIterator

from .. import snapshot
from ..contracts import Adapter, AdapterEvent, AdapterMode, EventKind

_DONE = object()
ENGINE_PACKAGES = ("ctranslate2", "sentencepiece", "mlx", "mlx-audio", "misaki", "huggingface-hub", "numpy")


class _Worker:
    """Un hilo dedicado."""

    def __init__(self, name: str) -> None:
        self._ex = cf.ThreadPoolExecutor(1, thread_name_prefix=name)

    async def run(self, fn, *a):
        return await asyncio.get_running_loop().run_in_executor(self._ex, fn, *a)

    def shutdown(self) -> None:
        self._ex.shutdown(wait=True, cancel_futures=True)


class TextMtTtsAdapter(Adapter):
    name = "text-marian-ct2"
    version = "0.1"
    mode = AdapterMode.TEXT
    produces_audio = False
    quiescence_ms = 100.0  # process_text espera a encolar todo; solo falta drenar el colector

    def __init__(self) -> None:
        self._q: asyncio.Queue = asyncio.Queue()
        self.cfg: dict[str, Any] = {}
        self.mt = None
        self.tts = None
        self._cpu = _Worker("bench-mt")
        self._gpu = _Worker("bench-tts")
        self._n = 0
        self.load_ms: dict[str, float] = {}

    # ---- fábricas (sobrescribibles en pruebas) ----
    def _make_mt(self, cfg: dict):
        from runtime.components import MarianCT2  # import perezoso: el benchmark no depende del runtime

        return MarianCT2(beam_size=cfg["beam_size"], threads=cfg["threads"])

    def _make_tts(self, cfg: dict):
        from runtime.components import KokoroTTS

        return KokoroTTS(cfg["voice"], cfg["speed"])

    @staticmethod
    def _split(text: str) -> list[str]:
        from runtime.components import split_clauses

        return split_clauses(text)

    async def open(self, config: dict[str, Any]) -> None:
        import os
        import time

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        self.cfg = {"stages": "mt", "voice": "af_heart", "speed": 1.0, "beam_size": 1, "threads": 4, **config}
        if self.cfg["stages"] not in ("mt", "mt+tts"):
            raise ValueError(f"stages inválido: {self.cfg['stages']!r}")
        self.produces_audio = self.cfg["stages"] == "mt+tts"
        self.name = "text-marian-ct2" + ("+kokoro" if self.produces_audio else "")

        def timed(label, fn):
            t = time.perf_counter()
            r = fn()
            self.load_ms[label] = round((time.perf_counter() - t) * 1000, 1)
            return r

        self.mt = await self._cpu.run(lambda: timed("mt", lambda: self._make_mt(self.cfg)))
        if self.produces_audio:
            self.tts = await self._gpu.run(lambda: timed("tts", lambda: self._make_tts(self.cfg)))

    async def start_stream(self, stream_id: str, seed: int) -> None:
        pass  # sin estado por stream

    async def process_text(self, stream_id: str, text_es: str) -> None:
        text_en = await self._cpu.run(self.mt.translate, text_es)
        self._n += 1
        await self._q.put(AdapterEvent(EventKind.MT_FINAL, stream_id, text=text_en, commit_id=f"{stream_id}#mt{self._n}"))
        if not self.produces_audio or not text_en.strip():
            return
        for k, clause in enumerate(self._split(text_en)):
            gen = self.tts.synth(clause)
            cid = f"{stream_id}#c{self._n}.{k}"  # un commit por cláusula (frase repetida = otro commit)
            while True:
                chunk = await self._gpu.run(lambda: next(gen, None))
                if chunk is None:
                    break
                await self._q.put(AdapterEvent(
                    EventKind.TTS_CHUNK, stream_id, pcm=chunk.pcm, sample_rate=chunk.sample_rate,
                    planned_text_en=clause, commit_id=cid,
                ))

    async def events(self) -> AsyncIterator[AdapterEvent]:
        while True:
            item = await self._q.get()
            if item is _DONE:
                return
            yield item

    async def close(self) -> None:
        self._cpu.shutdown()
        self._gpu.shutdown()
        await self._q.put(_DONE)

    # ---- reproducibilidad ----
    def _describe_tts(self) -> dict[str, Any]:
        from runtime import config as C

        kdir = C.snapshot(C.KOKORO_REPO, C.KOKORO_REV, ("config.json",))
        return {"repo": C.KOKORO_REPO, "revision": C.KOKORO_REV, "voice": self.cfg["voice"],
                "weights": snapshot.weights_fingerprint(kdir)}

    def describe(self) -> dict[str, Any]:
        d: dict[str, Any] = {"stages": self.cfg.get("stages"), "params": dict(self.cfg), "load_ms": self.load_ms}
        d["engine_versions"] = {}
        for p in ENGINE_PACKAGES:
            try:
                d["engine_versions"][p] = md.version(p)
            except md.PackageNotFoundError:
                d["engine_versions"][p] = None
        try:
            from runtime import components as K
            from runtime import config as C

            d["runtime_components_sha256"] = hashlib.sha256(Path(K.__file__).read_bytes()).hexdigest()
            d["adapter_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            models: dict[str, Any] = {}
            marian = C.marian_ct2_dir()
            models["mt"] = {"repo": C.MARIAN_REPO, "revision": C.MARIAN_REV, "ct2_int8_dir": snapshot.weights_fingerprint(marian)}
            if self.produces_audio:
                models["tts"] = self._describe_tts()
            d["models"] = models
        except Exception as exc:  # no tumbar la corrida: dejar constancia
            d["describe_error"] = f"{type(exc).__name__}: {exc}"
        return d

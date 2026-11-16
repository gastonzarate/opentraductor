"""Adaptador del benchmark para la baseline local Parakeet + Marian CT2 + Kokoro.

Uso: ``--adapter runtime.adapters.baseline:BaselineAdapter``. Configuración en
``--adapter-config`` (JSON, ver ``runtime.config.PipelineConfig``).
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from benchmark.contracts import Adapter, AdapterEvent, AdapterMode, EventKind

from ..config import PipelineConfig
from ..pipeline import Engine

_DONE = object()
_KIND = {
    "asr_partial": EventKind.ASR_PARTIAL,
    "asr_final": EventKind.ASR_FINAL,
    "endpoint": EventKind.ENDPOINT,
    "mt_final": EventKind.MT_FINAL,
    "tts_chunk": EventKind.TTS_CHUNK,
    "error": EventKind.ERROR,
    "overflow": EventKind.ERROR,
}


class BaselineAdapter(Adapter):
    name = "baseline-parakeet-marian-kokoro"
    version = "0.1"
    mode = AdapterMode.STREAMING

    def __init__(self) -> None:
        self._q: asyncio.Queue = asyncio.Queue()
        self.engine: Engine | None = None
        # ASR crudo vs texto usado como entrada de MT (el contrato AdapterEvent solo
        # lleva ``text``): queda en ``describe()['asr_text_log']`` -> run.json.
        self.asr_text_log: list[dict[str, Any]] = []
        # T-58: mt_ms (desde que el speaker loop toma el segmento) vs mt_compute_ms (dentro del hilo MT).
        self.mt_timing_log: list[dict[str, Any]] = []

    async def _emit(self, kind: str, f: dict) -> None:
        if kind == "asr_final" and len(self.asr_text_log) < 5000:
            self.asr_text_log.append({
                "stream_id": f["stream_id"],
                "commit_id": f"{f['stream_id']}:{f['seg_id']}" if f.get("seg_id") is not None else None,
                "asr_raw": f.get("text_raw", f.get("text")),
                "mt_input": f.get("text"),
            })
        if kind == "mt_final" and len(self.mt_timing_log) < 5000:
            self.mt_timing_log.append({
                "commit_id": f"{f['stream_id']}:{f['seg_id']}" if f.get("seg_id") is not None else None,
                "mt_ms": f.get("mt_ms"), "mt_compute_ms": f.get("mt_compute_ms"), "spec": f.get("spec"),
            })
        ev = AdapterEvent(
            _KIND[kind],
            f["stream_id"],
            text=f.get("text"),
            audio_pos_s=f.get("audio_pos_s"),
            pcm=f.get("pcm"),
            sample_rate=f.get("sample_rate"),
            planned_text_en=f.get("planned_text_en"),
            source_span_s=f.get("source_span_s"),
            error_kind=f.get("error_kind") or ("overflow" if kind == "overflow" else None),
            commit_id=(f"{f['stream_id']}:{f['seg_id']}" if f.get("seg_id") is not None else None),
        )
        await self._q.put(ev)

    def describe(self) -> dict[str, Any]:
        from .. import fingerprint

        cfg = self.engine.cfg if self.engine else PipelineConfig()
        return {
            "adapter_code_sha256": fingerprint.code_hash(),
            "weights": fingerprint.weights(cfg),
            "engines": fingerprint.engines(),
            "pipeline_config": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
            "load_ms": dict(self.engine.load_ms) if self.engine else {},
            "asr_text_log": list(self.asr_text_log),
            "mt_timing_log": list(self.mt_timing_log),
            "speculation": self.engine.spec_metrics() if self.engine else {},
            "mt_prewarm": dict(self.engine.prewarm_stats, enabled=self.engine._prewarm_enabled) if self.engine else {},
        }

    @staticmethod
    def name_for(cfg: PipelineConfig) -> str:
        """Nombre según el ASR real (el runner lo lee después de open(): meta y system_name)."""
        asr = "parakeet" if cfg.asr_backend == "parakeet" else cfg.asr_backend
        return f"baseline-{asr}-marian-kokoro"

    async def open(self, config: dict[str, Any]) -> None:
        cfg = PipelineConfig.from_dict(config)
        self.name = self.name_for(cfg)  # instancia; el valor de clase queda para corridas históricas
        self.engine = Engine(cfg, self._emit)
        await self.engine.open()

    async def start_stream(self, stream_id: str, seed: int) -> None:
        await self.engine.start_stream(stream_id)

    async def push_audio(self, stream_id, pcm, t_audio_start_s) -> None:
        await self.engine.push(stream_id, pcm, t_audio_start_s)

    async def end_stream(self, stream_id: str) -> None:
        await self.engine.end_stream(stream_id)

    async def events(self) -> AsyncIterator[AdapterEvent]:
        while True:
            item = await self._q.get()
            if item is _DONE:
                return
            yield item

    async def close(self) -> None:
        if self.engine:
            await self.engine.close()
        await self._q.put(_DONE)

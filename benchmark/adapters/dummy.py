"""Adaptador DUMMY. SOLO prueba de infraestructura. NO es un benchmark real.

No traduce nada: detecta fin de energía con un VAD ingenuo y emite un tono
sintético tras un retardo configurable. Sus números no representan ningún
candidato real y los informes los etiquetan como ``infra_test_only``.
"""
from __future__ import annotations

import asyncio
import hashlib
from typing import Any, AsyncIterator

from .. import audio as A
from ..contracts import Adapter, AdapterEvent, AdapterMode, EventKind

_DONE = object()


class DummyAdapter(Adapter):
    name = "dummy-infra-test"
    version = "0"
    mode = AdapterMode.STREAMING
    infra_test_only = True

    def __init__(self, mode: AdapterMode = AdapterMode.STREAMING) -> None:
        self.mode = mode
        self.cfg: dict[str, Any] = {}
        self._q: asyncio.Queue = asyncio.Queue()
        # Registro para pruebas de fuga temporal:
        self.push_log: list[dict] = []
        self.seen_audio: bytearray = bytearray()
        self.open_called = False
        self._state: dict[str, dict] = {}
        self._tasks: set[asyncio.Task] = set()
        self._commit_n = 0

    async def open(self, config: dict[str, Any]) -> None:
        self.cfg = {
            "endpoint_silence_ms": 400.0,
            "processing_ms": 100.0,
            "tone_ms": 500.0,
            "chunk_ms": 100.0,  # tamaño de cada chunk TTS emitido
            "behavior": "normal",  # normal|anticipate|silent|never|error
            "anticipate_at_s": 0.5,
            "text_en": "hello world",
            "emit_span": False,
            "emit_commit_id": True,
            "asr_text": "(dummy)",
            "vad_threshold_dbfs": -45.0,
            "open_delay_ms": 0.0,
            **config,
        }
        if self.cfg["open_delay_ms"]:
            await asyncio.sleep(self.cfg["open_delay_ms"] / 1000)
        self.open_called = True

    async def start_stream(self, stream_id: str, seed: int) -> None:
        self._state[stream_id] = {
            "in_speech": False,
            "speech_start": None,
            "last_speech_end": None,
            "silence_ms": 0.0,
            "anticipated": False,
        }

    async def push_audio(self, stream_id, pcm, t_audio_start_s):
        loop = asyncio.get_running_loop()
        dur = len(pcm) / 2 / A.TARGET_SR
        self.push_log.append(
            {
                "stream_id": stream_id,
                "t_audio_start_s": t_audio_start_s,
                "t_audio_end_s": t_audio_start_s + dur,
                "loop_time_s": loop.time(),
                "sha256": hashlib.sha256(pcm).hexdigest(),
            }
        )
        self.seen_audio.extend(pcm)
        st = self._state[stream_id]
        beh = self.cfg["behavior"]
        loud = A.rms_dbfs(pcm) > self.cfg["vad_threshold_dbfs"]
        end_s = t_audio_start_s + dur
        if (
            beh == "anticipate"
            and not st["anticipated"]
            and end_s >= self.cfg["anticipate_at_s"]
        ):
            st["anticipated"] = True
            self._spawn(self._emit_output(stream_id, 0.0, (0.0, end_s)))
            return
        if loud:
            if not st["in_speech"]:
                st["in_speech"] = True
                st["speech_start"] = t_audio_start_s
            st["last_speech_end"] = end_s
            st["silence_ms"] = 0.0
        elif st["in_speech"]:
            st["silence_ms"] += dur * 1000
            if st["silence_ms"] >= self.cfg["endpoint_silence_ms"]:
                st["in_speech"] = False
                await self._q.put(
                    AdapterEvent(EventKind.ENDPOINT, stream_id, audio_pos_s=end_s)
                )
                if beh == "error":
                    await self._q.put(
                        AdapterEvent(EventKind.ERROR, stream_id, error_kind="dummy_error")
                    )
                elif beh not in ("never",):
                    span = (st["speech_start"], st["last_speech_end"])
                    self._spawn(
                        self._emit_output(
                            stream_id, self.cfg["processing_ms"] / 1000, span
                        )
                    )

    def _spawn(self, coro) -> None:
        t = asyncio.ensure_future(coro)
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    async def _emit_output(self, stream_id: str, delay_s: float, span) -> None:
        if delay_s:
            await asyncio.sleep(delay_s)
        text = self.cfg["text_en"]
        self._commit_n += 1
        cid = f"{stream_id}#c{self._commit_n}" if self.cfg["emit_commit_id"] else None
        asr_span = span if self.cfg["emit_span"] else None
        if self.cfg["asr_text"] is not None:  # None = el candidato no emite asr_final
            await self._q.put(AdapterEvent(EventKind.ASR_FINAL, stream_id, text=self.cfg["asr_text"],
                                           commit_id=cid, source_span_s=asr_span))
        await self._q.put(AdapterEvent(EventKind.MT_FINAL, stream_id, text=text, commit_id=cid))
        total = self.cfg["tone_ms"] / 1000
        step = self.cfg["chunk_ms"] / 1000
        done = 0.0
        while done < total - 1e-9:
            d = min(step, total - done)
            pcm = A.silence(d) if self.cfg["behavior"] == "silent" else A.tone(d)
            await self._q.put(
                AdapterEvent(
                    EventKind.TTS_CHUNK,
                    stream_id,
                    pcm=pcm,
                    sample_rate=A.TARGET_SR,
                    planned_text_en=text,
                    source_span_s=span if self.cfg["emit_span"] else None,
                    commit_id=cid,
                    claimed_ns=0,  # valor falso a propósito: el arnés lo ignora
                )
            )
            done += d
            # Mantener ritmo de síntesis en tiempo real entre chunks.
            if done < total - 1e-9:
                await asyncio.sleep(d)

    async def process_file(self, stream_id: str, pcm: bytes) -> None:
        await asyncio.sleep(self.cfg["processing_ms"] / 1000)
        await self._emit_output(stream_id, 0.0, None)

    async def process_text(self, stream_id: str, text_es: str) -> None:
        await asyncio.sleep(self.cfg["processing_ms"] / 1000)
        await self._emit_output(stream_id, 0.0, None)

    async def events(self) -> AsyncIterator[AdapterEvent]:
        while True:
            item = await self._q.get()
            if item is _DONE:
                return
            yield item

    async def close(self) -> None:
        for t in list(self._tasks):
            t.cancel()
        await self._q.put(_DONE)

"""Watchdog de proceso para inferencia síncrona que bloquea el event loop.

``asyncio.wait_for`` no puede interrumpir código síncrono (p. ej. un modelo que
llama a ``time.sleep`` o a una inferencia larga dentro de ``push_audio``). Un
hilo vigila un latido que el loop actualiza; si el loop no late durante
``stall_limit_s``, escribe ``FAILURE.json`` (contexto + resultados parciales) y
termina el proceso con código 70. El fallo queda registrado, no se pierde.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

EXIT_CODE_STALL = 70


class Watchdog:
    def __init__(
        self,
        stall_limit_s: float,
        failure_path: str | Path | None,
        context_fn: Callable[[], dict[str, Any]] | None = None,
        *,
        exit_fn: Callable[[int], None] = os._exit,
        poll_s: float = 0.1,
    ) -> None:
        self.stall_limit_s = stall_limit_s
        self.failure_path = Path(failure_path) if failure_path else None
        self.context_fn = context_fn
        self._exit = exit_fn
        self._poll = poll_s
        self._last = time.monotonic()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle: asyncio.TimerHandle | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self.tripped = False

    def _beat(self) -> None:
        self._last = time.monotonic()
        if not self._stop.is_set() and self._loop is not None:
            self._handle = self._loop.call_later(self._poll, self._beat)

    def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._last = time.monotonic()
        self._beat()
        self._thread = threading.Thread(target=self._watch, daemon=True, name="bench-watchdog")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._handle is not None:
            self._handle.cancel()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _watch(self) -> None:
        while not self._stop.wait(self._poll):
            stalled = time.monotonic() - self._last
            if stalled > self.stall_limit_s:
                self.tripped = True
                rec: dict[str, Any] = {
                    "failure": "event_loop_stalled",
                    "stalled_s": stalled,
                    "stall_limit_s": self.stall_limit_s,
                    "note": "el candidato bloqueó el event loop (inferencia síncrona); proceso terminado por el watchdog",
                }
                try:
                    if self.context_fn:
                        rec["context"] = self.context_fn()
                except Exception as exc:  # no impedir la salida
                    rec["context_error"] = repr(exc)
                if self.failure_path:
                    try:
                        self.failure_path.parent.mkdir(parents=True, exist_ok=True)
                        self.failure_path.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
                    except Exception:
                        pass
                self._exit(EXIT_CODE_STALL)
                return

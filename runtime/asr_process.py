"""T-67: Moonshine en un proceso aparte, fuera del GIL del motor.

Sonda scripts/eval/t67_asr_tail_probe.py (20/11, artifacts/t67-asrtail): solo, el stop() de Moonshine tarda
13 ms p50 y el parcial del endpoint ya es el final (5/5). Con un hilo Python ocupando el GIL (como el decode de
Qwen o Kokoro en el motor) sube a 412 ms p50. En el pipeline mide ~200 ms p50 (finish_ms de t67-asrfin) y el
parcial pierde 1-4 palabras (asr_tail eq 0/16). Con el ASR en otro proceso no comparte GIL.

Misma interfaz que MoonshineStreamer (start/add/finish/close). Cada llamada es un ida y vuelta por Pipe
(bytes de PCM y texto). Solo código local del paquete, sin red. Flag: OPENTRADUCTOR_ASR_PROCESS=1.
"""
from __future__ import annotations

import multiprocessing as mp
import os


def asr_process_enabled(cfg_value: bool) -> bool:
    v = os.environ.get("OPENTRADUCTOR_ASR_PROCESS")
    return (v == "1") if v in ("0", "1") else bool(cfg_value)


def _serve(conn, model_dir: str, arch: str, update_interval: float) -> None:
    from .components import MoonshineStreamer

    try:
        asr = MoonshineStreamer(model_dir, arch, update_interval)
        conn.send(("ok", {"ort_provider": asr.ort_provider, "provider_error": asr.provider_error}))
    except Exception as exc:  # noqa: BLE001
        conn.send(("err", f"{type(exc).__name__}: {exc}"))
        return
    while True:
        try:
            op, arg = conn.recv()
        except EOFError:
            break
        try:
            if op == "add":
                conn.send(("ok", asr.add(arg)))
            elif op == "start":
                asr.start()
                conn.send(("ok", None))
            elif op == "finish":
                conn.send(("ok", asr.finish()))
            elif op == "close":
                asr.close()
                conn.send(("ok", None))
                break
            else:
                conn.send(("err", f"op desconocida: {op}"))
        except Exception as exc:  # noqa: BLE001
            conn.send(("err", f"{type(exc).__name__}: {exc}"))
    conn.close()


class ProcessMoonshineStreamer:
    """Proxy de MoonshineStreamer que corre en un proceso hijo (spawn). Llamar desde un solo hilo (asr_worker)."""

    def __init__(self, model_dir, arch: str, update_interval: float) -> None:
        ctx = mp.get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=_serve, args=(child, str(model_dir), arch, update_interval),
                                 name="opentraductor-asr", daemon=True)
        self._proc.start()
        child.close()
        info = self._recv()
        self.ort_provider = info.get("ort_provider")
        self.provider_error = info.get("provider_error")
        self.last_latency_ms = 0

    def _recv(self):
        kind, val = self._conn.recv()
        if kind == "err":
            raise RuntimeError(f"asr_process: {val}")
        return val

    def _call(self, op: str, arg=None):
        self._conn.send((op, arg))
        return self._recv()

    def start(self) -> None:
        self._call("start")

    def add(self, pcm: bytes) -> str:
        return self._call("add", bytes(pcm))

    def finish(self) -> str:
        return self._call("finish")

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.is_alive():
                self._call("close")
        except (EOFError, OSError, RuntimeError):
            pass
        finally:
            self._conn.close()
            self._proc.join(timeout=2)
            if self._proc.is_alive():
                self._proc.kill()
            self._proc = None

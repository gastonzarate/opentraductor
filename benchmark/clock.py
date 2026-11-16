"""Relojes del arnés.

Todo instante del benchmark sale de un ``Clock`` del arnés. Nunca se usan los
timestamps que declare un candidato.

* ``RealClock``: ``time.monotonic_ns`` + ``asyncio`` real. Para mediciones reales.
* ``VirtualClock``: tiempo virtual determinista sobre un event loop propio
  (``VirtualTimeLoop``). Solo para pruebas de infraestructura (reloj, fuga
  temporal). Las latencias obtenidas con él NO son mediciones de hardware.
"""
from __future__ import annotations

import asyncio
import selectors
import time
from typing import Protocol


class Clock(Protocol):
    kind: str  # "real" | "virtual"

    def now_ns(self) -> int: ...

    async def sleep_until_ns(self, t_ns: int) -> None: ...


class RealClock:
    kind = "real"

    def now_ns(self) -> int:
        return time.monotonic_ns()

    async def sleep_until_ns(self, t_ns: int) -> None:
        delta = t_ns - self.now_ns()
        if delta > 0:
            # asyncio.sleep puede dormir de más; el arnés registra la tardanza
            # de envío real (ver replay.SendRecord.lateness_ns).
            await asyncio.sleep(delta / 1e9)
        else:
            await asyncio.sleep(0)


class _VirtualSelector(selectors.BaseSelector):
    """Selector que avanza el tiempo virtual en vez de dormir."""

    def __init__(self, loop_ref: "VirtualTimeLoop") -> None:
        self._inner = selectors.DefaultSelector()
        self._loop = loop_ref

    def register(self, fileobj, events, data=None):
        return self._inner.register(fileobj, events, data)

    def unregister(self, fileobj):
        return self._inner.unregister(fileobj)

    def modify(self, fileobj, events, data=None):
        return self._inner.modify(fileobj, events, data)

    def select(self, timeout=None):
        ready = self._inner.select(0)
        if ready:
            return ready
        if timeout is not None and timeout > 0:
            self._loop._virtual_time += timeout
        return []

    def close(self):
        self._inner.close()

    def get_map(self):
        return self._inner.get_map()


class VirtualTimeLoop(asyncio.SelectorEventLoop):
    """Event loop con tiempo virtual: ``asyncio.sleep(1)`` tarda 0 s reales."""

    def __init__(self) -> None:
        self._virtual_time = 0.0
        super().__init__(selector=_VirtualSelector(self))

    def time(self) -> float:  # noqa: D401
        return self._virtual_time


class VirtualClock:
    kind = "virtual"

    def __init__(self) -> None:
        self._loop: VirtualTimeLoop | None = None

    def _get_loop(self) -> VirtualTimeLoop:
        loop = asyncio.get_running_loop()
        if not isinstance(loop, VirtualTimeLoop):
            raise RuntimeError("VirtualClock requiere run_virtual()/VirtualTimeLoop")
        return loop

    def now_ns(self) -> int:
        return round(self._get_loop().time() * 1e9)

    async def sleep_until_ns(self, t_ns: int) -> None:
        delta = t_ns - self.now_ns()
        await asyncio.sleep(max(delta, 0) / 1e9)


def run_virtual(coro_factory):
    """Ejecuta ``coro_factory(clock)`` en un loop de tiempo virtual."""
    clock = VirtualClock()
    with asyncio.Runner(loop_factory=VirtualTimeLoop) as runner:
        return runner.run(coro_factory(clock))

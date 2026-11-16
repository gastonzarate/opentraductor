"""Detección del micrófono virtual (BlackHole 2ch) y guía de instalación.

Solo lee la lista de dispositivos que ya enumeró el servicio. No instala ni modifica nada del sistema:
BlackHole es un driver de terceros (GPL-3.0) que instala la persona con su contraseña de administrador.
"""
from __future__ import annotations

from typing import Any

BLACKHOLE_NAME = "BlackHole 2ch"
DOWNLOAD_URL = "https://existential.audio/blackhole/"
BREW_COMMAND = "brew install blackhole-2ch"

STEPS = [
    {
        "n": 1,
        "title": "Instalá BlackHole 2ch",
        "detail": (
            "Descargalo gratis desde existential.audio/blackhole (pedido de correo, instalador .pkg) "
            "o, si usás Homebrew, ejecutá el comando de abajo en Terminal. Te pide tu contraseña de administrador."
        ),
        "command": BREW_COMMAND,
        "url": DOWNLOAD_URL,
    },
    {
        "n": 2,
        "title": "Reiniciá opentraductor",
        "detail": (
            "Cerrá la app por completo (⌘Q) y volvé a abrirla. Si el driver no aparece, reiniciá la Mac: "
            "macOS carga los drivers de audio al arrancar."
        ),
    },
    {
        "n": 3,
        "title": "Elegí BlackHole 2ch en tu reunión",
        "detail": (
            "En Meet, Zoom o Discord, abrí la configuración de audio y elegí «BlackHole 2ch» como micrófono. "
            "En opentraductor, la salida de traducción es BlackHole 2ch."
        ),
    },
]


def _is_blackhole_2ch(name: str) -> bool:
    return "blackhole" in name.lower() and "2ch" in name.lower()


def detect(devices: dict[str, Any]) -> dict[str, Any]:
    """``devices``: el resultado de ``service.devices()``. Devuelve el estado para la pantalla «Micrófono virtual»."""
    outs = [d for d in (devices.get("outputs") or []) if isinstance(d, dict)]
    ins = [d for d in (devices.get("inputs") or []) if isinstance(d, dict)]
    out = next((d for d in outs if _is_blackhole_2ch(str(d.get("name", "")))), None)
    inp = next((d for d in ins if _is_blackhole_2ch(str(d.get("name", "")))), None)
    installed = out is not None and inp is not None
    return {
        "installed": installed,
        "device_name": BLACKHOLE_NAME,
        "output_id": out.get("id") if out else None,  # id para configurar la salida de la app
        "audio_ok": bool(devices.get("audio_ok", True)),
        "steps": [] if installed else STEPS,
        "download_url": DOWNLOAD_URL,
        "note": (
            "opentraductor no instala ni modifica drivers. BlackHole es un proyecto de terceros bajo GPL-3.0."
        ),
    }

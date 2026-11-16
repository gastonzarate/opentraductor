"""Comprobación previa: todo lo necesario está en disco. Nunca descarga.

Fuerza modo offline para que ninguna librería intente red en ejecución
(Hugging Face, spaCy/misaki).
"""
from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass

from . import config as C


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def force_offline() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def run_checks(cfg: C.PipelineConfig) -> list[Check]:
    out: list[Check] = []

    def chk(name, fn):
        try:
            out.append(Check(name, True, str(fn())))
        except Exception as exc:  # noqa: BLE001
            out.append(Check(name, False, f"{type(exc).__name__}: {exc}"))

    if cfg.asr_backend == "parakeet":
        chk("asr:parakeet", lambda: C.snapshot(C.PARAKEET_REPO, C.PARAKEET_REV, ("config.json", "model.safetensors")))
    else:
        def moon():
            d = C.moonshine_model_dir(cfg.asr_backend)
            need = ("encoder.ort", "decoder_kv.ort", "tokenizer.bin", "streaming_config.json")
            miss = [f for f in need if not (d / f).exists()]
            if miss:
                raise FileNotFoundError(f"faltan {miss} en {d}")
            if importlib.util.find_spec("moonshine_voice") is None:
                raise ModuleNotFoundError("moonshine_voice no instalado (moonshine-voice==0.1.6)")
            return d

        chk(f"asr:{cfg.asr_backend}", moon)
    chk(
        "tts:kokoro",
        lambda: C.snapshot(
            C.KOKORO_REPO, C.KOKORO_REV,
            ("config.json", "kokoro-v1_0.safetensors", f"voices/{cfg.voice}.safetensors"),
        ),
    )

    def mt():
        d = C.marian_ct2_dir()
        miss = [f for f in ("model.bin", "source.spm", "target.spm") if not (d / f).exists()]
        if miss:
            raise FileNotFoundError(f"faltan {miss} en {d}")
        return d

    def qwen():
        d = C.qwen_model_dir(cfg)
        miss = [f for f in ("config.json", "model.safetensors", "tokenizer.json") if not (d / f).exists()]
        if miss:
            raise FileNotFoundError(f"faltan {miss} en {d} (no se descarga aquí)")
        if importlib.util.find_spec("mlx_lm") is None:
            raise ModuleNotFoundError("mlx_lm no instalado (mlx-lm==0.32.2)")
        return d

    if cfg.mt_engine == "qwen":
        chk("mt:qwen-mlx", qwen)
    else:
        chk("mt:marian-ct2", mt)

    def spacy_model():
        if importlib.util.find_spec("en_core_web_sm") is None:
            raise ModuleNotFoundError(
                "en_core_web_sm no instalado (misaki intentaría descargarlo). "
                "Instala con: uv sync --extra runtime"
            )
        return "en_core_web_sm"

    chk("g2p:en_core_web_sm", spacy_model)
    def vad():
        if importlib.util.find_spec("onnxruntime") is None:
            raise ModuleNotFoundError("onnxruntime no instalado")
        p = C.silero_vad_onnx_path()
        if not p.exists():
            raise FileNotFoundError(f"falta {p} (Silero VAD ONNX)")
        return p

    chk("vad:silero-onnx", vad)
    return out


def require_ready(cfg: C.PipelineConfig) -> None:
    force_offline()
    bad = [c for c in run_checks(cfg) if not c.ok]
    if bad:
        raise RuntimeError(
            "Preflight falló: " + "; ".join(f"{c.name} -> {c.detail}" for c in bad)
        )

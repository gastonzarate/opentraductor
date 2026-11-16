"""Adaptador EXPERIMENTAL de evaluación SOLO TEXTO: MT (Marian CT2) + TTS Qwen3 con clonación.

Aislado: no cambia el TTS por defecto del runtime (Kokoro) ni toca ``runtime/``. Reutiliza
``TextMtTtsAdapter`` y sustituye solo el TTS. No afirma velocidad ni calidad: solo mide
(primer PCM disponible/audible, producción continua) cuando se ejecute con una ventana
de cómputo autorizada.

Pesos: ``mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit`` fijado por revisión, solo desde
disco (``runtime.config.snapshot``; nunca descarga).

Referencia de voz: archivo público con licencia y consentimiento documentados en un JSON
adyacente (``data/audio/examples/kyutai-0a67.wav`` + ``.json``, CC0, voluntario; NO es voz
del usuario). Se verifica el sha256 declarado y se rechaza una referencia sin ``license``.

``ref_text`` (transcripción de la referencia) es configurable:
  - ``ref_text`` con texto: la API usa clonación ICL (audio + transcripción).
  - ``ref_text`` = ``null`` (hoy no hay transcripción verificada): la API NO usa ICL y cae a la
    ruta Base con solo el embedding de hablante; es otro modo, con otra calidad esperable.
    ``describe()`` registra ``clone_mode`` (``icl`` | ``speaker_embedding_only``) para que no se
    mezclen en el análisis. La transcripción no se inventa ni se infiere con un modelo aquí.

Config (además de las de ``TextMtTtsAdapter``): ``ref_audio``, ``ref_text``, ``ref_meta`` (por
defecto ``<ref_audio>.json``), ``lang_code`` ("english"), ``streaming_interval`` (s, por
defecto 0.5; pequeño = primer PCM antes, más sobrecoste), ``streaming_context_size``,
``temperature``, ``top_k``, ``top_p``, ``max_tokens``, ``seed`` (MLX ``mx.random.seed`` antes
de cada síntesis), ``split_clauses`` (False: una síntesis por enunciado; True: por cláusula,
con prefill de la referencia repetido).

Hilos: carga y uso en el hilo dedicado de TTS (arrays MLX ligados al hilo).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .. import snapshot
from .text_mt_tts import TextMtTtsAdapter

QWEN3_REPO = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit"
QWEN3_REV = "50f45ef0047cde7e84c2ef04326acb8ada2436a7"
DEFAULT_REF_AUDIO = "data/audio/examples/kyutai-0a67.wav"

DEFAULTS: dict[str, Any] = {
    "stages": "mt+tts",
    "ref_audio": DEFAULT_REF_AUDIO,
    "ref_text": None,
    "ref_meta": None,
    "lang_code": "english",
    "streaming_interval": 0.5,
    "streaming_context_size": 25,
    "temperature": None,
    "top_k": None,
    "top_p": None,
    "max_tokens": None,
    "seed": None,
    "split_clauses": False,
    # Condicionamiento POR ÍTEM (piloto interlingüe): JSON con items[].{audio_path, sha256, transcript_es,
    # transcript_es_provenance, license}. Cada texto de entrada (text_es == transcript_es) se condiciona con SU
    # propio audio original en español y su transcripción humana. Sustituye a ref_audio/ref_text fijos.
    "conditioning_file": None,
}


def load_reference(ref_audio: str, ref_meta: str | None = None) -> dict[str, Any]:
    """Verifica la referencia pública: existe, hash coincide y licencia/consentimiento declarados."""
    audio = Path(ref_audio)
    if not audio.is_file():
        raise FileNotFoundError(f"referencia de voz ausente: {audio}")
    meta_path = Path(ref_meta) if ref_meta else audio.with_suffix(".json")
    if not meta_path.is_file():
        raise FileNotFoundError(f"falta metadato de licencia de la referencia: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for k in ("license", "consent_source", "sha256"):
        if not meta.get(k):
            raise ValueError(f"metadato de referencia sin {k!r}: {meta_path}")
    sha = hashlib.sha256(audio.read_bytes()).hexdigest()
    if sha != meta["sha256"]:
        raise ValueError(f"sha256 de la referencia no coincide con {meta_path.name}")
    return {"path": str(audio), "sha256": sha, "license": meta["license"], "consent_source": meta["consent_source"],
            "scope": meta.get("scope"), "transcript_provenance": meta.get("transcript_provenance"),
            "duration_s": meta.get("duration_s"), "note": meta.get("note")}


def load_conditioning(path: str, root: str | Path = ".") -> dict[str, dict[str, Any]]:
    """Mapa texto_es → referencia por ítem. Exige transcripción humana, licencia, sha256 válido y textos únicos.
    No infiere nada con ASR; si falta algo, falla."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for it in doc["items"]:
        if it.get("transcript_es_provenance") != "human":
            raise ValueError(f"{it.get('item_id')}: la transcripción de condicionamiento debe ser humana")
        for k in ("audio_path", "sha256", "transcript_es", "license"):
            if not it.get(k):
                raise ValueError(f"{it.get('item_id')}: falta {k!r}")
        audio = Path(root) / it["audio_path"]
        if not audio.is_file():
            raise FileNotFoundError(f"audio de condicionamiento ausente: {audio}")
        sha = hashlib.sha256(audio.read_bytes()).hexdigest()
        if sha != it["sha256"]:
            raise ValueError(f"{it['item_id']}: sha256 no coincide")
        if it["transcript_es"] in out:
            raise ValueError(f"texto repetido entre ítems: {it['transcript_es']!r} (asociación ambigua)")
        out[it["transcript_es"]] = {"item_id": it["item_id"], "path": str(audio), "sha256": sha, "license": it["license"],
                                    "ref_text": it["transcript_es"], "speaker_id": it.get("speaker_id"),
                                    "dialect_label": it.get("dialect_label"), "attribution": it.get("attribution"),
                                    "license_url": it.get("license_url"), "consent_note": it.get("consent_note"),
                                    "utterance_duration_s": it.get("utterance_duration_s"),
                                    "file_duration_s": it.get("file_duration_s")}
    return out


class Qwen3CloneTTS:
    """Envoltorio mínimo con la misma interfaz que ``runtime.components.KokoroTTS``:
    ``synth(texto)`` es un generador de chunks con ``pcm`` (s16le), ``sample_rate``, ``text``."""

    def __init__(self, cfg: dict, *, model=None, to_pcm=None) -> None:
        self.cfg = cfg
        self.per_item = load_conditioning(cfg["conditioning_file"]) if cfg.get("conditioning_file") else None
        self.current: dict[str, Any] | None = None  # referencia del ítem en curso (modo por ítem)
        self.ref = (None if self.per_item is not None else load_reference(cfg["ref_audio"], cfg.get("ref_meta")))
        self.clone_mode = ("icl_per_item" if self.per_item is not None
                           else "icl" if cfg.get("ref_text") else "speaker_embedding_only")
        if model is None:  # carga real (pruebas inyectan un modelo falso)
            from mlx_audio.tts.utils import load_model

            from runtime import config as C

            path = C.snapshot(QWEN3_REPO, QWEN3_REV, ("config.json",))
            model = load_model(path)
        self.model = model
        self._to_pcm = to_pcm or self._mx_to_pcm
        self.sample_rate = int(getattr(model, "sample_rate", 24000))

    @staticmethod
    def _mx_to_pcm(audio) -> bytes:
        import mlx.core as mx
        import numpy as np

        mx.eval(audio)
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()

    def _kwargs(self) -> dict[str, Any]:
        c = self.cfg
        if self.per_item is not None:
            if self.current is None:
                raise RuntimeError("modo por ítem sin referencia activa: llamar select_item(text_es) antes de synth")
            ref_path, ref_text = self.current["path"], self.current["ref_text"]
        else:
            ref_path, ref_text = self.ref["path"], c.get("ref_text")
        kw: dict[str, Any] = {
            "ref_audio": ref_path, "ref_text": ref_text, "lang_code": c["lang_code"],
            "stream": True, "streaming_interval": c["streaming_interval"],
            "streaming_context_size": c["streaming_context_size"],
        }
        for k in ("temperature", "top_k", "top_p", "max_tokens"):
            if c.get(k) is not None:
                kw[k] = c[k]
        return kw

    def select_item(self, text_es: str) -> dict[str, Any]:
        """Fija la referencia por ítem a partir del texto de entrada (único). Sin coincidencia: error, nunca otra voz."""
        if self.per_item is None:
            raise RuntimeError("select_item solo aplica con conditioning_file")
        if text_es not in self.per_item:
            raise KeyError(f"texto sin referencia de condicionamiento: {text_es!r}")
        self.current = self.per_item[text_es]
        return self.current

    def synth(self, text_en: str):
        from runtime.components import TTSChunk

        if self.cfg.get("seed") is not None:
            import mlx.core as mx

            mx.random.seed(int(self.cfg["seed"]))
        for r in self.model.generate(text_en, **self._kwargs()):
            a = getattr(r, "audio", None)
            if a is None:
                continue
            pcm = self._to_pcm(a)
            if pcm:
                yield TTSChunk(pcm, int(getattr(r, "sample_rate", self.sample_rate)), text_en)


class TextMtQwen3CloneAdapter(TextMtTtsAdapter):
    name = "text-marian-ct2+qwen3-tts-clone"
    version = "0.1-experimental"

    def _make_tts(self, cfg: dict):
        return Qwen3CloneTTS(cfg)

    async def process_text(self, stream_id: str, text_es: str) -> None:
        if getattr(self.tts, "per_item", None) is not None:
            await self._gpu.run(self.tts.select_item, text_es)  # en el hilo del TTS (arrays MLX ligados al hilo)
        await super().process_text(stream_id, text_es)

    def _split(self, text: str) -> list[str]:  # type: ignore[override]
        if self.cfg.get("split_clauses"):
            return super()._split(text)
        return [text]

    async def open(self, config: dict[str, Any]) -> None:
        cfg = {**DEFAULTS, **config}
        if cfg["stages"] != "mt+tts":
            raise ValueError("este adaptador solo tiene sentido con stages='mt+tts'")
        if not cfg["streaming_interval"] or cfg["streaming_interval"] <= 0:
            raise ValueError("streaming_interval debe ser > 0")
        await super().open(cfg)
        self.name = "text-marian-ct2+qwen3-tts-clone"

    def _describe_tts(self) -> dict[str, Any]:
        from runtime import config as C

        qdir = C.snapshot(QWEN3_REPO, QWEN3_REV, ("config.json",))
        if self.cfg.get("conditioning_file"):
            per = load_conditioning(self.cfg["conditioning_file"])
            ref = {"mode": "por_ítem: audio original en español + transcripción humana",
                   "items": [{k: v[k] for k in ("item_id", "sha256", "license", "license_url", "attribution", "speaker_id", "dialect_label",
                                                "utterance_duration_s", "file_duration_s", "consent_note")} for v in per.values()],
                   "consent_statement": ("el consentimiento individual NO está publicado por la fuente (OpenSLR 61, CC-BY-SA-4.0); esto es evaluación "
                                         "local de voz pública autorizada por el usuario, sin redistribución ni afirmación de consentimiento "
                                         "específico de clonación ni jurídica")}
        else:
            ref = load_reference(self.cfg["ref_audio"], self.cfg.get("ref_meta"))
        keys = ("lang_code", "streaming_interval", "streaming_context_size", "temperature", "top_k", "top_p",
                "max_tokens", "seed", "split_clauses")
        return {"repo": QWEN3_REPO, "revision": QWEN3_REV, "weights": snapshot.weights_fingerprint(qdir),
                "clone_mode": ("icl_per_item" if self.cfg.get("conditioning_file")
                               else "icl" if self.cfg.get("ref_text") else "speaker_embedding_only"),
                "ref_text_provided": bool(self.cfg.get("ref_text") or self.cfg.get("conditioning_file")), "reference": ref,
                "generation": {k: self.cfg.get(k) for k in keys},
                "nota": "experimental; no hay afirmación de velocidad ni calidad"}

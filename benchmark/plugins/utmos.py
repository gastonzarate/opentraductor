"""Proxy automático de naturalidad: UTMOSv2.

NO es MOS humano: es un predictor entrenado para aproximar MOS de habla
sintética. Código y pesos MIT (verificado: LICENSE del repo, licencia `mit` en
la ficha de Hugging Face sarulab-speech/UTMOSv2). Parámetros explícitos
(fold, seed, repeticiones, dominio) quedan en cada resultado.

Limitaciones: el predictor se entrenó con juicios sobre habla sintética de
dominios concretos (ver paper); su validez para TTS en inglés generado a partir
de traducción es-AR→en-US y para voces clonadas no está demostrada. Sirve para
ordenar/descartar con cautela y debe contrastarse con la escucha ciega humana.

Pesos: ``create_model(pretrained=True)`` descarga con ``wget``; aquí se exige
``checkpoint_path`` local para que la descarga sea explícita y verificable.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as md
from pathlib import Path

from . import PluginUnavailable

LABEL = "UTMOSv2 (predictor automático de naturalidad; NO es MOS humano)"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class UtmosV2:
    def __init__(self, checkpoint_path: str | Path, *, fold: int = 0, seed: int = 42,
                 num_repetitions: int = 1, predict_dataset: str = "sarulab", device: str = "cpu"):
        try:
            import utmosv2  # type: ignore
        except ImportError as exc:
            raise PluginUnavailable("utmosv2 no instalado") from exc
        ck = Path(checkpoint_path)
        if not ck.exists():
            raise PluginUnavailable(f"pesos no encontrados: {ck}")
        self.params = dict(fold=fold, seed=seed, num_repetitions=num_repetitions,
                           predict_dataset=predict_dataset, device=device, config="fusion_stage3")
        self.checkpoint = ck
        self.checkpoint_sha256 = file_sha256(ck)
        self._model = utmosv2.create_model(pretrained=True, config="fusion_stage3", fold=fold,
                                           checkpoint_path=ck, seed=seed, device=device)

    def _seed_all(self) -> None:
        """UTMOSv2 elige tramos de audio al azar: fijar semillas hace el puntaje
        reproducible para el mismo archivo (si no, varía entre corridas)."""
        import random

        import numpy as np
        import torch

        sd = self.params["seed"]
        random.seed(sd)
        np.random.seed(sd)
        torch.manual_seed(sd)

    def predict_file(self, wav_path: str | Path) -> float:
        self._seed_all()
        v = self._model.predict(input_path=str(wav_path), num_repetitions=self.params["num_repetitions"],
                                predict_dataset=self.params["predict_dataset"], device=self.params["device"],
                                verbose=False)
        return float(v)

    def describe(self) -> dict:
        return {"label": LABEL, "params": self.params, "checkpoint": str(self.checkpoint),
                "checkpoint_sha256": self.checkpoint_sha256, "utmosv2_version": md.version("utmosv2"),
                "license": "MIT (código y pesos)", "is_human_mos": False}

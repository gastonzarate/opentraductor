"""Benchmark de traducción incremental causal (módulo 1, diagnóstico, sin modelos).

Contrato en TRES ESPACIOS que no se mezclan (ver ``contracts``): lo que mide el ARNÉS, lo que DECLARA el candidato
(solo claims) y lo que aporta el EVALUADOR de forma independiente (unidades de referencia, soporte, correspondencia).
No modifica ``runner``/``contracts`` actuales ni el runtime; la integración llega después con el contrato probado.
"""
from . import contracts, metrics, report  # noqa: F401

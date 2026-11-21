"""T-83: last_endpoint_to_audio_ms también mide frases cuya primera cláusula suena antes del endpoint (SIMUL=1).

Antes, service.py solo medía si el primer tts_chunk de la frase llegaba después del endpoint; con cláusulas
simultáneas esas frases quedaban sin medida (QA, clip 1516: 0 de 10).
"""
import asyncio

from tests.runtime.test_service import svc  # noqa: F401  fixture


def run(svc, kind, **f):
    asyncio.run(svc._on_engine_event(kind, f))


def chunk(svc, seg, plan):
    run(svc, "tts_chunk", seg_id=seg, pcm=b"\x00\x00" * 160, sample_rate=16000, planned_text_en=plan)


def test_classic_phrase_measured_at_first_chunk(svc):  # noqa: F811
    run(svc, "endpoint", seg_id=1)
    chunk(svc, 1, "hello")
    assert svc._m["last_endpoint_to_audio_ms"] is not None


def test_clause_before_endpoint_measures_next_clause_not_the_running_one(svc):  # noqa: F811
    chunk(svc, 2, "the police")          # primera cláusula, antes del endpoint
    run(svc, "endpoint", seg_id=2)
    chunk(svc, 2, "the police")          # sigue la misma cláusula: no es lo que falta oír
    assert svc._m["last_endpoint_to_audio_ms"] is None
    chunk(svc, 2, "in the streets")      # cláusula nueva tras el endpoint: se mide
    assert svc._m["last_endpoint_to_audio_ms"] is not None
    svc._m["last_endpoint_to_audio_ms"] = None
    chunk(svc, 2, "more")                # una sola medida por endpoint
    assert svc._m["last_endpoint_to_audio_ms"] is None

"""T-73 spec_on_silence: especulación por pausa, acierto normalizado, relanzada con el finish especulativo del ASR
y evento spec_result por frase. Componentes falsos (sin modelos)."""
import asyncio

from runtime.pipeline import Engine
from tests.runtime.test_speculative import FakeMT, FakeTTS, run_segment


def spec_results(ev): return [f for k, f in ev if k == "spec_result"]


def test_normalized_hit_reuses_partial_translation():
    async def go():
        mt = FakeMT()
        e, ev = await run_segment(partial="hola mundo", final="Hola, mundo", spec=False, spec_on_silence=True,
                                  mt=mt, settle=True)
        assert mt.calls == ["hola mundo"]
        assert e.spec_stats["hit"] == 1 and e.spec_stats["hit_norm"] == 1
        r = spec_results(ev)
        assert len(r) == 1 and r[0]["spec"] == "hit" and r[0]["launched"] == 1 and r[0]["wasted_ms"] == 0.0
        await e.close()
    asyncio.run(go())


def test_miss_reports_wasted_and_runs_normal_path():
    async def go():
        mt = FakeMT()
        e, ev = await run_segment(partial="hola mun", final="hola mundo entero", spec=False, spec_on_silence=True,
                                  mt=mt, settle=True)
        assert mt.calls == ["hola mun", "hola mundo entero"]
        r = spec_results(ev)
        assert r[0]["spec"] == "miss" and r[0]["launched"] == 1 and r[0]["wasted_ms"] > 0
        assert [f for k, f in ev if k == "mt_final"][0]["text"] == "EN hola mundo entero"
        await e.close()
    asyncio.run(go())


def test_finish_text_relaunches_speculation_and_hits():
    async def go():
        mt = FakeMT()
        e, ev = await run_segment(partial="hola mun", final="hola mundo", spec=False, spec_on_silence=True,
                                  asr_spec_finish_on_pause=True, asr_spec_finish_silence_ms=32, spec_silence_ms=32,
                                  mt=mt, settle=True)
        assert e.spec_stats["from_finish"] == 1
        assert mt.calls[-1] == "hola mundo" and mt.calls.count("hola mundo") == 1
        r = spec_results(ev)
        assert r[0]["spec"] == "hit"
        await e.close()
    asyncio.run(go())


def test_off_by_default_no_spec_result():
    async def go():
        e, ev = await run_segment(partial="hola", final="hola", spec=False, settle=True)
        assert spec_results(ev) == [] and e.spec_stats["started"] == 0
        await e.close()
    asyncio.run(go())


def test_spec_norm():
    assert Engine._spec_norm("¿Hola,  Mundo?") == "hola mundo"

import sys
from pathlib import Path

# El proyecto no tiene config raíz todavía: hacer importable ``benchmark``.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark import audio as A  # noqa: E402
from benchmark import runner  # noqa: E402
from benchmark.clock import run_virtual  # noqa: E402
from benchmark.replay import Segment  # noqa: E402


def make_item(i=0, lead=0.3, speech=1.0, tail=0.2, method="human_annotation", rec=None):
    pcm = A.silence(lead) + A.tone(speech, freq_hz=300 + 10 * i) + A.silence(tail)
    return runner.BenchItem(
        item_id=f"it{i}",
        recording_id=rec or f"rec{i}",
        pcm=pcm,
        segment=Segment(f"it{i}", lead, lead + speech, method, None),
        infra_synthetic=True,
    )


def run_dummy(adapter, items, **cfg_kw):
    cfg_kw.setdefault("repetitions", 1)
    cfg_kw.setdefault("warmup_streams", 0)
    cfg_kw.setdefault("n_boot", 50)
    asr = cfg_kw.pop("audible_asr", None)
    cfg = runner.RunConfig(**cfg_kw)

    async def main(clock):
        return await runner.run_benchmark(adapter, clock, items, cfg, audible_asr=asr)

    return run_virtual(main)

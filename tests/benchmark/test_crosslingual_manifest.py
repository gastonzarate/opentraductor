import importlib.util
import json
from pathlib import Path

import pytest
from conftest import ROOT

from benchmark.textset import load_text_items

spec = importlib.util.spec_from_file_location("clx", ROOT / "scripts" / "eval" / "crosslingual_manifest.py")
clx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clx)


def test_pilot_manifest_is_human_development_text_without_asr(tmp_path):
    out = tmp_path / "t.jsonl"
    rows = clx.build(ROOT / "docs" / "crosslingual-voice-pilot.json", out)
    assert len(rows) in (2, 3) and all(r["transcript_es_provenance"] == "human" and r["split"] == "development" for r in rows)
    items = load_text_items(out)
    assert {i.item_id for i in items} == {r["item_id"] for r in rows} and all(i.pcm == b"" for i in items)


def test_pilot_refuses_heldout_and_non_human(tmp_path):
    d = json.loads((ROOT / "docs" / "crosslingual-voice-pilot.json").read_text())
    d["selection"]["split"] = "test"
    p = tmp_path / "p.json"
    p.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="development"):
        clx.build(p, tmp_path / "o.jsonl")

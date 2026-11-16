import csv
import json
import re
import shutil
import subprocess

import pytest
from conftest import make_item, run_dummy

from benchmark import listening, listening_ui
from benchmark.__main__ import main as cli_main
from benchmark.adapters.dummy import DummyAdapter

NODE = shutil.which("node")


@pytest.fixture()
def package(tmp_path):
    dirs = []
    for name in ("sistemaAlfa_secreto", "sistemaBeta_secreto"):
        d = tmp_path / name
        run_dummy(DummyAdapter(), [make_item(0), make_item(1)], export_audio_dir=str(d), system_name=name)
        dirs.append(d / "export_index.jsonl")
    out = tmp_path / "pkg"
    listening.prepare_from_indexes(dirs, out, raters=["r1"], seed=7, degraded_controls=1)
    return out


def _script(html):
    return re.search(r"<script>(.*)</script>", html, re.S).group(1)


def test_build_is_blind_relative_and_separate(tmp_path, package):
    out = tmp_path / "html"
    info = listening_ui.build(package, out)
    assert info["n_stimuli"] == 5 and info["seed"] == 7
    html = (out / "index.html").read_text(encoding="utf-8")
    for secret in ("sistemaAlfa", "sistemaBeta", "control:", "degraded", "blind_key", "dummy", "it0"):
        assert secret not in html and secret not in (out / "listening_ui_manifest.json").read_text()
    assert not (out / "blind_key.json").exists() and not (out / "ratings_template.csv").exists()
    assert sorted(p.name for p in (out / "stimuli").iterdir()) == sorted(f"S{n:04d}.wav" for n in range(5))
    assert 'src="http' not in html and "https://" not in html and "fetch(" not in html  # sin red
    assert 'a.src="stimuli/"' in html or 'a2.src="stimuli/"' in html  # ruta relativa
    assert '<html lang="es">' in html and "aria-live" in html
    # nada puntuado de antemano: el JSON embebido no trae valores de puntuación
    data = json.loads(re.search(r'id="data">(.*?)</script>', html, re.S).group(1))
    assert set(data["stimuli"]["S0000"]) == {"ref", "refkind", "na"} and data["seed"] == 7
    assert listening_ui.check_blind(out, json.loads((package / "blind_key.json").read_text())["key"]) == []


def test_build_refuses_output_inside_package_and_leak(tmp_path, package):
    with pytest.raises(ValueError, match="aparte"):
        listening_ui.build(package, package / "html")
    (package / "stimuli" / "extra_secreto.wav").write_bytes(b"x")
    with pytest.raises(ValueError, match="no declarados"):
        listening_ui.build(package, tmp_path / "o")


def test_na_dims_preserved_and_references(tmp_path, package):
    out = tmp_path / "html"
    listening_ui.build(package, out)
    key = json.loads((package / "blind_key.json").read_text())["key"]
    data = json.loads(re.search(r'id="data">(.*?)</script>', (out / "index.html").read_text(), re.S).group(1))
    for sid, v in key.items():
        assert data["stimuli"][sid]["na"] == sorted(v["na_dims"])
        assert {"expresividad", "similitud_voz"} <= set(data["stimuli"][sid]["na"])  # sin referencia


def test_rater_order_deterministic_differs_and_is_permutation():
    ids = [f"S{n:04d}" for n in range(15)]
    a = listening_ui.rater_order(ids, 1, "ana")
    assert a == listening_ui.rater_order(ids, 1, " ana ") and sorted(a) == ids
    others = {tuple(listening_ui.rater_order(ids, 1, r)) for r in ("beto", "carla", "dani")}
    assert tuple(a) not in others and len(others) == 3
    assert a != listening_ui.rater_order(ids, 2, "ana")


@pytest.mark.skipif(NODE is None, reason="node no disponible")
def test_js_shuffle_matches_python_and_script_parses(tmp_path, package):
    out = tmp_path / "html"
    listening_ui.build(package, out)
    js = _script((out / "index.html").read_text(encoding="utf-8"))
    (tmp_path / "ui.js").write_text(js)
    chk = subprocess.run([NODE, "--check", str(tmp_path / "ui.js")], capture_output=True, text=True)
    assert chk.returncode == 0, chk.stderr
    fn = re.search(r"function fnv1a.*?\nfunction order\(.*?\}\n", js, re.S).group(0)
    ids = [f"S{n:04d}" for n in range(15)]
    for rater, seed in (("rater1", 20260514), ("Ñandú ünï", 3), ("x", 0)):
        prog = fn + f"console.log(JSON.stringify(order({json.dumps(ids)},{seed},{json.dumps(rater)})))"
        res = subprocess.run([NODE, "-e", prog], capture_output=True, text=True)
        assert res.returncode == 0, res.stderr
        assert json.loads(res.stdout) == listening_ui.rater_order(ids, seed, rater)


def test_exported_json_and_csv_feed_existing_summary(tmp_path, package):
    key = json.loads((package / "blind_key.json").read_text())["key"]
    sids = sorted(key)
    ratings = []
    for i, sid in enumerate(sids):
        na = set(key[sid]["na_dims"])
        ratings.append({"stimulus_id": sid, "reference_id": key[sid]["reference_id"] or "",
                        **{d: ("NA" if d in na else (4 if i % 2 else 2)) for d in listening.DIMENSIONS}, "comentario": "",
                        "escuchas_completas": 1})
    ratings[0]["naturalidad"] = ""  # sin dato: no se inventa
    doc = {"schema": listening_ui.SCHEMA, "rater": "ana", "package_seed": 7, "ratings": ratings}
    jp = tmp_path / "ratings_ana.json"
    jp.write_text(json.dumps(doc))
    from_json = listening.summarize_ratings(jp, package / "blind_key.json")
    # el mismo contenido como CSV (formato que exporta la UI) da el mismo resumen
    cp = tmp_path / "ratings_ana.csv"
    with cp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["rater", "stimulus_id", "reference_id", *listening.DIMENSIONS, "comentario"])
        w.writeheader()
        for r in ratings:
            w.writerow({"rater": "ana", **{k: r[k] for k in ["stimulus_id", "reference_id", *listening.DIMENSIONS, "comentario"]}})
    assert listening.summarize_ratings(cp, package / "blind_key.json") == from_json
    assert from_json["n_empty_cells"] == 1 and from_json["n_invalid_cells"] == 0
    assert "no es MOS formal" in from_json["caveat"]
    # varios evaluadores y CLI
    doc2 = dict(doc, rater="beto")
    jp2 = tmp_path / "ratings_beto.json"
    jp2.write_text(json.dumps(doc2))
    two = listening.summarize_ratings([jp, jp2], package / "blind_key.json")
    assert max(s["n_raters"] for s in two["systems"].values()) == 2
    assert cli_main(["listening-summary", "--ratings", str(jp), str(jp2), "--key", str(package / "blind_key.json")]) == 0


def test_summary_rejects_unknown_stimulus_and_schema(tmp_path, package):
    bad = tmp_path / "b.json"
    bad.write_text(json.dumps({"schema": "otro", "rater": "a", "ratings": []}))
    with pytest.raises(ValueError, match="esquema"):
        listening.summarize_ratings(bad, package / "blind_key.json")
    bad.write_text(json.dumps({"schema": listening_ui.SCHEMA, "rater": "a", "ratings": [{"stimulus_id": "S9999"}]}))
    with pytest.raises(ValueError, match="desconocido"):
        listening.summarize_ratings(bad, package / "blind_key.json")


def test_cli_listening_ui(tmp_path, package, capsys):
    assert cli_main(["listening-ui", "--package", str(package), "--out", str(tmp_path / "h")]) == 0
    assert (tmp_path / "h" / "index.html").exists()


def test_voice_reference_only_enables_similarity_and_warns_when_unpaired(tmp_path):
    from conftest import ROOT

    idx = []
    for name, ids in (("cloneSys", [0]), ("fixedSys", [1])):
        d = tmp_path / name
        run_dummy(DummyAdapter(), [make_item(i) for i in ids], export_audio_dir=str(d), system_name=name)
        idx.append(d / "export_index.jsonl")
    ref = tmp_path / "voz.wav"
    from benchmark import audio as A

    A.write_wav_pcm16(ref, A.tone(0.3, sample_rate=24000), 24000)
    out = tmp_path / "p"
    info = listening.prepare_from_indexes(idx, out, raters=["r1"], seed=1, system_meta={"cloneSys": {"voice_cloning": True}},
                                          voice_reference_audio={"it0": str(ref)})
    assert "no son comparables" in info["warning_unpaired"].lower() or "NO son comparables" in info["warning_unpaired"]
    key = json.loads((out / "blind_key.json").read_text())["key"]
    by_sys = {v["system"]: v for v in key.values()}
    assert by_sys["cloneSys"]["reference_id"] and "similitud_voz" not in by_sys["cloneSys"]["na_dims"]
    assert "expresividad" in by_sys["cloneSys"]["na_dims"]  # la muestra de voz no es el original
    assert {"expresividad", "similitud_voz"} <= set(by_sys["fixedSys"]["na_dims"])
    assert "MUESTRA DE VOZ" in (out / "INSTRUCCIONES.md").read_text()
    html_out = tmp_path / "h"
    listening_ui.build(out, html_out)
    assert "MUESTRA DE VOZ" in (html_out / "index.html").read_text()
    assert "cloneSys" not in (html_out / "index.html").read_text()


def test_reference_kind_is_declared_per_stimulus_and_labels_make_no_false_language_claim(tmp_path):
    from benchmark import audio as A

    d = tmp_path / "s"
    run_dummy(DummyAdapter(), [make_item(0), make_item(1)], export_audio_dir=str(d), system_name="sysX")
    ref = tmp_path / "ref.wav"
    A.write_wav_pcm16(ref, A.tone(0.3, sample_rate=24000), 24000)
    out = tmp_path / "p"
    listening.prepare_from_indexes([d / "export_index.jsonl"], out, raters=["r"], seed=1, system_meta={"sysX": {"voice_cloning": True}},
                                   voice_reference_audio={"it0": str(ref)}, reference_audio={"it1": str(ref)})
    key = json.loads((out / "blind_key.json").read_text())["key"]
    kinds = {v["item_id"]: v["reference_kind"] for v in key.values()}
    assert kinds == {"it0": "speaker_sample", "it1": "source_utterance"}
    md = (out / "INSTRUCCIONES.md").read_text()
    assert "MUESTRA DE VOZ" in md and "audio fuente público (en español)" in md
    assert not md.startswith("# Instrucciones de escucha ciega\n\nEscuchá cada estímulo `S####.wav` con auriculares, en un lugar silencioso, UNA vez completa\n(podés repetir una segunda vez si dudás). Si hay audio de referencia `R####.wav` (original")
    html_out = tmp_path / "h"
    listening_ui.build(out, html_out)
    html = (html_out / "index.html").read_text()
    data = json.loads(re.search(r'id="data">(.*?)</script>', html, re.S).group(1))
    assert {v["refkind"] for v in data["stimuli"].values() if v["ref"]} == {"speaker_sample", "source_utterance"}
    assert "NO es el original" in html and "Referencia (original en español)" not in html
    # solo muestra de voz: ninguna afirmación de original en español en el texto visible
    out2 = tmp_path / "p2"
    listening.prepare_from_indexes([d / "export_index.jsonl"], out2, raters=["r"], seed=1, system_meta={"sysX": {"voice_cloning": True}},
                                   voice_reference_audio={"it0": str(ref), "it1": str(ref)})
    assert "original en español" not in (out2 / "INSTRUCCIONES.md").read_text()




@pytest.mark.skipif(NODE is None, reason="node no disponible")
def test_local_storage_is_isolated_by_package_id_and_same_revision_keeps_progress(tmp_path, package):
    a = tmp_path / "a"
    listening_ui.build(package, a)
    pid_a = json.loads((a / "listening_ui_manifest.json").read_text())["package_id"]
    listening_ui.build(package, tmp_path / "a2")  # misma revisión
    assert json.loads((tmp_path / "a2" / "listening_ui_manifest.json").read_text())["package_id"] == pid_a
    # paquete distinto con la MISMA semilla, mismos S#### y el mismo evaluador: otro audio
    import shutil

    other = tmp_path / "pkg_other"
    shutil.copytree(package, other)
    wav = other / "stimuli" / "S0000.wav"
    from benchmark import audio as A

    A.write_wav_pcm16(wav, A.tone(0.4, freq_hz=777, sample_rate=16000), 16000)
    listening_ui.build(other, tmp_path / "b")
    pid_b = json.loads((tmp_path / "b" / "listening_ui_manifest.json").read_text())["package_id"]
    assert pid_a != pid_b
    assert json.loads((a / "listening_ui_manifest.json").read_text())["seed"] == json.loads((tmp_path / "b" / "listening_ui_manifest.json").read_text())["seed"]
    js = _script((a / "index.html").read_text(encoding="utf-8"))
    code = re.search(r"const storeKey=.*?function load\(\)\{.*?\n", js, re.S).group(0)
    prog = f"""
const db={{}};global.localStorage={{setItem:(k,v)=>{{db[k]=v}},getItem:k=>(k in db?db[k]:null),removeItem:k=>{{delete db[k]}}}};
let rater="ana",R={{}},D={{}};
{code}
D={{package_id:"{pid_a}"}};R={{S0000:{{naturalidad:"4"}}}};save();
const sameRev=JSON.stringify(load());
D={{package_id:"{pid_b}"}};const otherPkg=JSON.stringify(load());
// valor viejo (clave v1 seed+rater) NO se hereda
db["opentraductor-listening:20260514:ana"]=JSON.stringify({{S0000:{{naturalidad:"1"}}}});
const legacy=JSON.stringify(load());
// valor bajo la clave nueva pero con otro package_id guardado: se ignora
db["opentraductor-listening:v2:{pid_b}:ana"]=JSON.stringify({{package_id:"{pid_a}",ratings:{{S0000:{{naturalidad:"5"}}}}}});
const forged=JSON.stringify(load());
D={{package_id:"{pid_a}"}};const back=JSON.stringify(load());
console.log(JSON.stringify({{sameRev,otherPkg,legacy,forged,back}}));
"""
    res = subprocess.run([NODE, "-e", prog], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout)
    assert out["sameRev"] == out["back"] == '{"S0000":{"naturalidad":"4"}}'
    assert out["otherPkg"] == "{}" and out["legacy"] == "{}" and out["forged"] == "{}"
    # el export lleva package_id
    assert 'package_id:D.package_id' in (a / "index.html").read_text()


def test_summary_refuses_mixing_packages(tmp_path, package):
    key = package / "blind_key.json"
    sid = sorted(json.loads(key.read_text())["key"])[0]
    docs = []
    for n, pid in enumerate(("pkgA", "pkgB")):
        p = tmp_path / f"r{n}.json"
        p.write_text(json.dumps({"schema": listening_ui.SCHEMA, "rater": f"r{n}", "package_id": pid,
                                 "ratings": [{"stimulus_id": sid, "naturalidad": 3}]}))
        docs.append(p)
    with pytest.raises(ValueError, match="paquetes distintos"):
        listening.summarize_ratings(docs, key)
    assert listening.summarize_ratings(docs[:1], key)["n_invalid_cells"] == 0

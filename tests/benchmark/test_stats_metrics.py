import math

import pytest
from conftest import make_item, run_dummy

from benchmark import audio as A
from benchmark import metrics, stats
from benchmark.adapters.dummy import DummyAdapter
from benchmark.contracts import AdapterMode
from benchmark.replay import Segment, StreamTrace, TraceEvent


# ------------------------------ stats ------------------------------------


def test_nearest_rank_quantiles():
    v = list(range(1, 101))
    assert stats.nearest_rank(v, 0.5) == 50
    assert stats.nearest_rank(v, 0.95) == 95
    assert stats.nearest_rank(v, 0.99) == 99


def obs(values, deadline=5000.0, clusters=None):
    return stats.make_obs(values, clusters, deadline_ms=deadline)


def test_timeouts_are_right_censored_km_quantiles_identifiable_only_below_censoring():
    vals = [100.0 + i for i in range(80)] + [None] * 20  # 20 % timeouts
    s = stats.summarize_latency(obs(vals), n_boot=100)
    ca = s["censoring_aware"]["quantiles"]
    assert s["counts"]["timeout"] == 20 and s["n_total"] == 100
    assert ca["p50"]["identifiable"] and ca["p50"]["value"] == 149.0
    assert ca["p95"]["identifiable"] is False and ca["p95"]["value"] is None
    assert ca["p99"]["identifiable"] is False


def test_km_handles_heterogeneous_deadlines_where_inf_ordering_would_fail():
    # observados hasta 400 ms; un timeout con deadline 300 (< 400) y otro a 1000.
    recs = [stats.Obs("observed", v) for v in (100.0, 200.0, 400.0)] + [
        stats.Obs("timeout", 300.0), stats.Obs("timeout", 1000.0)
    ]
    # KM: S tras 100 = 4/5, tras 200 = 3/5, (censura 300), tras 400: at_risk=2 -> S=0.3
    assert stats.km_quantile(recs, 0.4) == 200.0
    assert stats.km_quantile(recs, 0.6) == 400.0
    assert stats.km_quantile(recs, 0.8) is None
    # el compuesto +inf NO conserva el orden con deadlines heterogéneos
    s = stats.summarize_latency(recs, n_boot=20)
    assert s["composite_time_to_valid_output"]["ordering_check"]["identifiable_ordering"] is False
    assert s["composite_time_to_valid_output"]["quantiles"]["p50"]["value"] is None


def test_common_deadline_above_all_observed_allows_composite():
    recs = obs([100.0, 200.0, 300.0, None])
    s = stats.summarize_latency(recs, n_boot=20)
    assert s["composite_time_to_valid_output"]["ordering_check"]["identifiable_ordering"] is True
    assert s["composite_time_to_valid_output"]["quantiles"]["p50"]["value"] == 200.0
    assert s["composite_time_to_valid_output"]["quantiles"]["p95"]["value"] is None


def test_unmeasurable_failure_and_anticipated_are_not_timeouts():
    recs = [stats.Obs("observed", 100.0), stats.Obs("observed", 200.0),
            stats.Obs("unmeasurable"), stats.Obs("failure"), stats.Obs("anticipated")]
    s = stats.summarize_latency(recs, n_boot=20)
    assert s["counts"] == {"observed": 2, "timeout": 0, "failure": 1, "anticipated": 1, "unmeasurable": 1}
    # distribución observada: no ve fallos ni no medibles (no son censura)
    assert s["censoring_aware"]["quantiles"]["p50"]["value"] == 100.0
    assert s["censoring_aware"]["quantiles"]["p50"]["n"] == 2
    # compuesto: fallo y anticipado = sin salida válida; no medible fuera del denominador
    assert s["composite_time_to_valid_output"]["quantiles"]["p50"]["n"] == 4
    assert s["composite_time_to_valid_output"]["quantiles"]["p50"]["value"] == 200.0
    assert s["composite_time_to_valid_output"]["quantiles"]["p95"]["value"] is None


def test_input_validation():
    with pytest.raises(ValueError, match="len"):
        stats.make_obs([1.0, 2.0], ["a"])
    with pytest.raises(ValueError):
        stats.quantile_report(obs([1.0]), 0.0)
    with pytest.raises(ValueError):
        stats.quantile_report(obs([1.0]), 1.5)
    with pytest.raises(ValueError, match="n_boot"):
        stats.quantile_report(obs([1.0]), 0.5, n_boot=0)
    with pytest.raises(ValueError, match="ci_level"):
        stats.quantile_report(obs([1.0]), 0.5, ci_level=1.0)
    with pytest.raises(ValueError):
        stats.Obs("observed", float("nan"))
    with pytest.raises(ValueError):
        stats.Obs("observed", float("inf"))
    with pytest.raises(ValueError):
        stats.Obs("observed", None)
    with pytest.raises(ValueError):
        stats.Obs("weird", 1.0)


def test_bootstrap_is_seeded_and_cluster_aware():
    vals = [float(i) for i in range(40)]
    cl = [f"r{i // 10}" for i in range(40)]  # 4 clusters
    a = stats.quantile_report(obs(vals, clusters=cl), 0.5, seed=1, n_boot=300)
    b = stats.quantile_report(obs(vals, clusters=cl), 0.5, seed=1, n_boot=300)
    c = stats.quantile_report(obs(vals), 0.5, seed=1, n_boot=300)
    assert a == b and a["n_clusters"] == 4
    assert a["ci"][1] - a["ci"][0] > c["ci"][1] - c["ci"][0]  # clusters => IC más ancho


def test_low_sample_uses_independent_clusters_not_stream_count():
    # 10 clips repetidos 10 veces = 100 streams, pero solo 10 clusters
    vals = [float(i % 10) for i in range(100)]
    cl = [f"clip{i % 10}" for i in range(100)]
    q = stats.quantile_report(obs(vals, clusters=cl), 0.99, n_boot=50)
    assert q["n"] == 100 and q["n_clusters"] == 10
    assert q["low_sample"] is True
    ok = stats.quantile_report(obs(vals, clusters=[f"c{i}" for i in range(100)]), 0.99, n_boot=50)
    assert ok["low_sample"] is False


def test_wilson_interval_bounds():
    lo, hi = stats.wilson_interval(0, 10)
    assert lo == 0.0 and 0 < hi < 0.4


# ---------------------------- métricas -----------------------------------


def _trace(segments, events, continuous=False, dur=10.0):
    return StreamTrace(
        stream_id="s",
        adapter_mode="streaming",
        clock_kind="virtual",
        t0_ns=0,
        input_duration_s=dur,
        segments=segments,
        events=events,
        deadline_s=5.0,
        continuous=continuous,
    )


def _tts(recv_s, dur_s=0.5, span=None, off=0.0):
    return TraceEvent(
        recv_ns=round(recv_s * 1e9),
        kind="tts_chunk",
        stream_id="s",
        pcm_samples=round(dur_s * 16000),
        sample_rate=16000,
        first_nonsilent_offset_s=off,
        source_span_s=span,
    )


def test_continuous_without_spans_is_no_correspondence_not_other_phrases_pcm():
    segs = [Segment("a", 0.0, 1.0, "human_annotation"), Segment("b", 3.0, 4.0, "human_annotation")]
    tr = _trace(segs, [_tts(1.4), _tts(4.4)], continuous=True)
    rs = metrics.segment_results(tr)
    assert [r.outcome for r in rs] == ["no_correspondence", "no_correspondence"]
    assert all(r.end_to_first_ms is None for r in rs)


def test_continuous_with_spans_attributes_each_output_to_its_own_phrase():
    segs = [Segment("a", 0.0, 1.0, "human_annotation"), Segment("b", 3.0, 4.0, "human_annotation")]
    tr = _trace(segs, [_tts(1.4, span=(0.0, 1.0)), _tts(4.6, span=(3.0, 4.0))], continuous=True)
    rs = metrics.segment_results(tr)
    assert rs[0].outcome == rs[1].outcome == "ok"
    assert rs[0].end_to_first_ms == pytest.approx(400.0)
    assert rs[1].end_to_first_ms == pytest.approx(600.0)


def test_continuous_phrase_with_no_own_output_is_timeout_even_if_other_phrase_has_output():
    segs = [Segment("a", 0.0, 1.0, "human_annotation"), Segment("b", 3.0, 4.0, "human_annotation")]
    tr = _trace(segs, [_tts(4.6, span=(3.0, 4.0))], continuous=True)
    rs = metrics.segment_results(tr)
    assert rs[0].outcome == "timeout"
    assert rs[1].outcome == "ok"


def test_late_first_pcm_beyond_deadline_is_timeout():
    segs = [Segment("a", 0.0, 1.0, "human_annotation")]
    tr = _trace(segs, [_tts(7.0)])
    assert metrics.segment_results(tr)[0].outcome == "timeout"


def test_no_annotated_end_gives_start_to_first_only():
    segs = [Segment("a", None, None, "unknown")]
    tr = _trace(segs, [_tts(1.2)])
    r = metrics.segment_results(tr)[0]
    assert r.outcome == "no_boundary" and r.end_to_first_ms is None
    assert r.start_to_first_ms == pytest.approx(1200.0) and r.start_origin == "stream_t0"


def test_backlog_is_not_semantic_lag():
    # 3 chunks de 0.5 s llegan juntos: el backlog crece sin spans -> lag no disponible
    tr = _trace([Segment("a", 0, 1, "human_annotation")], [_tts(2.0), _tts(2.0), _tts(2.0)])
    p = metrics.playout_analysis(tr)
    assert p["backlog"]["max_s"] == pytest.approx(1.0)
    assert p["semantic_lag"]["status"] == "unavailable"


def test_semantic_lag_with_spans_and_growing_drift():
    chunks = [_tts(1.0 + i * 0.6, 0.5, span=(i * 0.5, (i + 1) * 0.5)) for i in range(5)]
    tr = _trace([Segment("a", 0, 2.5, "human_annotation")], chunks)
    p = metrics.playout_analysis(tr)
    sl = p["semantic_lag"]
    assert sl["status"] == "available"
    assert sl["trend_s_per_s"] > 0  # el desfase crece 0.1 s por chunk
    assert sl["max_s"] > sl["min_s"]


def test_modes_are_never_mixed_in_aggregate():
    out = []
    for mode in (AdapterMode.STREAMING, AdapterMode.OFFLINE_FILE, AdapterMode.TEXT):
        ad = DummyAdapter(mode)
        it = make_item(0)
        it.text_es = "hola"
        res = run_dummy(ad, [it])
        out.extend(res.segment_results)
    agg = metrics.aggregate(out, n_boot=20)
    assert sorted(g["adapter_mode"] for g in agg["groups"]) == ["offline_file", "streaming", "text"]


def test_continuous_run_with_dummy_spans_end_to_end():
    items = [make_item(0), make_item(1)]
    res = run_dummy(
        DummyAdapter(),
        items,
        continuous=True,
        adapter_config={"emit_span": True},
        deadline_s=3.0,
    )
    assert [r.outcome for r in res.segment_results] == ["ok", "ok"]
    for r in res.segment_results:
        assert r.end_to_first_ms == pytest.approx(500.0, abs=30)
    assert res.playout[0]["semantic_lag"]["status"] == "available"


def test_continuous_run_without_spans_flags_no_correspondence():
    res = run_dummy(DummyAdapter(), [make_item(0), make_item(1)], continuous=True, deadline_s=3.0)
    assert {r.outcome for r in res.segment_results} == {"no_correspondence"}


def test_repetitions_seeds_and_warmup_are_separated_and_deterministic():
    kw = dict(repetitions=3, warmup_streams=2)
    a = run_dummy(DummyAdapter(), [make_item(0), make_item(1)], **kw)
    b = run_dummy(DummyAdapter(), [make_item(0), make_item(1)], **kw)
    assert len(a.warmup_traces) == 2 and len(a.traces) == 6
    assert [r.to_dict() for r in a.segment_results] == [r.to_dict() for r in b.segment_results]
    assert all(t.stream_id.startswith("warmup") for t in a.warmup_traces)
    assert a.cold_start["first_stream_end_to_first_ms"] is not None


def test_cold_start_open_is_measured():
    res = run_dummy(DummyAdapter(), [make_item(0)], adapter_config={"open_delay_ms": 250.0})
    assert res.cold_start["open_ms"] == pytest.approx(250.0, abs=1)


# ------------------- commits: varias frases y frases repetidas -------------------


def _chunk(recv_s, text, cid, span=None):
    e = _tts(recv_s, span=span)
    e.planned_text_en = text
    e.commit_id = cid
    return e


def test_planned_text_keeps_every_phrase_and_repeated_phrases_without_duplicating_chunks():
    seg = Segment("a", 0.0, 6.0, "human_annotation")
    ev = [
        _chunk(1.0, "hello", "c1"), _chunk(1.5, "hello", "c1"), _chunk(2.0, "hello", "c1"),  # 3 chunks, 1 frase
        _chunk(3.0, "how are you", "c2"), _chunk(3.5, "how are you", "c2"),
        _chunk(5.0, "hello", "c3"),  # frase legítimamente repetida
    ]
    r = metrics.segment_results(_trace([seg], ev))[0]
    assert r.planned_text_en == "hello how are you hello"
    assert r.n_commits == 3 and r.planned_text_ambiguous is False


def test_planned_text_without_commit_id_flags_ambiguity():
    seg = Segment("a", 0.0, 6.0, "human_annotation")
    ev = [_chunk(1.0, "hello", None), _chunk(1.5, "hello", None), _chunk(3.0, "bye", None)]
    r = metrics.segment_results(_trace([seg], ev))[0]
    assert r.planned_text_en == "hello bye"
    assert r.planned_text_ambiguous is True  # no se puede distinguir repetición pegada


def test_asr_final_per_segment_with_spans_in_continuous_streams():
    segs = [Segment("a", 0.0, 1.0, "human_annotation"), Segment("b", 3.0, 4.0, "human_annotation")]
    ev = [
        TraceEvent(recv_ns=int(1.2e9), kind="asr_final", stream_id="s", text="hola", source_span_s=(0.0, 1.0), commit_id="c1"),
        TraceEvent(recv_ns=int(4.2e9), kind="asr_final", stream_id="s", text="hola", source_span_s=(3.0, 4.0), commit_id="c2"),
    ]
    rs = metrics.segment_results(_trace(segs, ev, continuous=True))
    assert [r.asr_text_es for r in rs] == ["hola", "hola"] and rs[0].asr_status == "ok"
    ev_nospan = [TraceEvent(recv_ns=int(1.2e9), kind="asr_final", stream_id="s", text="hola")]
    rs = metrics.segment_results(_trace(segs, ev_nospan, continuous=True))
    assert rs[0].asr_status == "no_correspondence" and rs[0].asr_text_es is None


def test_two_sentences_and_repeated_sentence_end_to_end_with_dummy_continuous():
    items = [make_item(0), make_item(1)]
    res = run_dummy(DummyAdapter(), items, continuous=True, deadline_s=3.0,
                    adapter_config={"emit_span": True, "text_en": "same phrase", "asr_text": "misma frase"})
    # cada fragmento tiene su propia frase (igual texto, distinto commit)
    assert [r.planned_text_en for r in res.segment_results] == ["same phrase", "same phrase"]
    assert [r.n_commits for r in res.segment_results] == [1, 1]
    assert [r.asr_text_es for r in res.segment_results] == ["misma frase"] * 2


def test_semantic_lag_is_per_segment_flags_negative_and_skips_trend_with_few_chunks():
    chunks = [_tts(0.2 + i * 0.1, 0.5, span=(i * 0.5, (i + 1) * 0.5)) for i in range(3)]
    tr = _trace([Segment("a", 0, 1.5, "human_annotation")], chunks)
    sl = metrics.playout_analysis(tr)["semantic_lag"]
    assert "por segmento" in sl["granularity"] and "NO alineación palabra a palabra" in sl["granularity"]
    assert sl["trend_s_per_s"] is None and sl["trend_min_n"] == 5  # n=3: pendiente = ruido
    assert sl["min_s"] < 0 and sl["n_negative_unverified_anticipation"] >= 1


def test_trend_ignores_intra_segment_chunks_with_identical_span():
    # 6 chunks del MISMO span: la pendiente por reloj de recepción no es deriva de conversación
    chunks = [_tts(1.0 + i * 0.6, 0.5, span=(0.0, 2.0)) for i in range(6)]
    tr = _trace([Segment("a", 0, 2.0, "human_annotation")], chunks)
    sl = metrics.playout_analysis(tr)["semantic_lag"]
    assert sl["n_distinct_source_spans"] == 1 and sl["trend_s_per_s"] is None
    assert "intra-segmento" in sl["trend_note"]


def test_continuous_mt_unavailable_reason_says_not_attributed_not_missing():
    from benchmark import quality

    segs = [Segment("a", 0.0, 1.0, "human_annotation")]
    mt = TraceEvent(recv_ns=1, kind="mt_final", stream_id="s", text="hello")
    tr = _trace(segs, [mt, _tts(1.4)], continuous=True)
    res = metrics.segment_results(tr)
    q = quality.evaluate_run(res, [tr], {}, None, None)
    assert q["mt_text"]["status"] == "unavailable"
    assert "SÍ emitió 1 mt_final y 0 mt_partial" in q["mt_text"]["reason"]
    assert "No significa que falte la MT" in q["mt_text"]["reason"]
    tr0 = _trace(segs, [_tts(1.4)], continuous=True)
    q0 = quality.evaluate_run(metrics.segment_results(tr0), [tr0], {}, None, None)
    assert q0["mt_text"]["reason"] == "el candidato no emitió mt_final ni mt_partial"


def test_continuous_incremental_presence_is_not_reported_as_missing_when_unattributed():
    from benchmark import quality

    segs = [Segment("a", 0.0, 1.0, "human_annotation")]
    events = [
        TraceEvent(recv_ns=1, kind="asr_final", stream_id="s", text="hola"),
        TraceEvent(recv_ns=2, kind="mt_partial", stream_id="s", text="hello"),
        TraceEvent(
            recv_ns=3,
            kind="tts_chunk",
            stream_id="s",
            pcm_samples=1600,
            sample_rate=16000,
            first_nonsilent_offset_s=0.0,
            planned_text_en="hello",
        ),
    ]
    tr = _trace(segs, events, continuous=True)
    tr.stream_id = "r0-b0-a"
    for event in tr.events:
        event.stream_id = tr.stream_id
    q = quality.evaluate_run(
        metrics.segment_results(tr),
        [tr],
        {},
        None,
        transcripts={"a": ("hola", "human")},
    )
    assert q["event_presence"] == {
        "tts_chunk": 1,
        "tts_chunk_with_planned_text": 1,
        "mt_partial": 1,
        "mt_final": 0,
        "asr_final": 1,
    }
    assert q["rows_scored"] == 0
    assert q["rows_no_correspondence_excluded"] == 1
    assert "SÍ emitió 1 tts_chunk" in q["planned_text"]["reason"]
    assert "texto no se atribuye" in q["planned_text"]["reason"]
    assert "0 mt_final y 1 mt_partial" in q["mt_text"]["reason"]
    assert "SÍ emitió 1 asr_final" in q["candidate_asr_es"]["reason"]
    assert "text_checks" not in q
    assert any(
        "sin correspondencia fuente→salida" in caveat and "no se tratan como hipótesis vacías" in caveat
        for caveat in q["caveats"]
    )
    assert all(
        "no emitió" not in q[key]["reason"]
        for key in ("planned_text", "mt_text", "candidate_asr_es")
    )


def test_tts_without_planned_text_is_distinct_from_missing_attribution():
    from benchmark import quality

    segs = [Segment("a", 0.0, 1.0, "human_annotation")]
    tr = _trace(segs, [_tts(1.4)], continuous=True)
    q = quality.evaluate_run(metrics.segment_results(tr), [tr], {}, None, None)
    assert q["event_presence"]["tts_chunk"] == 1
    assert q["event_presence"]["tts_chunk_with_planned_text"] == 0
    assert "SÍ emitió 1 tts_chunk" in q["planned_text"]["reason"]
    assert "ninguno incluyó planned_text_en" in q["planned_text"]["reason"]
    assert "no es un problema de correspondencia fuente→salida" in q["planned_text"]["reason"]
    assert "no se atribuye" not in q["planned_text"]["reason"]


def test_tts_emission_rtf_is_labelled_not_synthesis_cost_in_streaming():
    tr = _trace([Segment("a", 0, 5, "human_annotation")], [_tts(1.0, 0.5), _tts(1.8, 0.5), _tts(2.6, 0.5)])
    r = metrics.playout_analysis(tr)["tts_emission_rtf"]
    assert r["rtf"] == pytest.approx(1.6 / 1.0) and r["valid_as_synthesis_cost"] is False
    assert "NO es costo de síntesis" in r["label"] and "formula" in r


def test_report_warns_ci_by_cluster_misses_temporal_dependence_in_continuous():
    from benchmark import report

    qs = {"p50": {"value": 1.0, "identifiable": True, "ci": (0.5, 1.5), "n": 8, "n_clusters": 4, "low_sample": True,
                  "min_n_for_quantile": 20}}
    assert "NO captura la dependencia temporal" in "\n".join(report._q_table("t", "l", qs, continuous=True))
    assert "dependencia temporal" not in "\n".join(report._q_table("t", "l", qs, continuous=False))

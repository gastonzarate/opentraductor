# scripts/eval

Measurement and analysis scripts. Nothing here runs in CI and none of it is needed to use the app. Most scripts load real models or need recordings that are not in this repository (`artifacts/` and `data/audio/` are git-ignored), and several play audio through BlackHole. Read the docstring at the top of each script before running it.

## Why some names start with `tNN_`

`tNN_` is the number of the work item (T-48, T-67, ...) of the project's private task tracker, where each measurement started. The prefix only groups scripts that belong to the same investigation. The tracker is not part of this repository, so the docstrings keep the number as a label and say what was measured. The investigations are grouped by topic below.

Numbers produced by these scripts and published in [README.md](../../README.md) are described with their method in [`docs/benchmark.md`](../../docs/benchmark.md) and [`docs/latency-final.md`](../../docs/latency-final.md).

## General tools (no `tNN_` prefix)

| Script | What it does |
|---|---|
| `setup.sh`, `run.sh` | Create the isolated evaluation environment (UTMOSv2, faster-whisper; CPU only) and run the benchmark CLI in it, offline. |
| `gate.sh`, `gate.py`, `store_baseline.json`, `weights.json` | Quality and latency gate: runs the metrics and fails if a threshold is broken. The baseline for the packaged-app configuration is `store_baseline.json`. |
| `compare_configs.sh`, `summarize_configs.py`, `ab_summary.py` | Compare runtime configurations on the development manifest and summarize p50/p95 of end of speech to first audio. |
| `final_validation.py` | Prepares the one-shot validation on the held-out test split (see [`docs/final-validation-protocol.md`](../../docs/final-validation-protocol.md)). |
| `continuous_recipe.sh`, `make_continuous_manifest.py`, `continuous_coverage.py`, `long_summary.py` | Long continuous stream (real recordings joined with synthetic pauses): build the manifest, run it, check coverage and summarize backlog and delay. |
| `audio_tail_stats.py`, `leading_quiet.py` | Silence at the end and start of exported WAV files, and internal pauses. |
| `device_loopback.py`, `player_latency_probe.py` | Output through BlackHole: measure latency and underflows of the player. They open the audio device. |
| `crosslingual_manifest.py` | Text manifest for the Spanish to English voice pilot in [`docs/crosslingual-voice-pilot.json`](../../docs/crosslingual-voice-pilot.json). |
| `mt_sentido_eval.py` | Does Qwen pick the right sense of ambiguous Argentine words (`sets/mt_sentido_ar.jsonl`). |
| `wait_quiet.sh` | Wait until the Mac is idle before measuring. |
| `mlxcap/` | Caps MLX memory so a run reproduces the packaged-app configuration. |
| `sets/` | Evaluation sets: talks with context and expected terms, ambiguous words. |

## Investigations by topic

### Speech recognition (ASR)

| Scripts | Question |
|---|---|
| `t48_asr_compare.sh`, `t48_short_set.py`, `t48_summary.py` | Parakeet against Moonshine tiny and small, including very short Rioplatense phrases. |
| `t58_asr_finish_probe.py`, `t58_spec_finish_probe.py`, `t67_asr_tail_probe.py`, `t67_asrfin.py`, `t67_asrfin_cmp.py`, `t67_tail.py`, `t67_vad_tail.py` | Where the time between end of speech and the final recognized text goes (end-of-speech wait, recognizer finish, trailing audio). |
| `t83_tail_ab.sh`, `t83_tail_cost.py`, `t83_tail_count.py` | Last syllable lost on soft endings ("policial" heard as "policia"): count them, and measure what keeping trailing audio costs. |
| `t94_asr_run.py`, `t94_finish_probe.py` | Transcribe a file with each recognizer and record timing, CPU and memory. |

### Translation (MT)

| Scripts | Question |
|---|---|
| `t50_run_all.sh`, `t50_pipeline_ram.py`, `t50_summary.py`, `t48_t50_queue.sh` | Peak memory and quality of Qwen3 4B, 1.7B and 0.6B in the full pipeline. |
| `t58_mt_*` (`glossary_latency`, `gpu_probe`, `kv_prefill_probe`, `stream_probe`, `timing_table`, `asrtexts`) | Translation latency: glossary on or off, cost inside the pipeline, prefill of the stable prefix, interleaving speech and decoding. |
| `t68_ctx_spec_probe.py` | Qwen with a long session context, with and without a small draft model. |
| `t69_make_charlas.py`, `t69_charlas_eval.py`, `t74_make_set.py`, `t74_make_set_bench.py`, `t62_make_idiom_set.py`, `t81_marian_terms.py` | Technical terms translated correctly in conversations with and without earlier context. Build the sets, score them, compare with Marian. |

### Delay and the sentence-by-sentence pipeline

| Scripts | Question |
|---|---|
| `t58_stage_breakdown.py`, `t58_write_base.py`, `t69_stages.py`, `t69_bench.sh`, `t69_summary.py` | Delay broken down by stage, and the A/B bench that uses it. |
| `t67_intercalado.sh`, `t67_intercalado.py`, `t80_intercalado.sh`, `t80_analisis.py`, `t81_bench.sh`, `t67_decision.py` | A/B runs with the variants alternated round by round, and the rule that decides between them. |
| `t67_simul_lag.py`, `t67_clause_count.py`, `t67_cuts.py`, `t67_make_charlas_audio.py`, `t88_first_cut_sim.py`, `t88_one_partial_sim.py`, `t88_trozos.py` | Translation by clauses: how often a sentence is split in two, how many voice chunks come out, and offline simulations of where to make the first cut. |
| `t80_service_metric.py` | Recompute the app's own delay metric from bench traces. |

### Voice (TTS) and playback

| Scripts | Question |
|---|---|
| `t67_tts_gaps.py`, `t67_duracion.py`, `t67_stream4_wavs.py`, `t67_wait_reason.py` | Gaps between voice chunks, total voice duration, prosody of the first chunk, and why the end-of-speech wait lasted what it did. |
| `t80_bh_gaps.py` | Does playing a test through BlackHole itself insert gaps in the audio? |

## Before you run one

- They assume the repository root as the working directory and the `.venv` of `uv sync --extra runtime`.
- Recordings, `artifacts/` and model weights are not included. Where a script needs them, its docstring says which.
- A busy Mac changes latency results. Use `wait_quiet.sh` or report the load average with your numbers.

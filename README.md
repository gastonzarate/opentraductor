# opentraductor

Local speech-to-speech translation from Argentine Spanish to US English for macOS on Apple Silicon.
You speak Spanish, and your Mac says it in English a moment later. Nothing leaves your machine: speech recognition, translation and voice synthesis all run on-device.

It is built for meetings: the English voice goes to a virtual microphone, so Google Meet or Discord can pick it up.

> **Status: experimental.** It works on the author's Mac, and the numbers below are honest but small samples. There is no 300 ms promise: today the delay between the end of your sentence and the first English audio is a bit under one second.
> Spanish version: [README.es.md](README.es.md).

## What it does

```
mic ──► VAD (Silero) ──► ASR (Parakeet or Moonshine) ──► translation (Qwen3 or Marian) ──► TTS (Kokoro) ──► virtual mic
```

- **100 % local.** After the one-time model download, the runtime forces offline mode. No telemetry, no accounts, no CDN.
- **No audio is stored.** Only text (original and translation) goes to a small local SQLite history you can export or delete from the UI.
- **Local web UI** on `127.0.0.1` only, with a per-run token and `Host`/`Origin` checks.
- Pick your microphone, your output (for meetings, a virtual microphone) and an optional local monitor.

## Requirements

- A Mac with **Apple Silicon**. Tested on an M5 Pro, 48 GB, macOS 26. Moonshine needs macOS 15 or later.
- Python 3.12 and [uv](https://docs.astral.sh/uv/).
- Disk: about 3 GB for the base models, more for Qwen (the 4B model needs about 16 GB of RAM or more; below that the code picks Qwen3-1.7B).
- Optional, for meetings: [BlackHole 2ch](https://github.com/ExistentialAudio/BlackHole). It is GPL-3.0 and it is **not** included. You install it yourself.

## Install

```bash
git clone https://github.com/<owner>/opentraductor.git
cd opentraductor

uv sync --extra runtime
uv run --extra runtime python -m runtime.setup     # downloads the models. Only step that uses the network.
uv run --extra runtime python -m runtime check     # verifies everything is on disk
uv run --extra runtime python -m runtime run       # opens http://127.0.0.1:8877/
```

Always pass `--extra runtime`; without it `uv` removes the runtime dependencies.

This default profile uses Parakeet (ASR), Marian (translation) and Kokoro (voice).

### Faster profile: Moonshine + Qwen3

This is the profile behind the latency numbers below. It needs the Moonshine extra and a Qwen model on disk.

```bash
uv sync --extra runtime --extra moonshine
python3 scripts/setup_moonshine.py                  # Moonshine Spanish streaming weights, hashes verified

# Qwen weights are not downloaded by `runtime.setup`. Put them anywhere and point to them:
uv run --extra runtime hf download mlx-community/Qwen3-4B-Instruct-2507-4bit --local-dir ~/qwen3-4b
export OPENTRADUCTOR_QWEN_DIR=~/qwen3-4b

uv run --extra runtime --extra moonshine python -m runtime run --asr-backend moonshine-tiny --mt-engine qwen
```

Keep both extras in every `uv` command of this profile, or `uv` removes the Moonshine package.

### Virtual microphone for meetings

Install BlackHole 2ch yourself (it asks for your admin password), choose "BlackHole 2ch" as the output in opentraductor and as the microphone in Meet or Discord. The UI has a **Virtual microphone** screen that detects it and shows the steps. Without BlackHole you can still listen on your speakers or a local monitor.

## Measured results

Source: internal benchmark by the project, 2026-11-16 to 2026-11-18. Same Mac, not idle (other processes were running), few clips. Treat them as an indication, not a certified result. Method and raw data: [`docs/benchmark.md`](docs/benchmark.md), [`docs/quality-latency-policy.md`](docs/quality-latency-policy.md).

| Measure | Value | Sample |
|---|---|---|
| Delay from end of speech to first English audio, median (Moonshine tiny + Qwen3-4B + Kokoro) | about 930 ms (p95 about 1.0 s) | 21 runs, 3 clips |
| Same, without the KV-prefill option | about 1060 ms (p95 about 1.2 s) | 21 runs, 3 clips |
| Speech recognition error rate (WER), Moonshine tiny | 0.047 | 3 clips |
| Technical terms translated correctly, no context | 69 % | internal set of ambiguous phrases |
| Technical terms translated correctly, with about 500 tokens of earlier talk | 77 % | same set |
| Voice naturalness proxy (UTMOSv2, not a human opinion) | 4.03 | 6 utterances |

Where the time goes in the median run: about 250 ms waiting for end of speech, 200 ms final recognition, 460 ms translation, 200 ms first synthesized audio.

Not tested: a physical microphone end to end, long sessions, and Meet or Discord with real participants. The simultaneous (incremental) mode is not usable from a fresh clone, see Limitations.

## Run the tests

```bash
uv run --extra runtime python -m pytest tests -q
```

Some tests need model weights or local benchmark artifacts and will be skipped or fail on a fresh clone. See [`docs/benchmark.md`](docs/benchmark.md).

## Project layout

- `runtime/`: the engine (`pipeline.py`), components (`components.py`), service (`service.py`), local HTTP control (`http_control.py`), history (`text_store.py`), audio I/O (`audio_io.py`) and the CLI (`__main__.py`, `setup.py`).
- `ui/`: the local web interface.
- `benchmark/`: the measurement harness. `scripts/`: dataset and evaluation scripts. `data/manifests/`: evaluation manifests.
- `docs/`: design notes, model candidates, datasets, usage.

## Limitations

- Latency is under one second, not the 300 ms the project first aimed for.
- Only Argentine Spanish to US English. Rioplatense Spanish has no complete dedicated evaluation yet.
- Output goes to one device. No echo cancellation, no mixing of several voices.
- The experimental incremental mode (`--engine-mode incremental_experimental`) depends on local files that are not in this repository. It does not run from a fresh clone.
- Mac only, Apple Silicon only. No Windows or Linux.

## Contributing

Issues and pull requests are welcome. Good places to start:

- **Latency:** the biggest costs are the end-of-speech wait, final ASR and the first translation tokens. Ideas in progress: adaptive end-of-sentence, speculative decoding with a small Qwen draft model, a persistent KV cache, clause-by-clause simultaneous translation.
- **Quality:** a larger test set of Argentine Spanish with human references, a user glossary file, longer context.
- **Portability:** a clean-machine install check and a cleaner way to fetch Qwen weights.

Before opening a pull request, run `uv run --extra runtime python -m pytest tests -q` and say which Mac and macOS you used. If you change latency or quality, include the benchmark command and the numbers.

## License

Code: [MIT](LICENSE). Models and datasets keep their own licenses and attributions, listed in [NOTICE.md](NOTICE.md).
Using this project to run a virtual microphone requires BlackHole, which has its own license (GPL-3.0).

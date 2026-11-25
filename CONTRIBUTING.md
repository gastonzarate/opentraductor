# Contributing to opentraductor

Thanks for helping. Issues and pull requests are welcome, in English or Spanish.

opentraductor is experimental and it is built and tested by one person on one Mac. Reports from other Macs are useful on their own, even without a code change.

## What to report: your Mac

Say which machine you used, because speed and memory change a lot between them:

- Mac model and chip (for example "MacBook Pro, M2 Pro, 32 GB").
- macOS version (Moonshine needs macOS 15 or later; the author tests on macOS 26).
- Which profile you ran: default (Parakeet + Marian + Kokoro) or fast (Moonshine + Qwen3).
- Whether BlackHole 2ch was in use, and what the Mac was doing at the time (a busy Mac adds delay).

Only Apple Silicon Macs are supported. Intel Macs, Windows and Linux are out of scope for now.

## Set up

You need Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/gastonzarate/opentraductor.git
cd opentraductor
uv sync --extra runtime
```

Always pass `--extra runtime` to `uv run`; without it `uv` removes the runtime dependencies. To run the app, see [README.md](README.md#install).

## Run the tests

```bash
uv run --extra runtime python -m pytest tests -q
```

Most tests use fakes and need no model weights, no microphone and no network. The few that need the evaluation audio (`data/audio/`, not in the repository for license and size reasons) skip themselves on a fresh clone. The same command runs in CI on every push and pull request (macOS 14, Apple Silicon): [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

A pull request needs the test run to pass. If you add behavior, add a test that does not need models or audio.

## Run the benchmarks

Benchmarks load real models and play real audio, so they are not part of CI and they are slow. Read [`docs/benchmark.md`](docs/benchmark.md) first. The scripts live in `benchmark/` (the harness) and `scripts/eval/` (the analysis and A/B scripts, grouped in [`scripts/eval/README.md`](scripts/eval/README.md)).

Rules for numbers, because the project has been wrong about them before:

- Close other apps, say what else was running, and report the load average. A busy Mac changes the median by tens of milliseconds.
- Alternate the variants round by round (A, B, A, B) instead of running all of A and then all of B.
- Give the sample size, the command, the commit and the Mac. One run is an indication, not a result.
- If you change latency or quality, include the numbers before and after in the pull request.

## Commit style

- One change per commit, with a subject line that says what changed and why it matters, in the imperative or the past tense, in English or Spanish. Keep the subject under about 100 characters.
- Start with an optional area tag in brackets when it helps: `[runtime]`, `[ui]`, `[docs]`, `[bench]`, `[packaging]`.
- Put the measured effect in the body when the change touches speed or quality (before, after, sample size).
- Do not rewrite published history. Do not force-push to `main`.

Example:

```
[runtime] Cap the recognition backlog at 20 s so a busy Mac skips old sentences

Test with recognition at 70 % of real time: backlog max 22 s over 300 s of speech
(40 s or more without the cap). OPENTRADUCTOR_ASR_SKIP_OLD_S=0 turns it off.
```

## Pull requests

1. Open an issue first for anything larger than a small fix, so we agree on the direction.
2. Branch from `main`, keep the change small and focused.
3. Run the tests. Say which Mac and macOS you used.
4. Fill in the pull request template.

New behavior that can hurt users should be behind an environment variable or a config field, and off by default until it has numbers (see how `OPENTRADUCTOR_*` switches are documented in the README).

## Licensing

The code is MIT ([LICENSE](LICENSE)). By opening a pull request you agree that your contribution is released under the same license. Do not add model weights, evaluation audio or code with an incompatible license. Third-party components are listed in [NOTICE.md](NOTICE.md).

## Security

Do not report vulnerabilities in a public issue. See [SECURITY.md](SECURITY.md).

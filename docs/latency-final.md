# Final latency and terms measurement (2026-11-20)

Measured by the project on one Mac, not idle (load average 2.0 to 5.5). Commit `e8bfbdb` (the defaults of the later commit `69fac8d` are the winning arm). The raw `artifacts/` folder is not in this repository; the numbers below are what it contained.

## Delay from end of speech to first English audio

Interleaved A/B: 12 rounds, 8 clips per round, the three arms run one after the other in every round. Scripts: `scripts/eval/t67_intercalado.sh`, `t67_intercalado.py`, `t67_decision.py`.

| Arm | Env | n | Median | p95 | Paired difference of the median per round |
|---|---|---|---|---|---|
| before | none | 96 | 635 ms | 743 ms | reference |
| first TTS chunk of 2 words | `OPENTRADUCTOR_TTS_FIRST_WORDS=2` | 96 | 592 ms | 694 ms | -25 ms (7 of 12 rounds at -20 ms or better) |
| 2 words + ASR in its own process + 1 ms GIL switch interval (**current defaults**) | `OPENTRADUCTOR_TTS_FIRST_WORDS=2 OPENTRADUCTOR_ASR_PROCESS=1 OPENTRADUCTOR_SWITCH_INTERVAL_MS=1` | 96 | **594 ms** | **691 ms** | -35 ms (8 of 12 rounds at -20 ms or better) |

Median by stage for the current defaults, in ms after the end of speech: waiting for end of speech 152, final ASR 180, translation 335. The first audio starts about 84 ms before the translation ends.

To go back to the previous behavior: `OPENTRADUCTOR_TTS_FIRST_WORDS=4`, `OPENTRADUCTOR_ASR_PROCESS=0`, `OPENTRADUCTOR_SWITCH_INTERVAL_MS=off`.

Checks on the same runs:

- Segmentation: 132 segments in 96 runs in every arm. Two clips (validation-1584 and validation-1652) are split in two at a long pause, identically in all arms. The new defaults add no extra cuts.
- TTS gaps: 0 phrases with a gap between audio chunks in the current arm (0 of 120 phrases with two or more chunks).

## Technical terms in conversations

7 short scripted conversations written by the project, 96 terms, Qwen3-4B with KV prefill, translation only. Script: `scripts/eval/t69_charlas_eval.py`.

| Earlier context | Terms correct |
|---|---|
| none | 81 of 96 (84.4 %) |
| about 200 tokens | 91 of 96 (94.8 %) |

Translation median 201 ms with context (p95 321 ms). Small, author-written set: use it to compare changes, not as a general quality claim.

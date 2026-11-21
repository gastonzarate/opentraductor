# Final latency and terms measurement (2026-11-20, updated 2026-11-21 with T-80 and T-83)

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

## T-80 (translation by clauses, `mt_simul_clauses` on by default)

Measured on commit `4806494`, 2026-11-21: 8 clips, 8 interleaved rounds, 64 runs per arm, load average 1.7 to 5.3 (median 3.0). Scripts: `scripts/eval/t80_intercalado.sh`, `t80_analisis.py`. The raw `artifacts/` folder is not in this repository.

| Arm | n | Median | p95 |
|---|---|---|---|
| translation by clauses off (`OPENTRADUCTOR_MT_SIMUL=0`) | 64 | 599 ms | 763 ms |
| translation by clauses on (**current default**) | 64 | **562 ms** | **673 ms** |

Difference: -37 ms at the median, -90 ms at p95. A negative total is valid: with clauses, the first audio can come out before the end of the annotated sentence (24 of 64 runs with the new version, 16 of 64 with the old one). Those runs lower the median. The 33 runs between 556 and 752 ms (short sentences) have a median of 619 ms; the old version's 48 non-negative runs have a median of 617 ms. Short sentences did not get faster.

First English audio counted from the start of speech, sentences longer than 5 s (2 clips, 16 runs per arm): 4.3 s with clauses, 6.3 s without.

Technical terms, 96 terms, 7 conversations, translation only:

| Earlier context | clauses off | clauses on (current) |
|---|---|---|
| none | 81 of 96 (84.4 %) | 79 of 96 (82.3 %) |
| about 200 tokens | 91 of 96 (94.8 %) | 90 of 96 (93.8 %) |

The term lost with clauses ("me sale" to "comes out") is a correct translation that the evaluator does not count.

False cuts in conversations (114 turns, synthetic audio): 99 without clauses, 100 with them.

### Wait after a complete sentence: 64 ms to 32 ms (T-81)

`endpoint_short_ms` is 32 ms by default. In an interleaved A/B on short clips (5 rounds x 8 clips, n=40 per arm) the median was 335 ms against 372 ms with 64 ms (-37 ms), p95 625 against 604 ms, false cuts in conversations 99 against 100. Translation does not change. To go back: `OPENTRADUCTOR_ENDPOINT_SHORT_MS=64`. This is a different clip set from the T-80 table above, whose 562 ms was measured with the 64 ms wait.

#### Re-measured on the 8-clip set (T-82)

Measured 2026-11-21 on code with a 32 ms wait and translation by clauses on, before the trailing-audio option of T-83 was in this repository. Same 8 clips, 8 interleaved rounds, 64 runs per arm, load average median 3.6, up to 4.9. The two arms differ only in `OPENTRADUCTOR_ENDPOINT_SHORT_MS`. The option to keep the first 96 ms of trailing silence for recognition (`OPENTRADUCTOR_ASR_TAIL_KEEP_MS`, T-83) was turned off in both arms. It is on by default now; see the next section.

| Arm | n | Median | p95 | Wait for end of speech (median) | Final recognition | Translation |
|---|---|---|---|---|---|---|
| wait 64 ms | 64 | 556 ms | 827 ms | 154 ms | 193 ms | 320 ms |
| wait 32 ms | 64 | 558 ms | 730 ms | 152 ms | 187 ms | 324 ms |

Paired by round, the median difference is +13 ms (n=8 rounds), that is, no gain at the median on these clips; p95 is 97 ms lower with 32 ms in this run. The median of one round goes from 285 to 614 ms in the 32 ms arm (it is 286 to 590 ms with 64 ms), so the 2 ms and 97 ms differences are inside the noise of a Mac that was running other work. The raw `artifacts/` folder is not in this repository.

### Trailing audio for recognition (T-83, `asr_tail_keep_ms` 96 by default)

When a phrase closes, the recognizer now gets the first 96 ms of the silent tail instead of none. The voice detector reports silence on soft endings, and Moonshine lost the last syllable ("policial" came out as "policia"). In an offline repro with moonshine-tiny + qwen, clip 1584 came out complete 0 of 4 times when the tail was discarded and 3 of 3 times when it was kept; clip 1516 came out complete in both cases (4 of 4 and 3 of 3). Two clips and a handful of runs: an indication, not a rate. To go back: `OPENTRADUCTOR_ASR_TAIL_KEEP_MS=0`.

Latency cost, measured 2026-11-21 on commit `5794853` (the code in this repository), same interleaved run as the T-82 table: same 8 clips, 8 rounds, 64 runs per arm, load average median 3.6, up to 5.9. Three arms ran one after the other in each round.

| Arm | n | Median | p95 | Wait for end of speech | Final recognition | Translation | Median paired by round |
|---|---|---|---|---|---|---|---|
| wait 64 ms, no trailing audio | 64 | 556 ms | 827 ms | 154 ms | 193 ms | 320 ms | reference |
| wait 32 ms, no trailing audio | 64 | 558 ms | 730 ms | 152 ms | 187 ms | 324 ms | +13 ms (n=8) |
| wait 32 ms + 96 ms of trailing audio (**current default**) | 64 | **572 ms** | 1039 ms | 153 ms | 190 ms | 344 ms | +66 ms (n=8) |

The 572 ms arm is 16 ms above the reference (14 ms above the 32 ms arm) at the median and 212 ms above at p95. The p95 of 1039 ms comes from a Mac with load average up to 5.9 and 64 runs, so the tail is a few runs: do not read it as a stable figure. The median of one round in that arm goes from 293 to 643 ms (rounds 5, 6 and 8 are above 580 ms and had load 3.5 to 5.9). The translation median is 20 to 24 ms higher in this arm; the cause was not investigated and it may be load. The raw `artifacts/` folder is not in this repository.

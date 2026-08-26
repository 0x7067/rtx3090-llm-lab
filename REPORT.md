# Tiel-Coder-35B-A3B vs qwen3.8-27B — local benchmark

First run 2026-08-25, re-run and corrected 2026-08-26.
Hardware: single RTX 3090 (24 GB), exclusive access for every measurement.

**The 2026-08-25 conclusions did not survive the re-run.** Two of the five rows
that justified replacing qwen were measuring reply formatting rather than code
quality, and the speed advantage reverses on the config that actually shipped.
The switch is still defensible, but not for the reasons originally given. Read
"What changed and why" before quoting anything here.

## The two deployments

- **qwen3.8-27b** — vLLM, W4A16 AutoRound, fp8 KV, MTP speculative decoding,
  140k ctx, `--max-num-seqs 8`
- **Tiel-Coder-35B-A3B** — llama.cpp (`full-cuda`), UD-Q4_K_S (20.9 GB),
  262k ctx on **one slot**, `K=q8_0 V=q4_0`, vision projector loaded

Both are the deployed profiles, not tuned-for-benchmark shapes. The 2026-08-25
run measured Tiel at 4 slots x 16k with f16 KV and no vision, which is not what
was put into production; every number below re-measures the shipped config.

## Results

All measured 2026-08-26 with one harness per row across both models, at a
12,288-token generation limit.

| metric | qwen3.8-27b | Tiel-35B-A3B | winner |
|---|---|---|---|
| Decode, single stream (tok/s) | 111.2 | **121.8** | Tiel, 1.10x |
| Prefill, ~6.8k uncached (tok/s) | **1269** | 188 | **qwen, 6.8x** |
| Aggregate, 4 concurrent (tok/s) | **278** | 114 | qwen, 2.4x |
| HumanEval pass@1 | 89.0% | **96.3%** | Tiel, +7.3pp |
| Multi-turn repair, 3 turns | 96.0% | **100%** | Tiel, +4pp |
| Multi-turn, turn 1 only | 90.4% | **97.6%** | Tiel, +7.2pp |
| Mean turns-to-solve | 1.07 | **1.02** | Tiel |
| Replies containing no code (of 164) | 15 | **1** | Tiel |
| MMLU-Pro, excl. truncated | 87.7% | 87.0% | tied |

## The prefill collapse

**Tiel prefills a 6.8k-token prompt at 188 tok/s. qwen does it at 1269 tok/s.**
Three samples each, medians, unique random preamble per request so neither stack
can serve it from cache. Time to first token on that prompt is 36.7 seconds for
Tiel against 5.3 seconds for qwen.

This is the single most important number in the re-run, and the 2026-08-25 report
had it backwards: it recorded Tiel at 2805 tok/s and called it a 2.16x win. That
measurement was real but taken on the 4-slot, 16k, f16-KV bench shape. The
shipped config prefills 15x slower than the shape that was benchmarked.

The cause is not established. The shipped config differs from the benchmarked one
in four ways at once — quantized KV, 262k context, one slot, projector loaded —
and no run isolates them. Quantized KV is the leading suspect: the deployment
manifest already records `V=q4_0` costing about 17% of decode speed, and
quantized cache writes plausibly cost far more on the prefill path.

**What this means operationally.** For an agent working against a large context,
prefill dominates. At 188 tok/s a 100k-token context takes about nine minutes
before the first token appears. The decode advantage of 1.10x cannot pay that
back. Anyone choosing between these deployments for long-context agent work
should treat this row as decisive until it is explained.

Settling it needs one GPU window and three configurations: shipped, shipped with
f16 KV, and shipped at 65k context. That was not run.

## What changed and why

### The token limit invalidated two rows

Both quality harnesses sent `max_tokens=6144`. A reply that ran out of budget
arrived with no fenced code block, and `bench_quality.py` scores an empty
extraction as a failure. The two models hit that wall at very different rates,
so the original pass@1 and multi-turn rows partly measured truncation.

`bench_mmlu.py` had already hit this during the first run and doubled its limit
to 12,288. That fix was never applied to the other two harnesses. It has been
now, and both harnesses record `finish_reason` and the full reply text so the
cause is visible instead of inferred.

| | 6,144 limit | 12,288 limit |
|---|---|---|
| qwen HumanEval | 140/164 (85.4%), 22 no code | 146/164 (89.0%), 15 no code |
| Tiel HumanEval | 152/164 (92.7%), 4 no code | 158/164 (96.3%), 1 no code |
| qwen no-code repair turns | 32 | 24 |
| Tiel no-code repair turns | 3 | 1 |

Every remaining no-code reply is now attributed: all 15 of qwen's and Tiel's one
carry `finish_reason: length`. For qwen both `content` and `reasoning_content`
came back empty, so vLLM returned nothing at all on truncation — this is genuine
truncation, not the harness reading the wrong field.

### The HumanEval gap is the same number for a different reason

+7.3pp is what the invalidated row reported and what the corrected row reports.
A reader who remembers the first table will assume nothing changed. What changed
is that the gap is now attributable. At a matched 12,288-token budget, qwen fails
to emit code on 15 of 164 problems and Tiel on 1. That is a reproducible
operational difference under identical conditions, not an artifact of a limit
that happened to bind one model harder.

For an agent, a reply with no code is a failed request whatever the cause, so the
raw row is the one to use.

Excluding truncated replies from both sides gives qwen 146/149 (98.0%) and Tiel
158/163 (96.9%). **Do not lead with that.** It drops 15 of qwen's problems
against 1 of Tiel's, and truncation preferentially hits the problems that provoke
the longest reasoning — the hard tail. The reversal is an artifact of the
exclusion. It is reported here only because the MMLU-Pro section sets that
precedent and a reader will ask.

### Multi-turn measured formatting, and still does

This is the finding that survived the re-run unchanged. Split the repair-loop
replies by whether they contained code at all:

| | qwen3.8 | Tiel |
|---|---|---|
| replies containing code | 120 | 127 |
| of those, the fix passed | **120 (100%)** | 125 (98.4%) |
| replies with no code | 24 | 1 |
| tasks left unsolved | 5 | 0 |

**qwen submitted code 120 times and was correct 120 times, at both token
limits.** It has still never proposed a wrong fix in this test. Tiel proposed two
(down from four at the lower limit) and repaired them on turn 2. All five of
qwen's unsolved tasks are tasks that returned prose on all three turns, so no
candidate was ever executed for them.

The loop treats a no-code reply as a distinct outcome and re-prompts, but that
re-prompt consumes one of the three turns. So 96.0% vs 100% is five tasks that
produced no code, not five wrong fixes; 90.4% vs 97.6% on turn one is largely
arithmetic; and mean turns-to-solve counts re-prompts for formatting as repair
iterations.

Do not invert the conclusion either. qwen's 120 coded replies exclude the tasks
where it answered with prose, which may be the harder ones. The split shows these
rows do not measure repair quality — not that qwen repairs better.

### The decode figure was always noisy

The 2026-08-25 report gave qwen 86.8 tok/s. That is the correct median of its
three samples, which were 109.4, 84.6 and 86.8 — a 29% spread within one run.
The 2026-08-26 samples are 111.2, 104.2 and 113.8. qwen runs MTP speculative
decoding, so its decode rate depends on draft acceptance and varies with content.
Tiel has no draft model and its samples are tight: 122.0, 121.9, 121.5.

Three samples of a bimodal quantity is too few. The 1.76x decode advantage in the
original table came from comparing Tiel's tight 152.8 against a median drawn from
qwen's low mode.

## Method

**Speed.** `bench_speed.py`. Every request carries a unique random preamble, so
neither prefix cache can inflate results. Prefill measured on a ~6.8k-token
prompt. Decode and prefill are medians of three; concurrency is one run of four
simultaneous requests.

**Single-shot quality.** All 164 HumanEval problems, greedy, generated code
executed in a network-less container. `bench_quality.py` extracts the longest
fenced block and falls back to the whole reply when there is no fenced block.

**Multi-turn.** 3 deterministic bugs seeded into each canonical solution with a
fixed RNG, so both models saw byte-identical broken code; the same
`mutants.jsonl` was reused for the re-run. Mutants were verified to actually fail
before inclusion: 125 of 134 survived. Each model got up to 3 turns, with the
candidate re-executed between turns and the real stderr appended to the same
conversation. This harness takes the **last** fenced block, not the longest,
because repair replies often quote the buggy code before the fix.

Difficulty was calibrated first. Single-operator bugs were fixed 5/5 on turn one,
so three simultaneous bugs were used instead to force iteration.

**The calibration did not work, and the original report claimed it had.** It
argued that qwen needing a second or third turn on 18 tasks proved the loop
engaged. Those 18 are exactly the tasks where qwen answered turn 1 with prose:
they reached turn 2 by submitting no code, not by submitting a wrong fix. The
only genuine repair iteration in either run was Tiel's handful of wrong fixes.

## MMLU-Pro (knowledge recall)

Measured 2026-08-25 and not re-run; it already used the 12,288-token limit, so
the defect corrected above does not apply to it. 350 questions, stratified 25 per
category across all 14, fixed seed so both models saw the identical draw.
Chain-of-thought prompting, greedy.

| | Tiel | qwen3.8 |
|---|---|---|
| raw | **82.0%** | 79.1% |
| excluding truncated | 87.0% | **87.7%** |
| truncated replies | 20 | 34 |

**Read the second row.** Tiel's 2.9-point raw lead is mostly an artifact of qwen
truncating more often. Excluding truncated replies from both, they are within 0.7
points, well inside noise at n=350 (±~3.5pp at 95% confidence).

The two models are **statistically tied on knowledge recall.** That matters
because it was the open risk: the model card puts Tiel 10.3 points behind, which
argued for keeping qwen. Measured against this qwen build, on this hardware, with
an identical question set, that gap does not appear — the card's comparison is
against a different model.

### Per category

| category | Tiel | qwen3.8 | delta |
|---|---|---|---|
| other | 72% | 56% | +16 |
| philosophy | 88% | 76% | +12 |
| health | 84% | 76% | +8 |
| math | 92% | 84% | +8 |
| computer science | 88% | 84% | +4 |
| history | 72% | 68% | +4 |
| law | 72% | 68% | +4 |
| psychology | 72% | 68% | +4 |
| economics | 96% | 96% | +0 |
| physics | 92% | 92% | +0 |
| biology | 92% | 96% | -4 |
| chemistry | 88% | 92% | -4 |
| engineering | 60% | 64% | -4 |
| business | 80% | 88% | -8 |

Tiel leads in 8 categories, trails in 4, ties in 2. Do not over-read individual
rows: each category is 25 questions, so one question moves it 4 points.

## KV-cache quantization

Measured 2026-08-25, to size Tiel for the context the deployment needed. Raw data
in `work_q8/`, `work_q4/`, `work_mixed/`.

| KV cache | server shape | HumanEval | n |
|---|---|---|---|
| f16 | 65k, 4 slots, no vision | 93.3% (56/60) | 60 |
| K=q8_0 V=q8_0 | 262k, 2 slots, no vision | 93.3% (56/60) | 60 |
| K=q4_0 V=q4_0 | 262k, 2 slots, vision | 90.0% (54/60) | 60 |
| K=q8_0 V=q4_0 | 240k, 1 slot, vision | 95.7% (157/164) | 164 |

The three 60-problem rows are the first 60 HumanEval problems, so they compare
only with each other — and they differ in context depth, slot count and projector
as well as in KV type. Symmetric q8_0 scored identically to f16 on all 60
problems, task for task. Symmetric q4_0 lost HumanEval/19 and /57 and gained
nothing, which is why K stays at q8_0 and only V drops to q4_0.

These runs measured quality only. Given the prefill collapse above, the KV
setting now needs a speed measurement it never got.

## Concurrency and parallelism

The concurrency row is not matched: qwen serves 8 sequences, Tiel serves 1. That
is the deployed reality on both sides rather than a harness choice, but it means
the 2.4x is a property of the configurations, not the engines.

Tiel does not have to stay at one slot. `PARALLELIZATION.md` measures the
alternatives: `--kv-unified --parallel 4` holds the full 262k depth per caller,
costs 188 MiB, and raises four-way aggregate throughput from 119 to 195 tok/s.
That closes about half the gap to qwen's 278 and costs an idle caller nothing.

## Caveats

- **Prefill is unexplained.** See above. It is the largest open item.
- **TTFT on short prompts is not comparable.** The harness marks first-token on
  the first reasoning delta, and vLLM's `qwen3` reasoning parser and llama.cpp's
  `--jinja` split reasoning from content differently.
- **Run-to-run instability is about ±3pp on HumanEval** and much larger on qwen's
  decode. Read any single figure as approximate.
- HumanEval is saturated at this level and measures short self-contained
  functions. Neither number predicts performance on your actual codebase.
- **Nothing here tests vision.** The projector is loaded in the shipped config
  and was loaded for the later measurements, but was never exercised.
- qwen ran as a standalone container reproducing its former k8s profile — same
  image, env, host paths. The Deployment was only scaled to 0, never edited.

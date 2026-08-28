# Tiel-Coder-35B-A3B vs qwen3.8-27B — local benchmark

First run 2026-08-25, re-run and corrected 2026-08-26.
Hardware: single RTX 3090 (24 GB), exclusive access for every measurement.

**The 2026-08-25 conclusions did not survive the re-run.** Two of the five rows
that justified replacing qwen were measuring reply formatting rather than code
quality, and the speed advantage reversed on the config that actually shipped.
The switch is still right, but not for the reasons originally given, and the
configuration that shipped on 2026-08-25 was wrong.

**The actionable finding: `--cache-type-v q4_0` costs 14x on prefill.** Setting
it to `q8_0` at 184,320 context recovers 2815 tok/s prefill and 149.7 tok/s
decode, uses less VRAM than the config it replaced, and still serves more
context than qwen did. See "The prefill collapse".

**Tiel was rolled back on 2026-08-26.** docker-services `08bf9af` restored the
qwen3.8-27b vLLM profile, and that is what serves traffic now. Everything below
describes Tiel as it was configured while deployed, 2026-08-25 to 2026-08-26.
Read it as a record of what was measured, not as a description of production.

The MTP build (`Tiel-Coder-35B-A3B-MTP-UD-Q4_K_S.gguf`) and the smaller
`IQ4_XS` tier are both downloaded to the model volume, and `sweep_mtp.sh` is
committed and ready. Neither was ever deployed: the speculative-decoding sweep
needs the card, and the card is serving qwen.

**The V=q8_0 change was live before the rollback.** It shipped on 2026-08-26 as docker-services `91f369e`,
and the deployment now runs `-c 184320 --parallel 1 -fa on K=q8_0 V=q8_0` with
the projector loaded. Read "Tiel, shipped" below as the *former* config
(`262k / V=q4_0`) — that is what the re-run measured — and the
`184,320 / V=q8_0` column as what serves traffic today.

## 2026-08-27 addendum: qwen's failures were a lost config default, now fixed

An overnight follow-up (see `NIGHT_PLAN.md`) root-caused qwen's headline
weaknesses and shipped the fix (docker-services `c39adf3`). Two corrections to
this report and one production change:

**1. The "empty" replies were never empty.** vLLM 0.27.1 renamed the output
field: the CoT arrives in `message.reasoning`, and `reasoning_content` does not
exist on responses (vLLM #33402). Both harnesses read the dead key, so every
`reasoning_len: 0` in the recorded qwen data is a measurement artifact. The
replies that scored as failures carried a full reasoning trace and no answer.
Harnesses now coalesce both keys.

**2. The failures were the template's xhigh fallback, not the model.** The
llama-swap deployment pinned `reasoning_effort: medium` server-side; the vLLM
migration dropped the pin, and `chat_template.jinja` falls back to `xhigh`. At
xhigh the model thinks past the 12,288-token budget on hard problems and
returns `content: null` with `finish_reason: length`. All 15 single-shot
no-code tasks and all 24 no-code repair turns trace to this (the two
`finish_reason: stop` empties also vanish at medium and did not reproduce in a
128-residue prefix-cache sweep or a 6-cycle KV-offload eviction/restore probe).

**Re-measured at `reasoning_effort: medium`** (same bench container, same
harnesses, greedy, 12,288-token limit):

| metric | qwen at xhigh (this report) | qwen at medium | Tiel |
|---|---|---|---|
| HumanEval pass@1 | 146/164 (89.0%) | **162/164 (98.8%)** | 158/163 (96.3%) |
| Multi-turn repair, 3 turns | 96.0% | **100%** | 100% |
| Multi-turn, turn 1 only | 90.4% | **99.2%** | 97.6% |
| Replies with no code | 15 | **0** | 1 |

Zero regressions: no task that passed at xhigh fails at medium. The two
remaining failures (HumanEval/50, /145) also fail at xhigh, and /145 fails on
Tiel's stack too. **Every quality row that favoured Tiel reverses or ties.**

**3. Not shipped — xhigh is intentional.** The medium pin went live briefly as
docker-services `c39adf3` and was reverted the same morning (`9a848c6`) on the
owner's decision: the xhigh default stays. The measurements above stand as
data. Clients that need guaranteed answers on a bounded budget can opt in per
request with `chat_template_kwargs: {"reasoning_effort":"medium"}` or
`thinking_token_budget` — both verified working on this deployment, and the
server-side flag (`--default-chat-template-kwargs`, verified to survive
`start_qwen.sh`'s unquoted EXTRA_ARGS expansion) remains available if that
decision ever changes.

**Speed: no safe lever exists on this stack.** Measured or re-confirmed
overnight: async scheduling guards open corruption bug #51571 (must stay off);
MTP k=4 still crashes (no upstream fix through 0.28.0); a drafter backend
split at k=3 OOMs on first inference at `GPU_UTIL=0.95` and costs 11.6% of the
KV pool; vLLM 0.28.0's fused GDN decode kernel gates on a v/k head ratio of 8
(this model is 3) and 0.28.0 cannot load this checkpoint anyway. Decode
baseline at production sampling: median 89.2 tok/s, draft acceptance 0.574.
A correctness backport (vLLM #51812, GDN spec-gate gather; affects sampled
traffic) is staged in `patches-night/` for the next image rebuild.

## The two deployments

- **qwen3.8-27b** — vLLM, W4A16 AutoRound, fp8 KV, MTP speculative decoding,
  140k ctx, `--max-num-seqs 8`
- **Tiel-Coder-35B-A3B** — llama.cpp (`full-cuda`), UD-Q4_K_S (20.9 GB),
  262k ctx on **one slot**, `K=q8_0 V=q4_0`, vision projector loaded. This is
  the profile the re-run measured. It was replaced on 2026-08-26 by
  `184,320 / V=q8_0` on this report's recommendation.

Both are the deployed profiles, not tuned-for-benchmark shapes. The 2026-08-25
run measured Tiel at 4 slots x 16k with f16 KV and no vision, which is not what
was put into production; every number below re-measures the shipped config.

## Results

All measured 2026-08-26 with one harness per row across both models, at a
12,288-token generation limit.

| metric | qwen3.8-27b | Tiel-35B-A3B | winner |
|---|---|---|---|
| Decode, single stream (tok/s) | 111.2 | **121.8** | Tiel, 1.10x |
| Prefill, ~6.8k uncached (tok/s) | **1269** | 188 | **qwen, 6.8x** † |
| Aggregate, 4 concurrent (tok/s) | 278 | 114 | qwen, 2.4x ‡ |
| HumanEval pass@1 | 89.0% | **96.3%** | Tiel, +7.3pp |
| Multi-turn repair, 3 turns | 96.0% | **100%** | Tiel, +4pp |
| Multi-turn, turn 1 only | 90.4% | **97.6%** | Tiel, +7.2pp |
| Mean turns-to-solve | 1.07 | **1.02** | Tiel |
| Replies containing no code (of 164) | 15 | **1** | Tiel |
| MMLU-Pro, excl. truncated | 87.7% | 87.0% | tied |

† Both speed rows are properties of the shipped configuration, not the engines.
Changing `--cache-type-v` to `q8_0` takes Tiel's prefill to 2815 tok/s and its
decode to 149.7.

‡ Superseded. Tiel served one slot when this was measured; qwen served eight.
Tiel now runs four unified slots, and its four-way aggregate on the shipped
config measures 257-271 tok/s against qwen's 278 — parity, not a 2.4x deficit.
See `PARALLELIZATION.md`.

## The prefill collapse

**Tiel prefills a 6.8k-token prompt at 188 tok/s. qwen does it at 1269 tok/s.**
Three samples each, medians, unique random preamble per request so neither stack
can serve it from cache. Time to first token on that prompt is 36.7 seconds for
Tiel against 5.3 seconds for qwen.

This is the single most important number in the re-run, and the 2026-08-25 report
had it backwards: it recorded Tiel at 2805 tok/s and called it a 2.16x win. That
measurement was real but taken on the 4-slot, 16k, f16-KV bench shape. The
shipped config prefills 15x slower than the shape that was benchmarked.

**The cause is now isolated: `--cache-type-v q4_0`.** Nothing else. The shipped
config differed from the benchmarked one in four ways, so each was varied
separately (`sweep_prefill.sh`, raw data in `results_prefill_isolation.json`):

| | ctx | K | V | vision | VRAM | decode | prefill |
|---|---|---|---|---|---|---|---|
| A | 262k | q8_0 | q4_0 | yes | 23,050 MiB | 121.8 | 200.2 |
| B | 262k | q8_0 | q4_0 | no | 21,938 MiB | 121.8 | 199.1 |
| C | 65k | q8_0 | q4_0 | yes | 21,490 MiB | 120.8 | 188.4 |
| D | 65k | f16 | f16 | no | 21,198 MiB | 151.7 | **2757.4** |
| E | 131k | q8_0 | q8_0 | yes | 22,694 MiB | 149.1 | **2792.5** |

A against B rules out the vision projector. A against C rules out context depth —
65k is no faster. D reproduces the 2026-08-25 figure, so that measurement was
sound. E is the discriminator: it differs from C only in the value cache, and
prefill goes from 188 to 2793 tok/s.

So this is not "quantized KV cache." K stays at q8_0 in row E and prefill is
fine. It is the **value** cache at 4 bits, and it costs about 14x on prefill and
about 18% on decode. The manifest recorded only the decode cost.

**What this means operationally.** For an agent working against a large context,
prefill dominates. At 188 tok/s a 100k-token context takes about nine minutes
before the first token appears; at 2793 tok/s it takes 36 seconds.

`V=q4_0` exists to buy the full 262k native context, and that is the real trade:
depth against prefill. Row E gives up depth to 131k — roughly what the vLLM
profile served — and in exchange costs *less* VRAM than the shipped config
(22,694 against 23,050 MiB), leaves more vision headroom (1,882 against 1,110
MiB), prefills 14x faster and decodes 22% faster. Quality should not move: the
weights are identical, and the KV-quantization sweep below found symmetric q8_0
scoring the same as f16 on all 60 problems, task for task.

### How much depth `V=q8_0` actually costs

131k was a guess. A depth ladder at `K=q8_0 V=q8_0` with vision loaded
(`results_depth_ladder.json`) shows speed is flat and only VRAM moves:

| ctx | VRAM at load | free | after a +416 MiB image | decode | prefill |
|---|---|---|---|---|---|
| 131,072 | 22,694 MiB | 1,882 | 1,466 | 149.1 | 2792.5 |
| 163,840 | 23,130 MiB | 1,446 | 1,030 | 151.0 | 2833.1 |
| **184,320** | **23,404 MiB** | **1,172** | **756** | **149.7** | **2815.0** |
| 196,608 | 23,566 MiB | 1,010 | 594 | 148.8 | 2780.2 |
| 212,992 | 23,784 MiB | 792 | 376 | 149.2 | 2785.8 |

Every row loads and every row runs at full speed, so depth buys nothing but
memory pressure. The binding constraint is the vision peak: the manifest
measured a 3000x2000 image costing +416 MiB on top of load, and recorded 682 MiB
of headroom as "too tight".

**184,320 is the recommendation.** It leaves 1,172 MiB at load — more than the
1,110 MiB the shipped config has today — and about 756 MiB under a worst-case
image. 212,992 leaves 376 MiB after an image, below the margin already judged
unsafe, so the practical ceiling is somewhere between 192k and 208k.

**The +416 MiB image cost was re-measured on the shipped config and holds
exactly, at every slot count** (`results_vision_slots.json`). It also does not
multiply with concurrent images:

| slots | load | one image | N concurrent images | replies |
|---|---|---|---|---|
| 1 | 23,404 | 23,820 | 23,824 | 1/1 |
| 2 | 23,466 | 23,882 | 23,886 | 2/2 |
| 4 | 23,592 | 24,008 | **24,018** | 4/4 |

Four simultaneous 3000x2000 requests peaked 10 MiB above one, and all four
returned. The image cost is 416 MiB in every row, the same figure the manifest
recorded at 262k. This was the gate on raising `--parallel`, and it cleared.

That leaves 756 MiB free at peak on a 24,576 MiB card, which is the margin the
row above predicted. The number most likely to invalidate the ladder did not, so
184,320 is confirmed rather than assumed.

**What the ceiling is has still not been established.** The manifest's rejected
config left 682 MiB *at load*, which is 266 after an image; its accepted one left
1,110 at peak. Nothing pins the safe threshold anywhere between those two, so
196,608 (594 at peak) and 212,992 (376) are not ruled out by that comparison —
they are simply closer to the rejected end. 184,320 is the recommendation
because it has the most measured headroom, not because the alternatives are
known to fail.

Measured under docker. The deployment reads the same figures under containerd —
23,592 MiB at rest on four slots, identical to the sweep — so no per-runtime
adjustment applies. An earlier 24 MiB gap at one slot was a request in flight.

The shipped four-slot config therefore peaks at 24,018 MiB of 24,576 under the
worst case, about 558 MiB spare, down from about 732 at one slot.

**Verified on the live deployment** (`verify_vision_concurrency.sh`), which also
settles the overlap question the sweep left open. Four concurrent 3000x2000
requests against the shipped four-slot config:

- 23,618 MiB at rest to a 24,020 MiB peak — **402 MiB for all four images**,
  not 4 x 416
- 4/4 replies, pod still `Running` with 0 restarts
- 556 MiB free at peak

The encodes did overlap. All four slots launched within 527 ms of each other and
released within 6 ms, each holding its slot for about 20 seconds, so all four
were resident together for essentially the whole window rather than serialised.

One thing to know: the allocation does not come back. VRAM stays at 24,020 MiB
after the requests finish. The image cost is a one-time step on first use rather
than per-request growth, so 556 MiB is the steady state once any image has been
served, not just a transient peak. That is consistent with a single projector
buffer allocated once and shared by every slot, which would also explain why it
does not multiply — though that reading is inferred from the allocation
behaviour, not read out of the source.

**Full depth per caller is real, not just a log line.** A 66,259-token prompt
was served on the four-slot config with `truncated = 0` and prefilled at 2,975
tok/s. A partitioned four-slot server caps `n_ctx_slot` at 46,080, so that
request could not have completed intact on it.

**But the depth is a shared budget, and overrunning it drops every request in
flight.** One caller can reach 184,320 tokens only when the others are idle.
Four concurrent 66k requests crossed the pool at 184,976 tokens and the server
returned HTTP 500 `Context size has been exceeded.` to all four — including the
slot that had already prefilled 66k and the slot that had reached 179. The
server stayed up. See `test_pool_limit.sh` and `PARALLELIZATION.md`.

With four slots that is roughly **46k tokens per session** when all four are
deep at the same time. Coding-agent sessions routinely exceed that, so this is
the operational limit to plan around, not the 184,320 figure.

**The deployment therefore runs two slots, not four** (docker-services
`00dcf03`). Two sessions get about 92k each, which is the range agent sessions
actually work in. Verified: two concurrent sessions of ~85k tokens, 170,948 live
tokens against the pool, both `HTTP 200` with `truncated = 0`.

The cost is about 20-25% of peak aggregate throughput — 208.9 tok/s across two
concurrent requests against 257-271 across four. Offering more requests than
slots is safe: four requests against two slots all completed at 203.7 tok/s,
queueing rather than failing. Single-stream decode is unchanged at 134.5.

### What the change is worth

| metric | qwen3.8 | Tiel, shipped | Tiel at 184,320 / V=q8_0 |
|---|---|---|---|
| Prefill, ~6.8k (tok/s) | 1269 | 200 | **2815** (2.2x qwen) |
| Decode, single stream (tok/s) | 111.2 | 121.8 | **149.7** (1.35x qwen) |
| Context | 140k | 262k | 180k |
| VRAM headroom at load | — | 1,110 MiB | **1,172 MiB** |

Every speed row that favoured qwen reverses, the context still exceeds what the
vLLM profile served, and it costs less memory than what runs today. The only
thing given up is native 262k depth — which at 200 tok/s was not reachable in
practice anyway: a 200k-token cold context needs about 17 minutes to prefill.

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

Every single-shot no-code reply is attributed: all 15 of qwen's and Tiel's one
carry `finish_reason: length`. For qwen both `content` and `reasoning_content`
came back empty, so vLLM returned nothing at all on truncation — this is genuine
truncation, not the harness reading the wrong field.

**The multi-turn side is not fully attributed, and this section used to claim it
was.** Of qwen's 24 no-code repair turns, 22 carry `finish_reason: length` and
**two carry `finish_reason: stop` with an empty reply** (HumanEval/99 turn 2,
HumanEval/134 turn 1). Those two are not truncation. An empty body on a clean
stop is a serving defect rather than a budget problem, and it is worth chasing on
the endpoint rather than filing under this benchmark.

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
limits.** It never proposed a wrong fix *in the repair test*. Tiel proposed two
(down from four at the lower limit) and repaired them on turn 2.

Do not widen that into "qwen never writes wrong code": in the single-shot
HumanEval run qwen submitted code that failed three times, on HumanEval/38, /50
and /160. The claim is scoped to this repair loop.

All five of qwen's unsolved tasks failed with no candidate ever executed. This
section used to call those replies "prose", carried over from the first run;
the recorded evidence contradicts it. 23 of the 24 no-code turns have
`raw_len: 0` — the replies were **empty**, not prose.

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

**Read the first row.** This section used to say "read the second row" and rest
its conclusion on the excluded-truncated comparison. That was inconsistent: the
HumanEval section above argues, correctly, that excluding truncated or no-code
replies is artifact-generating because truncation preferentially removes the hard
tail — and here it removes more of qwen's (34 against 20). The same objection
applies, so the exclusion cannot be the headline.

The conclusion does not need it. At n=350 with p≈0.8 the standard error on a
*difference* of proportions is about 3.0pp, so significance needs roughly 5.9pp
at 95% confidence. The raw gap is 2.9 points and the excluded gap is 0.7. Both
are inside the band, so **the two models are statistically tied on knowledge
recall** whichever row you take — and the raw row is the one to quote, since for
an agent a truncated reply is a failed request.

That matters because it was the open risk: the model card puts Tiel 10.3 points
behind, which argued for keeping qwen. Measured against this qwen build, on this
hardware, with an identical question set, that gap does not appear — the card's
comparison is against a different model.

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

**Tiel no longer stays at one slot.** `--kv-unified --parallel 4` shipped on
2026-08-26 (docker-services `d9b6f83`). Measured on the deployment before and
after, four-way aggregate went 140.7 -> 257.2 and 271.0 tok/s, about 1.9x, with
4/4 completions and single-stream decode unchanged. That closes the gap to
qwen's 278 rather than halving it, costs 188 MiB, and costs an idle caller
nothing. Every caller keeps the full 184,320 depth because the unified KV buffer
is shared rather than partitioned. See `PARALLELIZATION.md`.

## Quantization tier: IQ4_XS against Q4_K_S

Measured 2026-08-26 to decide whether the MTP build could use the smaller tier
(`results_quant_ppl.json`). It matters because IQ4_XS is 3.16 GB smaller, which
is roughly four times what the MTP head costs — so if the tiers were equal, MTP
would have been free in context terms.

They are not equal. Perplexity over 138 chunks of a code-heavy corpus
(HumanEval prompts with canonical solutions plus this repo's own sources), same
chunks in the same order, `n_ctx=512`:

| tier | size | PPL | vs best |
|---|---|---|---|
| **Q4_K_S** | 21.79 GB | **2.6985** | — |
| IQ4_XS | 18.63 GB | 2.7655 | +2.48% |

The gap is 0.067 against a measurement uncertainty of ±0.031, so about 2.2σ —
small but not noise.

The model card does not compare these two. It calls IQ4_XS "4-bit quality with
the most context headroom of any 4-bit tier" and Q4_K_S "tight 4-bit; useful
when Q4_K_XL leaves too little room", which reads as a recommendation for
IQ4_XS and is only a claim about headroom. Neither is the tier the card
benchmarked.

**What this is not.** KL divergence against the BF16 reference is the rigorous
way to score quantization damage, and it needs a 70 GB file that does not fit
here. Two quants' perplexity on the same text is directional rather than exact.
A 2.5% gap is large enough to act on and too small to put a number on in terms
of downstream task quality.

## Caveats

- **Prefill is unexplained.** See above. It is the largest open item.
- **TTFT on short prompts is not comparable.** The harness marks first-token on
  the first reasoning delta, and vLLM's `qwen3` reasoning parser and llama.cpp's
  `--jinja` split reasoning from content differently.
- **Run-to-run instability is about ±3pp on HumanEval** and much larger on qwen's
  decode. Read any single figure as approximate.
- HumanEval is saturated at this level and measures short self-contained
  functions. Neither number predicts performance on your actual codebase.
- **Vision is exercised only for VRAM.** One 3000x2000 image was put through
  the shipped config to measure its memory peak, and the reply came back, but
  no vision output was checked for quality.
- qwen ran as a standalone container reproducing its former k8s profile — same
  image, env, host paths. The Deployment was only scaled to 0, never edited.

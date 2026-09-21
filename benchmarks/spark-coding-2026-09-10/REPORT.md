# Spark coding pilot — 2026-09-10

A short system instruction improved coding results within an 8k output budget:
**4/14 → 9/14 on development, then 4/6 → 6/6 on reserved tasks**. Every
previously passing case remained passing in these comparisons. Tool-call
replay stayed at **12/12 development and 4/4 reserved**.

The selected [coding profile](coding-profile.json) retains temperature 1.0,
top_p 0.95, top_k 0 and thinking enabled. Its change is this system instruction:

> You are a practical coding assistant. Think briefly and check the stated contract, then give the requested answer. For code-only tasks, return one complete code block with no examples or explanation. Avoid revisiting decisions once resolved.

This is an experimental request-level profile. Merge its `request_defaults`
into a coding request and include `system_instruction` as a system message
before the task. **Serving defaults and weights were not changed.** The
393,216-token context, FP16 KV cache, Q8_0 weights and lack of speculative
decoding remain as deployed. No weight training ran in this phase.

## Development results

| Condition | Coding | Tool replay | Coding truncations | Median coding request |
| --- | ---: | ---: | ---: | ---: |
| Original prompt, temperature 1 | 4/14 | 12/12 | 9 | 66.94s |
| Temperature 0.2 | 6/14 | 12/12 | 7 | 60.86s |
| Non-thinking, temperature 1 | 3/14 | 12/12 | 0 | 1.78s |
| Non-thinking, temperature 0.2 | 2/14 | 12/12 | 0 | 1.73s |
| **Brief system instruction, temperature 1** | **9/14** | **12/12** | **4** | **26.95s** |

The selected development pass fraction rose from 28.6% to 64.3%: **+35.7
percentage points**, or +125% relative to this baseline. It added passes for
flattening nested data, byte-limited batching, retry budgets, redaction and
JSONL parsing, with no lost passes. Byte-limited batching had previously
returned finished but incorrect code that accepted an oversized item; the
new response correctly rejected it.

Median coding request time fell 59.7%; mean time fell from 51.36s to 34.66s
(32.5%). These are warm-endpoint observations reflecting different amounts
of generated text. They do not measure a change in GPU decode speed.

The profile was locked before any reserved-task inference. The selection rule
required all development tool cases to pass and chose the highest coding count;
incomplete diagnostics and one-case probes were ineligible. No tuning followed
the reserved results. The lock and plan hashes are in
[selection.json](artifacts/selection.json).

## Reserved comparison

| Condition | Coding | Tool replay | Coding truncations | Median coding request |
| --- | ---: | ---: | ---: | ---: |
| Original prompt | 4/6 | 4/4 | 2 | 23.96s |
| Brief system instruction | 6/6 | 4/4 | 0 | 13.15s |

The prompt added correct range coalescing and configuration merging without
losing a previously passing task. Coding rose from 66.7% to 100% on these six
cases: +33.3 percentage points, or +50% relative. Median request time was 45.1%
lower; mean time fell from 32.73s to 13.47s. Six reserved cases are a small pilot,
not enough to estimate broad repository-level accuracy.

## Live editing and test feedback

Three additional synthetic episodes exercised actual read, write and test
tools. Their final source was graded using separate immutable assertions.
These compare the original thinking mode with non-thinking at temperature 1;
they are a separate experiment from the selected brief-prompt comparison.

| Bug | Original thinking mode | Non-thinking mode |
| --- | --- | --- |
| Repeated calls for falsey cached results | Pass; 4.77s API time | Pass; 2.50s |
| Chunking a one-pass iterable | Pass; 33.42s | Pass; 3.63s |
| Lost days, sign and fractions in elapsed seconds | Stalled before editing; 67.51s | Pass; 3.00s |

Non-thinking completed all three read/write/test workflows and final tests;
the original mode completed two. API-time sums exclude tool/Docker overhead.
This suggests a useful quick-edit mode when tests provide feedback, but its
weaker standalone scores argue against a universal non-thinking default.

## Diagnostics and remaining failures

- A 2,048-token reasoning cap forced an earlier answer, but did not consistently
  produce correct or concise final code. That trial stopped after 10/26 completed
  requests plus one interrupted request. It remains an incomplete diagnostic,
  with no full-suite score assigned.
- Increasing the original SSE task's allowance to 32,768 tokens still produced
  no final code after 285.16s. The remaining request and grader fields were
  identical to its 8k baseline. More output allowance alone did not fix this case.
- The brief-prompt SSE probe passed once, then the identical request and seed
  truncated in the full run. Both results are retained. Cache histories differed,
  but the cause of the divergence was not established; a fixed seed did not
  guarantee identical output.
- The selected development profile still truncated on SSE framing, histogram
  buckets, rate limiting and nested-key differences. Its topological-sort answer
  finished but failed with an unhashable-list error. These remain failures.

## Scope and next training step

All matched quality comparisons used **8,192 output tokens per request** and
seed 42, one sample per case. The deployed service still allows 32,768 output
tokens. The test set consists of synthetic Python contracts and tool replay,
split by task/episode before inference. It is not a standard coding benchmark
or a full repository issue benchmark. The live editing fixtures use single
source files in memory. Generated code ran in the pinned Docker sandbox with
no network or host mounts.

These results support trying the prompt on coding requests. Larger, repeated
evaluations on real repositories are needed before a general quality claim or
a global default change. The same-seed divergence is another reason to keep
this recommendation experimental.

[LORA.md](LORA.md) and [lora-pilot.yaml](lora-pilot.yaml) prepare the next phase:
rank-8 BF16 LoRA through the publisher's pinned LLaMA-Factory fork, short
sequences, checkpointing and AdamW. The environment and GPU memory requirement
still need a training smoke test. Use fresh, licensed, test-verified examples
of contract handling and successful edit/test workflows; keep this entire
benchmark out of training. Target finished-code errors separately from runaway
reasoning, which prompting already mitigated on several tasks.

## Verification and evidence

The paired harness's unit suite completed with 12 passes and one opt-in test
skipped. All five new tool-boundary and fixture tests passed, including actual
Docker execution. Original fixture bugs fail visible and final assertions;
independent reference fixes pass both. Generated successful patches were
inspected. The analysis verifies complete observations, unchanged graders and
that removing only the selected instruction makes each paired request identical.

[analysis.json](artifacts/analysis.json) contains case-level scores, timings and
paired wins/losses. Frozen plans, manifests, compressed raw responses, selection
records and environment details are under [artifacts](artifacts/).
[EXPERIMENTS.md](EXPERIMENTS.md) records the decisions, and [README.md](README.md)
provides reproduction commands.

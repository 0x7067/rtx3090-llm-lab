# Follow-up research: speed and Strata quality

Research date: 2026-10-01. This document turns the five-arm baseline into a
ranked experiment plan. It does not claim speedups that were not measured on
this host.

## Starting point

The frozen run used an RTX 3090, a common 16,384-token output allowance, and
the same 210–1350 MHz/+100 VF/370 W GPU profile. The authored suite had one
measured run per arm, so its rates are useful baselines rather than estimates
with confidence intervals.

| Arm | Quality | Medium prefill | Medium decode | Main constraint |
| --- | ---: | ---: | ---: | --- |
| ThinkingCap Q4_K_S, llama.cpp | 54/54 | 1,050 tok/s | 38.95 tok/s | No matched draft model; already full GPU offload, flash attention, Q8 K/V |
| GSQ/RCO IQ3_S, llama.cpp | 53/54 | 1,047 tok/s | 36.73 tok/s | Same engine path; one coding miss |
| W4A16, vLLM | 52/54 | not reported | not reported | 98,304-token window rejected both 114K retrieval cases |
| OrcaSAQ2 EXL3, vLLM | 52/54 | not reported | not reported | Two coding misses; MTP depth 2 already enabled |
| Flash Next 125B IQ2_XS, Strata | 48.67/54 | 1,531 tok/s | 49.60 tok/s | Two 16K coding truncations plus four answer-quality misses |

The GPU profile is already the validated thermal operating point. The earlier
ThinkingCap run exceeded the 101°C VRAM guard, while the final profile did not.
More clock or memory overclocking is therefore a poor first speed experiment.

## Ranked experiments

### 1. Calibrate and profile Strata on the deployed artifact

This is the highest-confidence opportunity because it follows the publisher's
documented path and preserves the exact model, quant, runtime commit, context,
KV type, and sampling policy.

Run Strata 0.1.30's calibration on the same container limits and 150K context.
Its calibration searches `--pcie-frac`, `--spec-min-p`, and `--pool-workers`
and retains changes only when the measured gain exceeds 3%. Then collect
routing traces from a representative mixture of this suite and production
prompts with `--dump-routing`, and build a workload-specific expert profile
with `tools/make_profile.py`. Compare the generated profile with the current
automatic resident-expert selection.

The v0.1.30 release reports that its new 1K–4K streaming threshold improves
prompt processing by 17–28%, and that low-RAM resident mode improved a cold
prompt example from 150 to 1,197 tok/s with 26% faster answer generation.
Those are upstream examples, not additional gains available on top of this
baseline: the measured arm already runs v0.1.30 in low-RAM resident mode.
Calibration and a workload-specific profile are the remaining supported
host-specific paths.

Measure:

- cold and warm TTFT, prefill rate, decode rate, and wall time;
- draft acceptance, expert-cache hit rate, PCIe traffic, GPU/host memory, and
  memory PSI;
- raw responses and the frozen 54-case score, using at least three repeats per
  configuration and identical request seeds;
- one static-residency diagnostic with `--adapt-every 100000`, because the
  default adaptive expert residency can make even seeded sampling diverge.

Sources: [Strata v0.1.30 release notes](https://github.com/Niko1221/Strata/releases/tag/v0.1.30),
[pinned Strata details and tuning guide](https://github.com/Niko1221/Strata/blob/30ec18ec7094550fcc594fd948220d511d80464e/docs/DETAILS.md).

### 2. Test Strata's smaller English/code draft vocabulary

For English and code traffic, compare the publisher's reduced draft
vocabulary with the current default. The Strata guide reports about 110 MiB
less memory and a 1–2% English speed improvement. Treat that number as a prior,
not an expected result. Retain it only if all frozen quality gates pass and
non-English or structured-output traffic that matters in production is tested
separately.

This is a small expected gain, but it is cheap and changes the speculative
path rather than the target weights. Record acceptance distributions, not just
mean decode rate.

Source: [pinned Strata details and tuning guide](https://github.com/Niko1221/Strata/blob/30ec18ec7094550fcc594fd948220d511d80464e/docs/DETAILS.md).

Transparent huge pages (THP) is a low-confidence field lead, not a documented
Strata 0.1.30 optimization. Linux documents THP as automatic promotion of
eligible anonymous mappings, and NVIDIA's `nvbandwidth` explicitly suggests
`madvise` mode when its own huge-page allocation is requested. I found no
publisher claim that Strata marks its expert arena with `MADV_HUGEPAGE`, and no
reproducible Strata A/B attributable to the reported “StrataGP” lead. Check the
current mode and `AnonHugePages` for the Strata process read-only first. Only if
the arena is eligible should THP be a separate cold-start, page-fault, TTFT,
decode, and tail-latency A/B. Do not roll a host-wide `always` setting into the
main experiment: compaction can introduce latency, and any result would mix an
OS memory-policy change with engine tuning.

Sources: [Linux THP documentation](https://github.com/torvalds/linux/blob/master/Documentation/admin-guide/mm/transhuge.rst),
[NVIDIA nvbandwidth THP check](https://github.com/NVIDIA/nvbandwidth/blob/main/nvbandwidth.cpp).

### 3. Repair Strata quality through explicit task policies

The baseline's six misses are not one failure mode:

- `code-diff-keys` and `code-merge-config` consumed the full 16,384 tokens;
- `code-redact` stopped normally but failed its executable check;
- `extract-missing` and `extract-quoted` returned wrong values;
- `support-relevant-db` earned 2/3.

First test hard reasoning budgets of 512, 1,024, and 2,048 tokens for the low
reasoning coding requests. Strata documents hard
`reasoning_budget_tokens` support. A bounded reasoning phase may leave enough
of the existing 16K allowance for executable code in the two length failures.
Do not raise the output cap again until this matrix shows that answer space,
rather than looping reasoning, remains the cause.

Second, test a non-thinking policy for extraction and simple transformation
tasks using Qwen's official non-thinking sampler: temperature 0.7, top-p 0.8,
top-k 20, and presence penalty 1.5. Keep the existing official thinking
sampler—temperature 1.0, top-p 0.95, top-k 20, min-p 0—for reasoning,
retrieval, and harder coding. The current suite already uses the recommended
thinking temperature/top-p/top-k values.

This policy comparison is fair only if task routing and request parameters are
applied to every arm that supports them. Report it as a new policy arm, not as
an improvement to Strata's old score. Re-run all 54 cases, retain raw outputs,
and require no regression in retrieval, tool replay, reasoning, or transcript
QA. For the three coding misses, the executable grader remains authoritative;
shorter output is not itself a quality win.

Sources: [Qwen3.8-Flash-Next model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next),
[pinned Strata details and tuning guide](https://github.com/Niko1221/Strata/blob/30ec18ec7094550fcc594fd948220d511d80464e/docs/DETAILS.md).

### 4. Qualify Flash Next IQ3_XXS, then assess IQ3_S feasibility

If policy changes do not close the Strata gap, test the publisher's IQ3_XXS
artifact. In the publisher's results, IQ2_XS is 68.0 GB total and scores 89.16
across AIME25, GPQA-Diamond, and LiveCodeBench v6; IQ3_XXS is 75.8 GB and scores
92.57, versus 93.12 for BF16. IQ3_XXS matches BF16 on AIME25 and trails it by
1.14 points on LiveCodeBench. It uses more memory and is slower than IQ2_XS,
so it is a quality-for-capacity trade, not a free tuning win.

IQ3_S is the publisher's recommended quality point: 83.6 GB total and a 93.26
task average, including a 0.57-point LiveCodeBench gap to BF16. It adds 15.6 GB
of transformer weights over IQ2_XS, however. The IQ2_XS baseline already
reached 33.48 GiB container RAM and left 25.95 GiB host memory available. A
rough additive projection puts IQ3_S near or below the campaign's 12 GiB
host-available gate and beyond the current 44 GiB container cap. Treat IQ3_S
as a feasibility probe after IQ3_XXS, with an early resource abort, rather than
the default next download.

Name it as a separate model/quant arm. Start with conservative container memory
and the same 12 GiB host-available and PSI gates; the IQ2_XS run already used
33.48 GiB of container RAM. Compare at the same context, output cap, sampler,
reasoning policy, and GPU profile.

Source: [ISTA-DASLab Flash Next GSQ/RCO GGUF model card](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF).

### 5. Tune llama.cpp prefill, then test prompt lookup only where it fits

ThinkingCap is the quality baseline and already uses upstream llama.cpp b11277,
full GPU offload, flash attention, Q8 K/V, the default 2,048-token batch, and
512-token microbatch. Run a small `--batch-size`/`--ubatch-size` matrix around
those defaults for medium and 114K prompts. This can affect prefill and memory;
there is no strong evidence that it will improve the 39–42 tok/s decode rate.

An open llama.cpp Qwen3.8 27B report reproduces decode degradation across Q4
KV, Q8 KV, FP16 KV, batch/microbatch changes, MTP on/off, graph settings, and
cache settings. It does not prove the behavior is identical here, but it argues
against a broad launch-flag search. Include first-token, 2K-output, and
16K-output decode traces to see whether rate declines with generation length.

Test prompt lookup only for repetitive code editing or summarization workloads,
and only if the pinned server exposes the required option. Use cold-cache and
repeated-prefix cases separately. llama.cpp's discussion describes prompt
lookup as useful when the output copies substantial input; it is not a generic
reasoning accelerator.

Do not attach the base Qwen MTP projector to ThinkingCap without a full model
qualification. llama.cpp maintainers and users report jagged acceptance by
draft depth and non-bitwise-equivalent quantized output, and ThinkingCap has no
matched publisher MTP artifact.

There are two stronger but non-portable speed leads. A llama.cpp Qwen3.8 27B
community A/B attributes most of a 21.97% CUDA gain to a speculative-draft and
rollback patch chain; it also reports a jagged acceptance curve, CUDA graph
allocation cliffs at deeper drafts, and output divergence from target-only
decode. A separate Flash Next research fork reports a shape-keyed CUDA graph
cache improving generation 12–14% with byte-identical responses on its narrow
test. Both used different GPUs, model artifacts, and patched forks. The pinned
ThinkingCap logs already show graph reuse, and there is no matched MTP head, so
these are code-audit leads: first determine whether b11277 contains the fixes
and whether its graph cache is missing the reported shape key. Build a separate
pinned candidate only if a relevant patch is absent; compare graph reuse,
allocations, acceptance, raw output, and 2K/16K decode. Do not replace the
upstream baseline with either fork wholesale.

Sources: [pinned llama-server options](https://github.com/ggml-org/llama.cpp/blob/eae11d221/tools/server/README.md),
[Qwen3.8 decode degradation report](https://github.com/ggml-org/llama.cpp/issues/27444),
[prompt lookup discussion](https://github.com/ggml-org/llama.cpp/discussions/4235),
[Qwen3.8 MTP discussion](https://github.com/ggml-org/llama.cpp/discussions/27290),
[Flash Next CUDA graph research fork](https://github.com/thadreber-web/llama.cpp-qwen38-flash-next).

### 6. Target the vLLM arms by bottleneck

For OrcaSAQ2, benchmark MTP depths 1, 2, and 3 with acceptance telemetry. Depth
2 is already active, so deeper is useful only if accepted tokens offset extra
draft work without changing the two failed executable coding behaviors. Keep
the exact OrcaSAQ2 checkpoint and kernel versions.

For both vLLM arms, tune chunked-prefill scheduling for the observed short,
medium, and 114K prompt mix. Automatic prefix caching helps only when requests
reuse an exact prefix; it saves prefill work and does not improve decode. Split
cold, repeated-prefix, and unrelated-prompt results. Do not count a warm-cache
latency result as general throughput.

W4A16's first problem is capacity: its 98,304-token configuration cannot admit
the frozen 114K prompts plus 16K output. Increasing maximum sequence length may
restore coverage at the cost of KV capacity, but should not be reported as a
speed optimization. The W4 checkpoint has no matched MTP artifact in this
campaign, so do not graft Orca's projector onto it.

Sources: [vLLM automatic prefix caching](https://docs.vllm.ai/en/v0.30.0/features/automatic_prefix_caching/),
[vLLM optimization and tuning](https://docs.vllm.ai/en/v0.30.0/configuration/optimization/).

## Options to reject or isolate

| Option | Decision | Reason |
| --- | --- | --- |
| Strata Q4 KV | Reject for the quality track | Publisher reports about 4% speed but 8–12% worse perplexity; current int8 KV is the safer control |
| Strata speed projection | Reject for fair comparison | Changes output and safety behavior, worsens perplexity, and costs roughly 0.2–0.4% according to the publisher |
| `STRATA_IQ_MT_MIN=1` | Skip unless determinism is required | Publisher reports a 1–3% cost; its benefit is greedy-output independence rather than speed |
| Higher repetition or presence penalties | Avoid on Strata speed runs | Publisher reports 1–11% lower speed from reduced draft acceptance |
| Conversation/prefix cache as generic speedup | Isolate | Benefits exact repeated prefixes; it does not accelerate unrelated cold requests or decode |
| More GPU overclock | Reject as first-line work | Existing pre-profile ThinkingCap run crossed the VRAM guard; final profile is already stable |
| Swift or coder-tuned weights | Separate model arm | Different weights cannot establish an engine or tuning improvement for the five tested artifacts |
| Unmatched MTP projector | Reject | Changes the model path without publisher compatibility evidence |

The Strata figures in this table come from the
[pinned publisher details](https://github.com/Niko1221/Strata/blob/30ec18ec7094550fcc594fd948220d511d80464e/docs/DETAILS.md).

## GitHub and Reddit field reports

GitHub material above is either publisher documentation or an issue/discussion.
Issues and discussions are useful for experiment selection, but they are not
performance specifications. One recent llama.cpp tool-trigger slowdown was an
O(n²) scan and was closed by the lazy trigger fix in PR 27679; the pinned b11277
build postdates that fix, so reproducing it is low priority:
[issue 27615](https://github.com/ggml-org/llama.cpp/issues/27615),
[fix 27679](https://github.com/ggml-org/llama.cpp/pull/27679).

Reddit reports are anecdotal and use uncontrolled hardware, weights, prompts,
and settings:

- A Flash Next user reports roughly 50 tok/s decode and 1,500 tok/s prefill on
  a 12 GB setup, and says earlier KV/CPU defects were fixed. That is the same
  order as this host's 49.6/1,531 result, so it supports treating the baseline
  as plausible rather than evidence of a large missing switch:
  [r/LocalLLaMA report](https://www.reddit.com/r/LocalLLaMA/comments/1wtv43r/).
- A Swift/Flash Next thread reports about 130 tok/s decode and 6,000 tok/s
  prefill for an IQ3_XXS setup, while another participant reports poor coding
  and tool behavior from an IQ1 variant. Different weights, quantization, and
  setup make both discovery leads only:
  [r/LocalLLaMA thread](https://www.reddit.com/r/LocalLLaMA/comments/1wuark9/).
- A multi-GPU result uses a different quant and topology. It is not a useful
  target for a single RTX 3090:
  [r/LocalLLM report](https://www.reddit.com/r/LocalLLM/comments/1wu6fka/).

No field report should replace the frozen suite, raw response review, thermal
guard, or matched repeats.

## Acceptance protocol

1. Reproduce the current arm before changing it. Preserve pinned images,
   revisions, request plan SHA, GPU profile, and container limits.
2. Change one engine or policy variable at a time. Use at least three measured
   repeats and multiple seeds for quality-affecting changes; report medians and
   raw observations rather than the best run.
3. Separate cold cache, exact-prefix reuse, prefill, and decode. Record draft
   acceptance and cache hit rates whenever speculation or caching is involved.
4. Require the existing thermal and resource gates: no 101°C VRAM trip, at
   least 12 GiB host memory available, and memory PSI `some` avg10 at or below
   5. Preserve the 150K-context retrieval checks where the arm claims support.
5. A speed change passes only if the full frozen score does not regress. A new
   sampler, reasoning budget, quant, weight set, context limit, or task router
   is reported as a separately named arm and compared under the same policy
   wherever runtimes support it.
6. Promote a change only after it improves the relevant end-to-end metric by
   more than run-to-run noise. Retain the old command and artifact as rollback.

## Recommended order

1. Freeze three-repeat baselines and run the static-residency Strata diagnostic.
2. Run Strata calibration and a suite-derived expert profile.
3. Test the reduced English/code draft vocabulary.
4. Run the reasoning-budget and non-thinking policy matrix across compatible
   arms, with executable and extraction graders unchanged.
5. Qualify Flash Next IQ3_XXS only if the policy track does not recover quality.
6. Run the narrow llama.cpp batch/microbatch and generation-length diagnostics;
   add prompt lookup only for a real repetitive-code workload.
7. Tune Orca MTP depth and vLLM chunked prefill after the two priority arms.

This order tests publisher-supported, host-specific changes before changing
model identity or spending time on low-evidence launch flags.

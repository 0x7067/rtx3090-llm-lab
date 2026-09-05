# Paired quality and performance harness

A stdlib Python harness for comparing inference profiles through their real
OpenAI-compatible chat API. It borrows the paired prompts, explicit quality
margins, and separate quality/performance measurements from the
[September 5 NInfer comparison](https://reddit.com/r/LocalLLaMA/comments/1w821fg/ninfer_vs_llamacpp_vs_vllm_quality_speed/).
The dataset and implementation here are new; the Reddit author's dataset was
not supplied. This harness does not reproduce that author's scores.

The first [58-case production baseline](baseline-2026-09-05.md) records the
2026-09-05 v18 results and their limits, including cache reuse and the deepest
observed prompt sizes.

## Run a comparison

From this directory, with Python 3.10+:

```sh
cp arms.example.json arms.local.json
# Fill in both endpoints (including /v1), model IDs, and runtime metadata.
# For authenticated endpoints, add api_key_env: "BENCH_API_KEY" to the arm;
# export the value in your shell. Never put a key in the JSON or URL.
python3 harness.py freeze --suite suite-smoke.json --arms arms.local.json \
  --campaign qwen-trial-001 --out plan.local.json
python3 harness.py run --plan plan.local.json --arm control --out runs/control
python3 harness.py run --plan plan.local.json --arm candidate --out runs/candidate
python3 harness.py compare --plan plan.local.json \
  --control runs/control --candidate runs/candidate --out report.local.json
python3 -m unittest -v test_harness.py
# Also exercise the installed, pinned Docker code sandbox:
PAIRED_DOCKER_TESTS=1 python3 -m unittest -v
```

`freeze` materializes every request and hashes the plan and harness source.
Both arms receive the same messages, seeds, generation parameters, tools, and
output budgets; only the model field and endpoint differ. Unsupported parameters
produce recorded failures, never a silent retry with different settings. An API
may silently ignore a parameter: verify effective sampling, reasoning settings,
and template in the engine configuration/logs before a qualification run.
Identical seeds do not synchronize different engines' random-number generators.

`run` writes a manifest, an excluded warmup response, and one JSONL record per
attempt, flushing each record immediately. Outputs include content, reasoning,
parsed SSE events, tool calls, finish reason, usage, available server timings,
and grading details. HTTP errors retain their body; malformed streams retain
received bytes as text. Authentication header values are not recorded. These
artifacts contain prompts and outputs, so keep private workloads out of Git.
Output directories cannot be overwritten. An interrupted run remains incomplete;
start a fresh directory rather than silently replacing or dropping observations.
`compare` refuses incomplete, duplicated, or differently frozen observations.

For a single 3090, run arms sequentially, switching the engine outside this
harness. It does not deploy images, change GPU settings, or stop services.
Record image/build version, target and draft weight hashes, quantization, KV
precision/capacity, chat template, speculation, concurrency, GPU driver/power,
and cache policy in each arm's metadata. Keep unrelated GPU activity absent.
For serious measurements repeat the campaign in AB/BA order, reset caches between
arms, and retain each run. Do not treat repeated observations as new documents.

Each measured request starts with a unique campaign/case/repeat/seed identifier,
identical across arms. This reduces prefix and ngram reuse but cannot guarantee
cold cache state or defeat all engine caches. No `cache_prompt` override is sent.
Use a fresh backend or an independently verified cache reset for cold-prefill
claims. A warm-cache session is a separate experiment with an explicit policy;
this harness's replay fixtures do not measure cumulative agent-session latency.

## Starter suite and customization

For a larger authored regression workload, `build_workload.py` generates **58
cases**: 20 programming contracts, 16 tool-replay steps across eight episodes,
six extraction cases, six supporting reasoning/relevance/transcript cases, six
retrieval variants, and four performance probes. Coding contracts include LRU,
topological order, TTL boundaries, recursive config merge, Unicode byte batching,
rate limits, and SSE framing. Gold assertions are checked against reference
implementations before generation; references are never included in requests.
The tests also inject known boundary defects and verify that gold checks reject
them. Passing these checks does not establish complete test coverage.

```sh
mkdir -p runs/workload
python3 build_workload.py --out runs/workload/suite.json
python3 harness.py freeze --suite runs/workload/suite.json --arms arms.local.json \
  --campaign infrastructure-001 --out runs/workload/plan.json
python3 harness.py run --plan runs/workload/plan.json \
  --arm control --out runs/workload/control
python3 harness.py summarize --plan runs/workload/plan.json \
  --run runs/workload/control --out runs/workload/baseline.json \
  --markdown runs/workload/baseline.md
```

The single-arm summary reports observed quality, failures, timing, and prompt
token ranges; it never issues an equivalence or speedup verdict. Run a candidate
against the same frozen plan and use `compare` when a second profile is ready.
`--depth-lines 64 1024 4096` and `--seeds 42 314` customize workload generation.
Depths specify background records, not token counts: consult server usage in the
results. The deepest default requests approach the current 131k allocation and
can take minutes to prefill. Smaller-context arms may reject them; those failures
stay in the primary score. A timeout is also a failure, not an excluded item.

This is **synthetic development/regression data**, not an independently sampled
production benchmark or a release gate. The six supporting cases reuse smoke
fixtures. Retrieval lengths share two document-family clusters; the two steps
of each tool episode share one cluster. Extra lengths, steps, and seeds do not
add independent sources. The default uses greedy sampling and one seed. To
study sampled production behavior, change common sampling parameters and seeds
in the suite before freezing; do not mix these results with the greedy baseline.

`suite-smoke.json` has 16 quality items and two performance probes:

| Tier | Scoring contract |
| --- | --- |
| Relevance | Set F1 over returned document IDs; extra IDs hurt precision |
| Retrieval | Exact needle answer at two document lengths and two positions |
| Transcript QA | Final decision extracted from English/Portuguese dialogue |
| Reasoning | Boundary logic and dependency ordering |
| Extraction | JSON equality or set F1, including Portuguese and corrections |
| Coding | Generated Python exercised against independent behavior assertions |
| Tool replay | Next tool name and parsed arguments after frozen episode history |

The small synthetic suite checks harness plumbing. It is **not a release gate**.
Its two retrieval lengths are 32 and 512 **lines**, not 32K/512K tokens. Actual
prompt token counts come from server usage; no character heuristic is used.
For a real campaign, replace fixtures with reviewed, held-out infrastructure and
coding tasks, realistic distractors, longer documents, and more independent
sources. Cases sharing a source, template family, or episode must share a
`cluster`; seed/repeat copies stay inside that cluster automatically.
Do not use development fixtures as qualification evidence. Freeze gold labels
and margins before observing candidates; label corrections require a new plan
and rerunning both arms.

Each case supplies `id`, `tier`, `cluster`, `phase`, and a chat `request`.
Quality cases also supply a `grader`: `exact`, `json`, `set_f1`, `tool`, or
`python`. Case request fields override `request_defaults` by whole top-level
field. `model`, `seed`, `stream`, and `stream_options` belong to the harness.
Examples are in the suite. Adding `response_format` to selected requests tests
that API feature explicitly; unsupported JSON mode is an error, not a skipped
tier. Prompt-only JSON is useful but is not equivalent to constrained decoding.

Python tests run in a disposable, network-disabled, read-only Docker container
with no host mounts, dropped capabilities, an unprivileged UID, and CPU/memory/
process/time limits. Both plain Python and a single Python code fence are
accepted. The grader checks behavior, not the presence of a fence or substring.
The suite pins `code_image` to the local `python:3.12-slim` image SHA256 used in
this lab. On another host, provision an appropriate image yourself and set its
local image ID **before freezing**. The harness never pulls an image.
Tool fixtures never execute model-proposed tools; they are diagnostic replays,
not evidence that a live agent can complete a task. The existing
[editing-session harness](../qwen-speed-2026-09-04/README.md) remains useful for
full multi-turn coding episodes and concurrency qualification.

## Reading results

Quality scores range from 0 to 1. Errors, invalid formats, and output truncations
score zero in the primary denominator. Completed-request-only scores are shown
separately with counts, so a small context window cannot improve its apparent
score by dropping difficult requests. Retained HTTP error bodies distinguish
context-capacity failures from unsupported parameters and other API failures;
the harness does not guess their cause from status 400 alone.

The quality delta is **candidate minus control**, averaged within each cluster,
then equally across clusters. Bootstrap samples resample whole clusters, retaining
all paired items/seeds/repeats inside them. Reports show the configured two-sided
confidence interval, a predeclared absolute score margin, and a per-tier verdict.
A lower bound above `-margin` establishes non-inferiority at that margin for that
tier; an upper bound below it indicates inferiority beyond the margin. Otherwise
the result is inconclusive. Fewer than `min_clusters` always yields inconclusive.
Identical differences across all observed clusters also yield inconclusive:
resampling a constant cannot estimate uncertainty about unseen tasks. Such
intervals remain visible and are marked `degenerate_bootstrap`; a zero-width
interval is not evidence of population-level equivalence. Comparisons record
the analysis source hash as well as the frozen inference plan hash.
The default 20-cluster guard is a minimum, not a power calculation. Plan sample
size for the chosen margin using independent pilot data. Intervals are exploratory
and not corrected for testing multiple tiers. There is no automatic overall
quality-equivalence or promotion verdict. Failure to detect a difference is not
proof of equivalence.

Performance reports keep five metrics separate:

- `wall_s`: client time from submission through stream completion.
- `ttft_s`: arrival of the first nonempty content, reasoning, or tool delta;
  role-only events are excluded. This is event-arrival latency, not GPU time.
- `e2e_output_tps`: server-reported completion tokens / total client wall time,
  including prefill, scheduling and reasoning tokens as counted by that server.
- `server_decode_tps`: llama.cpp's explicit `predicted_per_second`; null for a
  single output token, which has no meaningful decode interval.
- `server_prefill_tps`: explicit `prompt_n / prompt_ms` from llama.cpp timings.

Missing usage/timings produce null metrics, never token or chunk estimates.
vLLM/NInfer server timings need an explicit adapter if their SSE schema differs;
client E2E throughput is never substituted into a server-decode column.
Every metric reports the number of valid pairs and medians, including the
candidate/control ratio. Failures and per-observation prompt/completion counts
and finish reasons remain visible. Performance output caps (`length`) are
expected in fixed-budget probes; they fail quality items. Inspect token counts
and early EOS before interpreting wall-time ratios. Speed ratios are descriptive,
not confidence-backed speedup claims. There is no streaming-chunk token rate.

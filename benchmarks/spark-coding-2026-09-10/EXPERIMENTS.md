# Spark coding pilot (2026-09-10)

Objective: find a coding-quality improvement on the existing Spark X2.5 4B
Q8_0, FP16 KV, 393,216-token profile, while preserving tool-call correctness.
The user chose benchmarking and settings tuning first; prepare a LoRA plan
from the findings, without training weights in this phase.

## Frozen pilot

The existing paired-harness workload supplies 20 Python coding contracts and
16 frozen tool-replay requests in eight episodes. A deterministic cluster
split reserves six coding cases and two tool episodes (10 requests) for a
final comparison. Development has 14 coding cases and six tool episodes
(26 requests). The split was frozen before inference, and holdout responses
were generated only after locking the selected profile. Every screening request uses seed 42, an 8,192-token
total output budget, top_p 0.95, top_k 0 and thinking enabled.

These are synthetic development fixtures, not an independently sampled
repository benchmark. Tool replay does not execute an agent's tools. One
sample per case is a screening result, not a population-level accuracy claim.

Working runs were written to `/tmp/spark-coding-20260910` and archived under
`artifacts/` beside this log. The harness records
all failures and output truncations. Generated Python runs in its existing
network-disabled Docker sandbox with no host mounts. Neither the model nor
its generated code receives the grading assertions in the prompt.

## Experiments

| Condition | Hypothesis / status |
| --- | --- |
| Publisher sampling baseline | Temperature 1.0, unrestricted thinking within the total 8k budget. Development: coding 4/14, tool replay 12/12; nine all-thinking truncations. |
| Reasoning budget 2048 | Same settings except `reasoning_budget_tokens: 2048`. Stopped after 10/26 completed requests, plus one interrupted in-flight request. SSE still hits the total limit; histogram and flatten finish but fail assertions. Incomplete diagnostic, no suite score or equivalence claim. |
| Non-thinking | Same baseline sampling, `enable_thinking: false`. Development: coding 3/14, tool replay 12/12, no truncations. Median coding latency 1.78s versus 66.94s, but correctness regressed. Reject as coding-quality improvement. |
| Temperature 0.2 | Same baseline except temperature. Development: coding 6/14, tool replay 12/12; seven truncations. Gains retry-budget, topological and JSONL; loses env-lines. Superseded by the brief-prompt condition. |
| Non-thinking, temperature 0.2 | Isolates temperature relative to the non-thinking run. Coding 2/14, tool replay 12/12. Reject as a quality improvement. |
| Concise system instruction | Development 9/14 coding, 12/12 tool replay, four truncations; selected. The initial SSE probe passes but its identical full-run request truncates; both retained. |
| 32k output diagnostic | Original SSE request with only max_tokens increased from 8,192 to 32,768; all other request and grading fields verified identical. Fails: all 32,768 tokens spent thinking, no final code, 285.16s. |

Early diagnostic observation: `code-sse-frames` and `code-histogram` both
consumed 8,192 tokens and ended with `finish_reason=length`, returning no
final code. These remain failures in the baseline denominator. This does
not establish their eventual correctness with an unlimited output budget.

Compare policy changes by matching case IDs, prompts, graders, seed and total
budget, while explicitly recording the intended parameter difference. The
existing harness's `compare` command requires identical inference plans and
must not be used to pretend these different policies have identical requests.
Freeze the selected candidate before evaluating the reserved cases. Keep the
current serving configuration unless the final comparison supports a change.

The baseline uses publisher sampling at a deliberately bounded 8k output budget;
it is not a measurement of maximum quality with the deployment's 32k allowance.
Non-thinking removes truncation but loses three previously passing cases and
gains two others (net -1). This is evidence against treating latency alone as
an improvement. A forced reasoning end also does not guarantee a concise or
correct final answer.

`agent-suite.json` freezes three additional synthetic read/write/test episodes.
`agent_smoke.py` reuses the existing streaming client and Docker code grader;
files exist in memory, only one exact source path is writable, and visible and
hidden tests are immutable. Original bugs fail both test sets and independent
reference fixes pass both. Invalid paths, tool names and argument shapes were
checked, as were successful reads/writes and failed/passing test feedback.
The paired harness's unit suite completed with 12 passes and one opt-in Docker test skipped;
the new fixtures separately exercised the actual Docker sandbox.

Live editing: original thinking profile completed 2/3 fixes and workflows;
elapsed-days stalled after reading the files, using all 8k tokens without an
edit. Non-thinking at temperature 1 completed 3/3 fixes and workflows, with
every separate final assertion passing. The generated patches were inspected.
This small editing result does not erase non-thinking's weaker standalone
coding result and does not justify a universal non-thinking default.

Reproducibility limit: the concise SSE probe and full-run request have identical
request hashes and seeds but different results. Their cache histories differ
(9 versus 1 cached prompt tokens); this does not establish cache state as the
cause. Keep both observations and do not promise bitwise reproducibility from
the seed. No claim of stable general improvement rests on this one probe.

The selection rule was recorded before the full concise run finished: choose
the completed development condition with the most coding passes, requiring
all 12 tool replay cases to pass. Retain temperature 0.2 if concise ties it.
Incomplete diagnostics and one-case probes are not eligible. Lock the winner
before holdout inference and do not tune further on holdout outcomes.

## Final decision

Locked the concise instruction at 2026-09-10 10:11:54 UTC before any holdout
inference. The reserved comparison then scored original 4/6 versus concise
6/6 for coding, with 4/4 tool replay for both. The selected instruction gained
five development and two reserved coding passes, losing none. No additional
tuning followed the holdout.

Keep it as the experimental request-level `coding-profile.json`, retaining
publisher sampling and thinking. Serving defaults, model weights, 384k context
and speculation settings are unchanged. Preserve the failed/incomplete trials
and the same-seed divergence. The small synthetic sample supports a prompt
pilot; real repository evaluations and repeated samples remain the next
qualification step before a global default or trained-weight replacement.

Five persistent tool-boundary/fixture tests pass, including Docker execution.
The report analysis checks complete observations, unchanged graders and exact
request equality after removing only the intended system instruction. Raw
results and frozen manifests are archived under `artifacts/`.

Final verification also checked compilation, YAML syntax, exact profile/plan
settings, unchanged harness hashes and lossless raw-result compression. The
service remained ready with context 393216, maxTokens 32768 and function calling
enabled. Static quality review found no worsened existing symbols; new findings
concerned fixture-JSON verbosity, unittest discovery and shared setup in two
focused tests. The frozen evidence and distinct behavioral tests were retained.

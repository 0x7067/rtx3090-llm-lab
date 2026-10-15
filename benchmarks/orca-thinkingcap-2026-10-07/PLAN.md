# OrcaSAQ2 versus ThinkingCap: qualify speed without a quality regression

Compare the current OrcaSAQ2 speed settings (MTP 3, 4096 batch tokens, eight sequences, FP8 KV, 150K context) against ThinkingCap Q4_K_S on the already retained upstream b11277 image with two slots, unified 150K context pool and Q8 K/V. Preserve weights, GPU settings, sampling and output allowances. No unmatched draft model or lower-precision ThinkingCap KV is introduced.

The full authored regression suite runs once per profile. All 20 coding cases repeat on two additional fixed seeds. Four fixed-budget performance probes repeat on those seeds. This is development evidence, not proof of population-level equivalence. Clarify three previously ambiguous prompt contracts equally for both arms: the sensitive argument is a list, empty SSE events without data are skipped, and the extraction key is literally `value`. Retain old scores unchanged.

After quality, run fixed 512-output-token requests with 1/2/4/8 callers on short prompts, 1/2 callers around 32K, and two simultaneous retrieval requests around 68K each. Record actual input/output token counts, each request's TTFT and latency, aggregate throughput, failures, and VRAM. Eight admitted callers must not be interpreted as eight resident 150K contexts. ThinkingCap's 150K pool is shared.

Gate: no new executable, tool, extraction or retrieval failures relative to the paired ThinkingCap control; no transport or engine errors; both 68K retrievals correct; no truncation on graded requests. Report speed separately if a quality gate fails. Do not turn an inconclusive statistical result into equivalence. A single failure is investigated, not removed from the denominator.

Execution uses loopback-only isolated Docker servers, 32 GiB container memory limits and an independent memory/thermal monitor. Abort below 12 GiB host available memory, above memory PSI avg10 5, or on thermal guard latch. Require the existing thermal guard and fan controller to be active. Freeze live deployment replica/suspension state before the GPU window; stop only the llama Deployment, then restore its original replica count and Flux suspension state in a finally block. Do not reconcile a changed Git revision or change production manifests.

The GPU remains on its current host profile. Earlier measurements used a cooler profile; new results are compared only within this campaign. If ThinkingCap cannot finish under the active thermal profile, report that limitation instead of bypassing the guard.

# Two long agents: ThinkingCap with a 200K shared Q8 pool

Before any ThinkingCap inference, a tokenizer preflight measured the identical concurrency filler at 83,323 tokens per long request, instead of the rough 68K estimate. Two such prompts do not fit a 150K shared pool. This is a context-capacity issue, not a coding-quality result.

Qualify a 200,000-token unified pool, retaining two slots, Q8 K/V, exact weights, pinned b11277 runtime and all sampling/output settings. Do not lower cache precision. The measured prompts, graders, request hashes, seeds and job ordering are preserved from the initial campaign; only the ThinkingCap arm metadata changes. Orca observations remain in their original files and original frozen plan. The comparison explicitly validates each run against its own plan, then requires identical request and grader bytes across plans.

The parent GPU window is stopped after Orca completes its observations. Its original service restoration must finish before starting this second guarded window. The second window repeats the same original-state capture and restoration. A partial 150K ThinkingCap attempt, if any, remains separate and excluded. No artifacts are rewritten as if measured under the larger pool.

Use the full 54-case quality suite, two extra seeds for every coding case, three-seed fixed-token speed probes, and the exact same concurrency batches. Failure to load or a thermal/resource abort is a profile feasibility failure. No lower-precision substitute is admitted. A 200K shared pool does not provide 200K to both agents at once; the sum of live contexts and generated tokens must fit it.

Fixture correction before ThinkingCap measurement: the initial Orca long-concurrency question repeated its reference key at the end, so it tested admission but not retrieval integrity. Preserve those observations as admission-only. The corrected question asks for the key near the beginning without repeating it. ThinkingCap's two long batches and the later Orca concurrent-quality window use identical corrected request bodies. Short and medium fixed-budget request bodies are unaffected. The original parent runner is preserved as `../run-orca-window-frozen.py`.

Both models additionally run all 20 executable coding cases with two callers, plus the SSE and histogram cases on two extra seeds (24 observations total). That separate panel explicitly requires a list return in the SSE prompt; it does not rewrite the earlier generator failure. The histogram repeats follow its newly observed genuine final-edge failure and retain the original prompt. Inference time excludes grading time.

Each corrected cold 83K two-agent retrieval batch is followed by two simultaneous continuations preserving the original input and actual visible assistant answer. Record warm TTFT, total latency, correctness and server-reported cached tokens. This represents agents continuing their sessions rather than measuring only unrelated cold prompts. Cold and warm batches are reported separately.


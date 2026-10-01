# Benchmarks

Each campaign keeps its runner, frozen inputs, raw results, and decision record
together. Numbers from different campaigns are not interchangeable unless the
model bytes, runtime, sampling, context depth, and GPU profile match.

- [`local-llm-reset-2026-09-30/`](local-llm-reset-2026-09-30/) compares five
  publisher artifacts under one 16,384-token output cap and records the
  ThinkingCap llama.cpp promotion profile.
- [`orcasaq-vs-w4-2026-09-25/`](orcasaq-vs-w4-2026-09-25/) records the
  OrcaSAQ2 EXL3 promotion over W4A16, the W4A16 admission failures, and the
  54/54 hybrid reasoning-effort result.
- [`tiel-vs-qwen/`](tiel-vs-qwen/) compares the deployed Tiel llama.cpp and
  Qwen vLLM configurations.
- [`qwen-vllm-hillclimb-2026-08-28/`](qwen-vllm-hillclimb-2026-08-28/)
  records the Qwen slowdown investigation and rejected optimization arms.

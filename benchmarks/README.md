# Benchmarks

Each campaign keeps its runner, frozen inputs, raw results, and decision record
together. Numbers from different campaigns are not interchangeable unless the
model bytes, runtime, sampling, context depth, and GPU profile match.

- [`tiel-vs-qwen/`](tiel-vs-qwen/) compares the deployed Tiel llama.cpp and
  Qwen vLLM configurations.
- [`qwen-vllm-hillclimb-2026-08-28/`](qwen-vllm-hillclimb-2026-08-28/)
  records the Qwen slowdown investigation and rejected optimization arms.

# Legacy-runner MTP suffix-lookup derivative

This v2 arm targets the execution path actually used by local
`qwen38-27b-3090:v9`: `vllm.v1.worker.gpu_model_runner.GPUModelRunner` with
`vllm.v1.spec_decode.llm_base_proposer.SpecDecodeBaseProposer` and
`DraftModelProposer`. The earlier v1 arm modified the V2
`AutoRegressiveSpeculator`, which v9 did not instantiate.

## Build

From this repository root:

```bash
docker build -t qwen38-27b-3090:mtp-suffix-poc-v2 experiments/mtp-suffix-vllm-v2
```

The derivative uses the local `qwen38-27b-3090:v9` image and applies the
legacy-runner patch only. The v9 image already contains the reusable
`v1/worker/gpu/spec_decode/dflash2/lookup.py` Triton `suffix_lookup` and
`fuse_draft` kernels.

## Run contract

Use the normal single-user launcher with `DRAFT_TOKENS=7`,
`VLLM_MTP_SUFFIX_LOOKUP=1`, and `ASYNC_SCHED=0`. The synchronous requirement is
intentional: in v9, async bookkeeping leaves `-1` placeholders in the legacy
CPU token history until the next iteration. The patch runs three MTP passes,
then returns either a three-token MTP proposal or a complete seven-token local
continuation to the scheduler.

The startup marker is:

```text
MTP_SUFFIX_LOOKUP_ACTIVE legacy proposer
```

Successful lookup rounds emit:

```text
MTP_SUFFIX_LOOKUP_HIT legacy: hits=... verify_tokens=...
```

Lookup-owned rows rewrite the dense draft probability tensor to a point mass;
this preserves probabilistic rejection sampling and is not a greedy-only
shortcut.

## Validation boundary

The patch applies and compiles, but this arm has no GPU result yet. Validate
startup marker, hit marker, 3/7 scheduler transitions, rejection in a 7-token
round, hybrid GDN state alignment, structured outputs, preemption, output
hashes, and speed before considering promotion. Do not alter production config.

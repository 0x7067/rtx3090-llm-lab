# MTP suffix-lookup derivative

This is an isolated derivative of the local `qwen38-27b-3090:v9` image. It
does not modify the live deployment or either production repository.

## Build

From this repository root:

```bash
docker build -t qwen38-27b-3090:mtp-suffix-poc experiments/mtp-suffix-vllm
```

The Dockerfile applies `mtp-suffix-vllm.patch` to the v9-installed
`/app/venv/lib/python3.12/site-packages/vllm` tree, compiles both changed
modules, then reverse-dry-runs the patch. v9 already contains the Triton
`dflash2/lookup.py` implementation; the derivative reuses its
`suffix_lookup` and `fuse_draft` kernels.

## Opt-in contract

Start the normal v9 runner with `VLLM_MTP_SUFFIX_LOOKUP=1` and configure
`num_speculative_tokens=7` (the bundled single-user launcher uses
`DRAFT_TOKENS=7`). The patch then runs exactly three MTP draft passes and asks
the scheduler for either three tokens (fallback) or seven tokens (a complete,
strong/confirmed local continuation). Lookup-owned q rows become point masses,
so probabilistic rejection sampling remains lossless; this is not a
greedy-only shortcut.

The first implementation intentionally synchronizes one GPU flag read per
proposal to choose 3 versus 7. This makes the state transition explicit for
validation. Replace it with an asynchronous one-step-stale flag only after
measuring the overhead and proving mixed-batch determinism.

## Required GPU validation

The patch compiles and imports with lookup disabled, but no GPU speedup is
claimed. Before any service use, run fresh-boot A/B tests for output hashes,
acceptance, TTFT, decode rate, structured output, 3-to-7 and 7-to-3 scheduler
transitions, mid-block rejection, preemption/removal, and Qwen3.8 hybrid GDN
state alignment. The v9 image's 0.94 memory-utilization profile and existing
GDN/xgrammar patches must remain unchanged.

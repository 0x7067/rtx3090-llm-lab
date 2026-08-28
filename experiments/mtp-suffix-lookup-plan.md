# MTP + suffix lookup proof of concept

This experiment composes the Qwen MTP drafter in vLLM 0.27.1 with a local
suffix continuation table. It is intentionally isolated from the deployment
repositories and is not a production patch.

The applyable phase-2 derivative is under
[`mtp-suffix-vllm/`](mtp-suffix-vllm/). It layers onto the local
`qwen38-27b-3090:v9` image and is still opt-in and unqualified.

## What is implemented here

[`mtp_suffix_lookup.py`](mtp_suffix_lookup.py) is a dependency-free CPU
reference. It fixes the first experiment's contract:

- MTP drafts three tokens (`MTP_DRAFT_STEPS = 3`).
- The target may verify at most seven (`MAX_VERIFY_STEPS = 7`).
- Lookup searches only the accepted history, chooses longest match then most
  recent tie, and permits overlap.
- A match of six or more tokens is strong. A four-token match is confirmed only
  when MTP agrees with its first two tokens.
- The tail is used only when all seven continuation tokens are available and
  the head is strong/confirmed. Otherwise the scheduler receives exactly three
  MTP tokens, so no uninitialised tail can enter verification.
- Lookup-owned rows replace q with a point mass on the actual proposed token.
  The residual rejection sampler therefore remains exactly target-distributed.

Run from this directory:

```bash
python3 -m unittest -v experiments/test_mtp_suffix_lookup.py
```

## vLLM 0.27.1 integration plan

Phase 2 implements the plan in
[`mtp-suffix-vllm/mtp-suffix-vllm.patch`](mtp-suffix-vllm/mtp-suffix-vllm.patch):
the patch reuses v9's existing DFlash2 Triton lookup/fusion kernels, adds the
MTP 3/7 arbitration in `AutoRegressiveSpeculator`, rewrites dense q rows to
point masses, uses the existing v9 `set_req_states` and scheduler-count hooks,
and captures target query widths 4 and 8. The first phase below remains the
CPU oracle and the rationale for the seams.

The installed source was inspected under
`/data/buttercup_6tb/k3s/vllm-trial/venv/lib/python3.12/site-packages/vllm`.
Qwen MTP resolves through `v1/worker/gpu/spec_decode/mtp/speculator.py` to
`AutoRegressiveSpeculator`; the supplied
`df2-repo/patches/dflash2-lookup-drafting.patch` instead extends DFlash2 and
cannot be applied to MTP without changing these ownership boundaries.

1. Add a Triton implementation beside the MTP speculator. It must read
   `RequestState.all_token_ids.gpu` (UVA) and `total_len.gpu`, map batch rows via
   `AutoRegressiveSpeculator.idx_mapping`, and write fixed `[requests, 7]`
   lookup tokens, `match_len`, and `valid`. Keep the CPU module as its oracle.
2. In `AutoRegressiveSpeculator`, retain the configured target block at seven,
   but gate the MTP generation loop at three when
   `VLLM_MTP_SUFFIX_LOOKUP=1`. Apply arbitration after the three MTP passes,
   before returning `draft_tokens`. A point-mass rewrite must update the dense
   `draft_logits` rows for probabilistic sampling (`-inf` except `0` at the
   actual lookup token). Greedy MTP does not consume q, but must use the same
   token arbitration.
3. Add one optional `set_req_states()` hook from `model_runner.py` after
   `RequestState` construction. Do not copy `all_token_ids` to CPU; that would
   turn a long-context optimization into a synchronization and memory-bandwidth
   regression.
4. Extend `DraftTokensHandler.set_draft_tokens()` with an optional count and
   pass three or seven tokens to the scheduler after `propose()`. The first
   prototype should use a synchronous all-request flag read for correctness;
   only a later benchmarked version should replace it with a one-step-stale
   pinned async copy.
5. Capture target decode CUDA graphs for both query widths (4 for one sampled
   token plus three MTP drafts, 8 for one plus seven lookup tokens). The MTP
   drafter's own prefill graph must remain width 4; its per-token decode graph
   remains unchanged.

## Hybrid Mamba/GDN seam and blockers

The target runner's `model_state.preprocess_state()` and `prepare_attn()` use
the scheduler's `num_draft_tokens_per_req` to advance and align recurrent state.
The experiment must therefore change only the scheduler-visible count (3 or 7)
and never mutate Mamba state from the lookup kernel. Every transition 3 -> 7,
7 -> 3, rejection in the middle of a 7-token block, request removal, and
preemption needs runtime tests on the Qwen3.8 hybrid GDN model.

The following cannot be claimed safe from static inspection:

- Triton compilation and UVA access at 128k/140k context on the RTX 3090.
- Whether a seven-query FlashInfer/attention plan fits the current FP8 KV and
  GDN state layout without a new OOM or illegal access.
- CUDA-graph replay when the scheduler alternates widths in one batch.
- Exact seed/position behavior for probabilistic MTP after replacing processed
  logits, including structured-output grammar validation.
- Acceptance, TTFT, throughput, output hashes, and crash/restart behavior.

Do not enable this against the live service until those checks pass. This
repository contains no deployment/configuration change and this POC has no GPU
speed claim.

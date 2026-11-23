# Compact GDN serving integration

Experimental integration for the pinned Qwen3.8-27B DFlash7 runtime on RTX 3090.
The predecessor's arithmetic and microbenchmarks are documented in
`../../optimization/ordered-kernels/README.md`.

## Contract

The verification kernel reads the previous accepted state by its indexed cache
slot, then stores FP32 deltas, normalized keys and decays. After sampling,
`V2 GPUModelRunner.postprocess_sampled` reconstructs the accepted
state and any prefix-boundary state using its batch-to-request map and post-step computed counts, before the existing Mamba postprocess copies
and before KV offload finalization. The pinned OffloadingConnector has an empty
`save_kv_layer`; `wait_for_save` queues stores for the next step. The decoder, scheduler and prefix cache keep
their existing full-state format and allocation. This integration targets
snapshot write traffic; it does not claim reduced allocated KV cache memory.

Enable with `VLLM_COMPACT_GDN=1`. The flag is registered with vLLM and included
in its compilation-cache key. Startup logs record its engine-side value and
layer dimensions. Qualification requires both runtime engagement messages.
The initial serving scope is pure speculative
batches with uniform lengths 2–8, FP16 recurrent state, BF16 QKV, TP=PP=1,
the V2 runner, PIECEWISE graphs and aligned Mamba caching. Single-token, ragged and mixed-prefill
batches use the existing implementation. The underlying indexed kernels are
also tested on ragged sequences. Each layer owns scratch buffers allocated
before memory profiling. Request-row identity is preserved until sampling;
reconstruction completes before the next batch may reorder requests.

The kernel preserves the reference's rounded state decay before the outer-product
FMA. Every reconstructed destination is written after its initial state has been
loaded into that program's disjoint register tile.

## Qualification gate (declared before serving runs)

- Content checks and serving behavior checks pass in all four ABBA arms.
- Both four-request copy fixtures improve median full-request latency by ≥3%.
- Baseline arm drift stays within 5% for each measured fixture/concurrency.
- Every single-request fixture has at most 3% median latency regression.
- Candidate logs prove verification and finalization hooks engaged.

Content checking preserves the original strict formatting grade and additionally
accepts the already observed outer EXTRACT markers for copy fixtures. JSON,
tool round trips, long-prefix retrieval/reuse and cancellation recovery have
explicit assertions. This is a separate serving experiment from the prior
strict-format draft-chain adoption gate.

`test_gpu.py` checks exact outputs and required snapshots on indexed, interleaved
cache storage, shuffled request slots and rotating batch rows, ordinary/slow decay, recurrent steps and prefix boundaries.
`bench_serving.py` supplies matching streamed requests for ABBA measurement.
`quality_serving.py` covers serving behavior. Full artifacts and the restoration
runner live at `/data/development/reports/qwen-compact-serving-2026-11-12`.

## Build

The Dockerfile pins the baseline image. `apply_patch.py` verifies all three
installed source hashes before registering the flag and inserting the hooks.
Use the existing `strata-local-registry` builder, with network disabled for build
steps and the repository's 8 GiB / two-CPU resource bounds. Its registry exporter
publishes a unique tag and records the resulting digest. The default Docker
builder duplicates this large runtime into the root filesystem.

## Serving result — 12 November 2026

The production gate failed. Retain this as an experimental image recipe;
production keeps the existing AutoRound/DFlash7 image and settings.
`qualification.json` records the image digest, source revision and measurements.

All 288 indexed recurrent blocks remained bitwise exact. Both candidate runs
engaged verification and reconstruction. All 172 benchmark responses passed the
predeclared content/termination checks, and all four runs passed the six serving
behavior checks. Strict outer-format failures remained: 13 baseline responses
and 20 candidate responses included EXTRACT markers. These were accepted only
by the separate content grade described above.

| Fixture | Requests | Baseline median | Compact median | Latency reduction | Baseline drift |
|---|---:|---:|---:|---:|---:|
| copy-1024 | 1 | 6.168 s | 6.304 s | -2.21% | 2.07% |
| copy-1024 | 4 | 15.657 s | 15.453 s | +1.30% | 0.87% |
| copy-256 | 1 | 5.762 s | 5.872 s | -1.91% | 1.58% |
| copy-256 | 4 | 11.266 s | 13.174 s | -16.94% | 14.52% |
| edit-module | 1 | 1.330 s | 1.427 s | -7.33% | 0.08% |
| prose | 1 | 1.424 s | 1.363 s | +4.28% | 0.55% |

Positive reduction means faster. The long-context concurrent copy gain (1.30%)
falls below 3%, the edit regression (7.33%) exceeds 3%, and the short-context
concurrent copy case also exceeds the 5% baseline-drift limit. Individual faster
batches did not translate into a qualifying combined result. The earlier kernel
microbenchmark therefore supports further implementation work, while this
serving comparison supports retaining production's current path.

The integration fixes found during qualification were package permissions,
the V2 sampling callback and request mapping, registered compilation-cache flag,
and logger namespace. The missing INFO messages came from using a logger outside
vLLM's configured namespace; the initial environment-handoff hypothesis was not
established. The final build uses `vllm.compact_gdn` and proves hook engagement.

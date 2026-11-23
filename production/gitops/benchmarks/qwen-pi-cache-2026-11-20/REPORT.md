# Qwen Pi cache and decoding optimization, 2026-11-20

The user approved an exclusive 30-minute GPU maintenance window. The selected configuration is **MTP4, 16-token prefix matching and shared-prefix checkpoints**, with the original rank-32 adapter, W4A16 base, per-head int8 KV, FP32 recurrent state, 131072 maximum context, four admission slots and the existing 93% GPU budget. Deployed through reviewed GitOps in `9b22f1823cac16b34461dee68685a9279e5fda9a`. Production and all four Pi clients are verified; Flux is ready and unsuspended.

## Measured comparison

All profiles use the same pinned image, weights, recommended temperature 0.8/top-p 1, fixtures and output caps. Times include request processing. Each figure below is a single controlled probe, not a statistical latency guarantee.

| Profile | Short stream tokens/s | Cached 46K stream tokens/s | Four-caller aggregate tokens/s | Repeated ~3175-token prompt, s | Repeated 46K prompt, s |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline-mtp3 | 98.58 | 64.66 | 339.41 | 0.868 | 1.439 |
| fine-mtp3 | 99.48 | 89.70 | 340.67 | 0.217 | 0.256 |
| fine-mtp4 | 118.70 | 89.29 | 343.34 | 0.212 | 0.238 |
| fine-mtp2 | 88.32 | 76.47 | 292.73 | 0.153 | 0.244 |
| fine-mtp5 | 122.65 | 89.09 | 293.21 | 0.158 | 0.221 |

Fine-prefix MTP3 isolates the cache change: the ~3175-token repeat reused 3152 tokens instead of 1584 and completed in 0.217 rather than 0.868 seconds. The 46K repeat reused 46000 rather than 44352 tokens and completed in 0.256 rather than 1.439 seconds. Cached long-stream time to first output fell from 1.364 to 0.153 seconds. The three-token prompt-length variation comes from unique run markers.

MTP4 adds a substantial short-stream improvement without the MTP5 four-caller regression. MTP5's small additional short-stream gain did not compensate for falling to 293 aggregate tokens/s and reducing cache capacity further. MTP2 was slower on the streaming and concurrent probes. MTP4's measured cache seat budget is 172599 tokens versus 183233 for MTP3; its startup confirms one full 128K request fits. Admission slots share this cache and do not reserve four full windows.

The installed runtime exposes `prefix_match_unit` independently of physical page policy and an opt-in shared-prefix checkpoint for aligned Mamba/EAGLE paths. The two cache controls do not change the compilation hash. The MTP count also changes recurrent-state staging and the available cache budget; `cache-capacity.json` retains the actual runtime reports. No new weights, image or quantization are introduced.

## Correctness and cache gates

Every profile passed repeated and divergent prompts, 46K retrieval, four concurrent short responses and a complete function call/result exchange. The short four-caller probe measures throughput. Following review, a separate concurrent-cache gate was added: four callers fork a warm long conversation, each must produce its own expected answer and reuse more than 90% of its input. The MTP5 receipt contains 46000 cached tokens for each branch. The exact selected MTP4 profile passed the stronger acceptance probe before rollout (`selected-acceptance.json`): two different 31K-token parent documents were primed, then four concurrent branches had to recover the correct parent reference and add their own branch marker. All four answers were exact, with 31360–31376 cached tokens each. Streamed function-call arguments were reconstructed and checked, followed by a correct tool result. A 120171-token retrieval passed cold in 101.58 seconds and repeated in 0.392 seconds with 120144 cached tokens. The receipt binds these checks to the image digest, MTP4 environment and original adapter hash.

The first acceptance attempt asserted a bare tool value although the instruction did not require that output format. The returned value was correct. The prompt was corrected to require the exact value explicitly, and every gate was rerun; `acceptance-harness-correction.json` records the reason.

`engine-trials.json` contains the raw provider usage, times and metrics. Reproduce it with `kubectl -n apps exec -i <pod> -c <container> -- /app/venv/bin/python - <profile-label> < benchmarks/qwen-pi-cache-2026-11-20/engine_probe.py` once the relevant worker is loaded on loopback port 5894. This full probe belongs in an exclusive GPU window.

## Client investigation

Initial server counters reported 628848 cached tokens from 749842 queried tokens (83.86%). Studio Pi telemetry recorded 2231856 cached input tokens and 261033 new input tokens across 44 calls (89.53% reuse). Those scopes differ and are not controlled before/after measurements.

A real three-turn Pi parser task passed five independent behavior checks. Provider payload fingerprints showed stable tool definitions and append-only message history on every call. The baseline took 5.03, 15.44 and 3.40 seconds by user turn. The Pi-Lens retained-injection trial took 4.78, 10.87 and 3.04 seconds, but followed a different generated tool trajectory. No transient injection broke this probe's prefix; these timings do not establish a retention-mode speedup, so no client setting was changed. Raw receipts retain usage and fingerprints without prompt bodies or credentials.

Pi's model metadata intentionally has no invented time-to-live: this cache is capacity/eviction based. Adding a nominal five-minute lifetime would enable background warm requests without documenting an actual server expiry. The resident worker and existing aligned prefix caching remain enabled.

## Production verification

The authenticated Tailnet endpoint measured 126.99 output tokens/s on the same short-stream prompt, with 0.115 seconds to first output. A repeated ~3175-token request completed in 0.211 seconds with 3152 cached tokens. `production-performance.json` retains these live endpoint receipts; they are separate from the temporary-worker comparison.

Server, Studio, Pro and Ravn each launched a real Pi parser-editing task without a model override. Every client selected its shared home-llama/qwen3.8-pi default and passed five independent behavior checks of the edited file. Each recorded cached tokens across the tool loop. End-to-end times ranged from 13.62 to 54.70 seconds, including Pi startup, extensions and tool execution; those are not decoder throughput measurements. `production-client-verification.json` contains the per-machine receipts. All temporary client fixtures were removed.

`production-status.json` confirms the actual ready pod/image digests, MTP4 and fine-cache configuration, native CUDA GDN path, 172599-token cache seat budget, reconciled/unsuspended apps and infra, and a Prometheus scrape from the current pod. The temporary qualification pod was deleted and the maintenance watchdog was stopped after service restoration.

The shared GPU cache is capacity bounded. Other model selections unload Pi and reset its GPU cache; a new or evicted prompt still pays cold-prefill cost. No background keepalive or extra client prefix rewrite was introduced.

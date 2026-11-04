# Swift production qualification — 2026-09-15

> **Archived 2026-11-04** (moved from the GitOps repo's `k8s/workloads/apps/llama/`). Records the
> 2026-09-15 Swift Q4_K_S profile on `llama:cuda-swap-v25`. That profile and the image tag are gone; the ukisai weights remain under `/data/models/qwen3.8/ukisai/`.

## Selected profile

The existing `qwen3.8-27b` model ID now selects ukisai Swift Qwen3.8-27B Q4_K_S, with Q4_K_M DFlash2 draft weights and seven draft proposals. The engine remains `llama:cuda-swap-v25`; experimental CUDA patches are not included.

Context is 153,600 tokens, with Q8 K/V caches for both target and draft. The target cache allocates 5,100 MiB. One slot serves requests sequentially, preserving the full context for each request. Batch/microbatch sizes are 512/128, the vision projector runs on CPU, and xhigh reasoning is the default. The default output budget is 16,384 tokens; the existing advertised 32,768-token maximum remains available to explicit requests within the total context limit. The model ID and authenticated endpoint remain unchanged.

The previous production profile used base Qwen3.8 UD-Q4_K_XL-v3, a 131,072-token Q4 cache, and two slots sharing that pool. Its model files and engine remain available for rollback.

## Measured selection

All measurements used the RTX 3090, driver 595.91.07, fixed requests and graders, seed 0, and the same cache/context constraints. Percentages below compare against the tested Swift Q4_K_M + three-draft Q8 control, not directly against the previous production configuration.

| Full-suite metric | Swift Q4_K_M, 3 drafts | Selected Q4_K_S, 7 drafts | IQ4_XS, 7 drafts |
|---|---:|---:|---:|
| Original quality checks | 48/48 | 46/48; two grader timeouts | 48/48 |
| Sum of 52 request durations | 877.03 s | 838.53 s | 820.48 s |
| Fixed-probe decode gain | Control | +8.49% | +4.61% |
| Peak sampled GPU memory | 23,928 MiB | 23,324 MiB | 22,516 MiB |
| Cold 150,036-token retrieval | Pass | Pass | Pass |
| Time to first token for that prompt | 253.94 s | 250.77 s | 243.05 s |

The selected profile's two timed-out saved answers passed the unchanged sandbox grader twice each, in 0.20–1.21 seconds. No answer was edited or regenerated. All 48 saved answers therefore have passing checks, while the original timeout records remain preserved. A repeat control also encountered a transient grader timeout; its unchanged answer passed twice.

Fresh-process screening confirmed a 9.04% decode gain for Q4_K_S. Each profile reproduced all ten of its first-screen text/reasoning outputs. Q4_K_S passed all six repeat quality checks. Its full-suite request-time saving was only 4.39%, because it generated more output than the control. IQ4_XS completed the full suite 2.15% sooner than Q4_K_S, but had lower decode throughput and more aggressive weight quantization. Q4_K_S is selected for the decode-focused workload; it is not claimed to be fastest on every metric.

This is a limited one-seed qualification, not proof of general quality equivalence. Request-time sums exclude loading, grading containers, restoration, and the separate long-context request. Decode gain is the geometric mean of four matched fixed-probe ratios.

## Remaining optimizations tested

- **Native MTP2/3 and MTP graph patch PR28549:** passed their six-case screens but did not improve fixed-probe decode over the DFlash control.
- **GSQ/RCO IQ3_S:** slower overall. DFlash screening failed one output-budget case; native MTP passed the screen but remained slower.
- **GDN chunked prefill, PR26001:** adapted to the installed source and passed 51 CUDA-versus-CPU numerical cases. Seven-draft speculation uses eight recurrent states, whereas the new kernel requires one; the speculative profile falls back to the original kernel. Its 150K first-token time remained 250.00 s. A separately matched no-speculation test exercised the eligible profile: first-token time fell from 243.89 to 235.48 s (3.45%), with roughly 41–44 tok/s short-context decode. Both no-speculation profiles scored 5/6, failing the same histogram check. Not selected.
- **Q4_K gate/up fusion, PR28702:** adapted and compiled; 139/139 focused numerical cases passed. The 150K retrieval passed at 244.31 s, but screening scored 5/6: generated histogram code raised IndexError. It also reduced several decode-probe rates. Not selected. This is an observed generated-answer failure, not a demonstrated CUDA arithmetic bug.

## Production-path acceptance checks

The checksum-verified files were loaded from the production model volume with the selected runtime and defaults. Both existing image regression requests passed, and both images passed again using the production-default xhigh setting. Two distinct 150K requests submitted concurrently both returned all three exact codes: 150,036 and 150,033 prompt tokens, with no truncation. They serialized correctly in the one slot; the pair completed in 507.20 seconds. The trial then restored the original deployment and Flux state without errors.

Cold loading from the model HDD is slower than loading the benchmark copies; disk throughput during the production-path load was approximately 88 MiB/s. Model loading is excluded from the inference comparisons above. The profile keeps a one-hour warm-model TTL.

## Provenance and rollback

Swift repository: https://huggingface.co/ukisai/Swift-Qwen3.8-27B-GGUF

Pinned revision: `b12fc8f44af7bdf31a3c3309891acb345d3106a2`.

- Q4_K_S SHA-256: `974812e92df1ce82d91aa7aafa466ef1b707c91fdd3074dfb29438aa59d279d3`
- F16 projector SHA-256: `daa1116c9422fa390cc8688495da0e91781f92841dfc3b31a378ff252571745a`
- Runtime Docker image ID: `sha256:b4010615d24f3f928afc1b9fb2136e1f9782c48ef7e22b6573f3b334b708d224`

The fetch manifest pins the revision and checksums. Staged production files passed full checksum verification. Raw requests, responses, build identities, numerical tests, and failure records remain on the server under `/data/docker-services/benchmarks/swift-next-2026-09-15/`.

Rollback: restore the previous `qwen3.8-27b` configuration from commit `39975a323005ed68bd4ed660384a5c7597fa40f9` through GitOps, or revert this promotion commit and reconcile Flux. No previous weights or runtime image were removed.

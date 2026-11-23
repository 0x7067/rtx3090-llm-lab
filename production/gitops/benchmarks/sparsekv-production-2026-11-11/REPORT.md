# SparseKV versus production HyperQwen — 2026-11-11

SparseKV improved the tested long-conversation workload, especially two clients
sharing the GPU. It delivered **3.0× the end-to-end output throughput** for two
~124K-token clients and reduced the time to resume a conversation after another
chat from **5.83 s to 1.25 s**. It was **15% slower at 32K single-request decode**.
There is no general speedup across all tested workloads.

Both configurations passed the small correctness screen. **Production was
restored to its original image and configuration and verified by inference;
no candidate was promoted.** SparseKV merits further qualification for long,
concurrent conversations. These measurements do not justify replacing the
default for every workload.

## Measurements

Higher decode throughput is better. Single-request decode is the inverse of
vLLM's per-request time-per-output-token metric. The two-client figure is
aggregate output divided by complete client wall time, including waiting and
prefill; it is deliberately a different metric.

| Workload | Production | SparseKV | Change |
|---|---:|---:|---:|
| 4K decode, median of 3 | 86.1 tok/s | 83.2 tok/s | −3.3% |
| 32K decode, median of 3 | 100.2 tok/s | 85.2 tok/s | −15.0% |
| ~124K warm single-request decode, one sample | 91.6 tok/s | 99.4 tok/s | +8.6% |
| Two ~124K clients, aggregate end-to-end output | 29.9 tok/s | 89.5 tok/s | 3.0× |
| Two-client batch duration, 512 total output tokens | 17.14 s | 5.72 s | −66.6% |

Lower first-token latency is better.

| Request | Production | SparseKV |
|---|---:|---:|
| 4K cold, median of 3 | 1.85 s | 1.92 s |
| 32K cold, median of 3 | 16.78 s | 17.18 s |
| ~124K cold, median of 2 | 95.70 s | 97.33 s |
| Same long conversation, follow-up | 1.80 s | 1.21 s |
| Return to long conversation A after B | 5.83 s | 1.25 s |
| Two-client first token, client A | 14.43 s | 1.97 s |
| Two-client first token, client B | 6.01 s | 2.20 s |

On return after B, production reused 119,232 prompt tokens and SparseKV reused
123,552 of the same 124,069-token prompt. Both returned the correct answer.
The complete paired values are in [comparison.json](run-3/comparison.json).

## Correctness and comparison contract

| Check | Production | SparseKV |
|---|---:|---:|
| Arithmetic, tool dispatch and tool continuation | 3/3 | 3/3 |
| Executable code, tool sequencing and JSON/retrieval | 5/5 | 5/5 |
| Long-context retrieval | 4/4 | 4/4 |

All six speed samples and three long decode-budget checks also completed in
both arms. Fixed-budget decode samples intentionally reach their token cap;
they are throughput checks, not completed-answer quality scores.

This compares **complete serving configurations**, not an isolated attention
kernel. Production uses HyperQwen DFlash2-7 with its 16 GiB CPU prefix cache.
The candidate uses SparseKV's MTP-7, FA2 prefill, INT4 key sketches and INT4 host
KV tier. Candidate startup confirmed a 17.85 GB host KV pool plus 2.73 GB of
pinned GDN checkpoints. Target weights, INT8 output head, chat template, test
inputs and sampling were held constant. Both retained vision configuration,
but this screen did not send images.

Both used the same RTX 3090 at the production 290 W profile, a 200,000-token
configured context, four admitted request slots, 12 CPUs, a 36 GiB container
memory cap and a 24 GiB shared-memory ceiling. Each used a private runtime
cache. Minimum host available RAM observed was 8.91 GiB for the baseline and
9.18 GiB for SparseKV; other host workloads continued running, so these are
headroom observations rather than isolated memory-cost comparisons.

The 4K/32K tests contain three samples each. The long-context sequence was run
once per arm: A cold, A follow-up, warm solo decode, B cold, return to A, then
two simultaneous clients. Prompts actually contained 124,003–124,124 tokens.
Speed, quality and long-context request hashes match across arms. The smoke
tool-follow-up hash differs only in the server-generated tool-call ID;
[the audit](run-3/tool-followup-hash-audit.json) confirms otherwise identical
content and tool arguments.

This is a small synthetic screen with baseline-first ordering, not a
statistical significance test or proof of equivalent output quality. It does
not test four clients, 250K contexts, long coding tasks, images, or cancellation.
No performance claim is made for those cases.

## Versions and evidence

- [SparseKV 29595e8712520ef017c51bcccb9ab6dbc3c68a01](https://github.com/bradj00/sparsekv-hyperqwen/tree/29595e8712520ef017c51bcccb9ab6dbc3c68a01), applied to [HyperQwen ce9615c4f2309be727407918a6c979bd67968069](https://github.com/syv-ai/HyperQwen/tree/ce9615c4f2309be727407918a6c979bd67968069).
- Production HyperQwen: `2cf50a01d9d43fe9b6e031d3718bd98ebffb6938`; vLLM 0.31.0, Torch 2.13.0 and Triton 3.7.1 in both arms.
- [Candidate manifest and source pins](candidate/manifest.json), [production snapshot](production/manifest.json), and [exact executed runner sources](run-3/runtime-source/).
- [Raw baseline](run-2/baseline/), [raw candidate](run-3/candidate/), [combined summary](run-3/summary.json), and [resource observations](resource-summary.json).
- [Final production verification](final-verification.json): original container specifications and model configuration, one available replica, successful inference, all Flux Kustomizations ready and unsuspended, no benchmark containers, and original image-retention labels restored.

The first candidate launch was interrupted by kubelet image garbage collection,
not by model inference. Re-pulling and unpacking the large production image
exceeded the automatic restore verifier's deadline; Kubernetes continued the
restore, and a subsequent verifier confirmed a correct answer before the retry.
The retry temporarily retained the exact three deployment images against
kubelet GC and reused the completed baseline with the same trial ID. Temporary
retention was removed after successful final restoration. Logs of both attempts
are retained; no global GC or production resource policy was changed.

Repository tests, supply-chain checks, both Kustomize renders and shell lint
passed. The existing DSH ingress test was skipped because Caddy is not installed
on PATH. See [checks.log](checks.log). GPU results and restoration were checked
on the actual server, not inferred from the CPU preparation tests.

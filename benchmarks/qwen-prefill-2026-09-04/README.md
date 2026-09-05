# Next Qwen experiment: prompt processing and compute memory

Follow-up campaign after v18 deployment, run with the user's authorization for
benchmark windows and automatic restoration. The cold-prompt screening test uses
24,587–24,590 actual tokens. The generator's 50,000-token request is an estimate;
reported throughput uses the tokenizer count.

The original 70,843-token session spent 77.90 seconds before the first response
token. This motivated testing prompt batches and compute-memory allocation while
preserving the target weights, reasoning, and 131k context.

## Result: keep v18

| Configuration | Cold prefill tok/s | Eight-turn editing seconds | Concurrent tok/s |
| --- | ---: | ---: | ---: |
| v18 control, first run | 1114.5 | 54.61 | 117.62 |
| v18 control, closing repeat | 1089.6 | 54.68 | 118.26 |
| Batch/ubatch 1024, two slots | 1133.5 | 59.55 | 121.66 |
| Ubatch 256, two slots | 1068.8 | 56.69 | 118.65 |
| Ubatch 256, three slots | 1066.2 | 56.93 | OOM |
| cuBLAS for prompt batches >=256, two slots | 988.4 | 61.13 | 113.16 |
| Ubatch 128, three slots | 938.4 | 59.02 | 124.62 |
| 8 MiB sorting chunks, ubatch 512, three slots | OOM loading | — | — |
| 8 MiB sorting chunks, ubatch 256, three slots | 1086.0 | 56.85 | 121.59 |
| 8 MiB sorting chunks, ubatch 512, two slots | 1091.0 | 60.39 | 114.37 |

Eight candidate configurations were screened; none was promoted. These are
single-run candidate measurements, not confidence intervals. The two-slot sorting
arm had a localized slowdown on turns 3–5; its aggregate result is insufficient to
attribute that slowdown to the patch, and it provides no evidence of a speed win. No configuration above clears
the 5% improvement gate while preserving balanced coding performance. The control
editing times differ by only 0.13%; cold-prompt throughput varies by about 2.2%.
The small prefill gain from batch 1024 comes with a roughly 9% editing slowdown.
Three slots at ubatch 128 trade about 6% concurrent throughput for 8% slower editing
and substantially slower prompt processing. They are not the default recommendation
for the user's mix of coding sessions and concurrency.

The sorting-memory patch is opt-in and leaves the original 64 MiB chunk limit as
the default. At 8 MiB it resolves the runtime CUB scratch allocation failure for
three slots at ubatch 256, but the measured throughput gain is only about 3% while
editing is 4% slower. It cannot fix the separate 553.90 MiB startup compute-buffer
allocation failure at ubatch 512. No patch from this campaign has been promoted.

## Correctness

Loaded arms passed the four functional/tool probes. Completed generated-file
sessions passed the separate eight-file executable behavior checks; incomplete concurrent requests do not count as passing
quality results. Any promotion would require the existing vision, 125k retrieval,
overlapping-context stress, and deep-session gates.

The sorting patch passed the standard TOP_K/ARGSORT CUDA-vs-CPU operator suite,
including tied inputs and 202,048-column, 16-row cases. These exceed the new 8 MiB
chunk bound and exercise multiple chunks. The added exported generic-op tests in
`sort-cases.txt` were unsuitable: they compare raw indices under tied values and
failed on both the unchanged v18 and candidate. Both failure logs are retained;
they are excluded as evidence for or against the patch. The upstream specialized
operator tests handle ties and ordering according to the operator's contract.

## Reproduce the first sweep

All arms retain v18, the current target and Q4 drafter, CPU vision, medium reasoning,
131072 context, and target/draft KV precision. The reference is two slots with
batch/ubatch 512/512. Arm tags distinguish these runs from the completed campaign.

1. Reference: 512/512, two slots.
2. 1024/1024, two slots: test whether wider prompt batches improve GPU utilization.
   Reject on OOM or correctness failure.
3. 512/256, two slots: measure the speed/memory cost of smaller physical batches.
4. 512/256, three slots: test whether compute-memory savings resolve the previous
   553.90 MiB allocation failure and improve distinct-request concurrency.

The existing run-arm.py now accepts --batch, --ubatch, and --prefill-tokens.
For example, from the completed campaign directory:

```sh
python3 run-arm.py prefill-v18-control-r1 --image llama:cuda-swap-v18 --q4-draft --cpu-vision --parallel 2 --prefill-tokens 50000
python3 run-arm.py prefill-v18-u1024-r1 --image llama:cuda-swap-v18 --q4-draft --cpu-vision --parallel 2 --batch 1024 --ubatch 1024 --prefill-tokens 50000
python3 run-arm.py prefill-v18-u256-r1 --image llama:cuda-swap-v18 --q4-draft --cpu-vision --parallel 2 --ubatch 256 --prefill-tokens 50000
python3 run-arm.py prefill-v18-u256-p3-r1 --image llama:cuda-swap-v18 --q4-draft --cpu-vision --parallel 3 --ubatch 256 --prefill-tokens 50000
```

These commands require exclusive GPU use; do not run them beside production.
Capture deployment replicas and Flux suspension state before the window. Suspend
reconciliation, scale llama down, and wait for GPU release. Always remove the trial
container and restore the captured deployment/reconciliation state after the sweep,
including on failure. Confirm a live model response after restoration.

Each arm includes a nonced cold prompt, the existing four functional probes, eight
complete-file edits, four distinct concurrent tasks, and fresh generation. Use
reported prompt token counts, not the approximate --prefill-tokens input. Execute
the generated-file behavioral checks. Repeat any promising arm and the reference
three times, then validate 125k recall, vision, overlapping long requests, and a
70k-start editing session before promotion. A shorter prefill alone does not pass
if session quality, context support, or useful concurrent throughput regresses.

## Kernel lead reviewed

[llama.cpp PR 27140](https://github.com/ggml-org/llama.cpp/pull/27140) proposes
vectorized conversion of small KV quants to f16. Its author reports q4 prefill
recovering from 74 to 1182 tok/s on two 3090s. It remains unmerged at inspected
head 665dac0f110f730af851ab27bdb597ab1d94e3ad. Those claims are not local results:
our current prefill is already near that range at moderate depth. Profile before
paying for a build, and verify conversion outputs plus end-to-end behavior if tried.

[PR 27150](https://github.com/ggml-org/llama.cpp/pull/27150) addresses mixed K/V
precision dispatch and was closed unmerged. The current profile uses matching
q4_0 K/V, so this is not a reason to change it.

## Experimental builds and artifacts

`Dockerfile.cublas` and `prefill-cublas.patch` add an opt-in large-batch dispatch
switch, enabled with run-arm.py `--prefill-cublas 256`. Image:
`llama:qwen-prefill-cublas-v1`, ID
`sha256:d90105de6f79b4d7bf2f551bc0690da833aa4a6e2b0f7130572a8a5e0e3749bd`.
It uses the existing cuBLAS path; no target weight conversion is persisted.

`Dockerfile.scratch` and `sort-scratch.patch` expose the existing sorting chunk
limit, enabled with `--sort-chunk-mib 8`. Image:
`llama:qwen-sort-scratch-v1`, ID
`sha256:3026184e02a43c89044f531434342fc33444ee50d16cfc574e45a4cdff454d8b`.
Both overlays replace only libggml-cuda.so over the exact v18 runtime. Their build
stage is the existing llama:qwen-speed-build from the previous campaign. Both
switches default to the original behavior. Neither image was deployed or published
to the production registry.

Raw measurements, command snapshots, server logs, and generated code use tags
prefill-v18-*, prefill-cublas-*, and sort8-* in
`../qwen-speed-2026-09-04/`. They are separate from that campaign's selected control
and candidate tags. `decision.tsv` is the local, gitignored decision log.

The maintenance wrapper captured replicas and Flux suspension state before each
window and restored them in a finally block, including the failed operator-test
window. After the final window on 2026-09-05, llama was restored to v18, one healthy
replica, with Flux apps reconciliation resumed. The final live functional/tool
probe results are recorded in `restoration-checks.jsonl`.

Further work should begin with a GPU trace of real long-context requests. The
completed sweeps leave v18 as the selected configuration.
The remaining larger bets are a dedicated low-memory top-k implementation or better
drafter acceptance, both requiring more work and a fresh end-to-end quality gate.

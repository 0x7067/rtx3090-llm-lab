# Qwen3.8 27B on RTX 3090: speed qualification, 2026-09-04

The selected configuration improves four distinct concurrent coding requests by
about 54% and reduces an eight-turn editing session by about 7%. A session starting
at 70,843 tokens is effectively unchanged. These results apply to the existing
UD-Q4_K_XL-v3 target, not to an unquantized model.

The target weights, target/KV precision, sampling, medium reasoning default, and
131,072-token single-request limit are unchanged. The changes are a Q4_K_M DFlash2
drafter, two slots sharing one KV pool, and CPU execution of the existing vision
projector. A server admission patch queues requests whose combined prompt/output
budgets cannot fit. Without it, overlapping long requests can fail with HTTP 500.

## Results

| Workload | Original v17 | Candidate | Interpretation |
| --- | ---: | ---: | --- |
| Eight complete-file editing turns, median of 3 | 59.15 s | 54.69 s | 7.5% less wall time |
| Four distinct concurrent coding requests, median of 3 | 76.19 tok/s | 117.31 tok/s | 54.0% higher throughput |
| Fresh 2048-token generation, median of 3 | 102.45 tok/s | 102.11 tok/s | Essentially unchanged |
| Eight editing turns starting at 70,843 tokens, one paired run | 150.79 s | 151.38 s | Essentially unchanged; 0.4% longer |

The final candidate medians use guard-r1/r2/r3, including the admission patch.
Their session times were 54.69/54.66/54.91 s and concurrent rates were
117.31/113.52/123.84 tok/s. Three earlier unguarded repeats were consistent
(54.77 s and 117.55 tok/s medians), but the capacity failure rules out deploying
them. Raw results are in `screening.jsonl`. Control medians use control-r2/r3/r4.
Deep results use control-deep-r1
and guard-deep-r1. Do not use raw decode rate from copy-heavy edits as a general
chat speed claim: ngram speculation benefits repeated code substantially.

The GPU reached roughly 23,300 MiB used. No clock or power changes were made.
Hardware: RTX 3090 24,576 MiB, i9-10850K, 64 GB RAM, NVIDIA driver 595.71.05.

## Quality and capacity checks

- Four probes: arithmetic, Portuguese structured output, Python repair, tool call.
- Eight complete generated Python files exercised in a network-disabled, read-only
  container: task insertion/replacement, status, deletion, priority filtering,
  JSON export, and requeue. Short and deep sessions passed all eight.
- Three synthetic vision fixtures passed all ten criteria with GPU and CPU vision.
  Individual CPU-image response times were approximately 3.0–6.8 seconds, close
  to the original in these small fixtures. Larger images are not characterized.
- Exactly 125,000 prompt tokens: correct recall of a code from the first line,
  with no truncation (165.08 seconds on the published v18 image). This checks capacity/retrieval, not general long-context reasoning.
- Two simultaneous 70,000-token prompts requesting 8,192 output tokens each:
  unguarded server completed only one; guarded server completed both without
  truncation, serializing them (about 101 and 204 seconds from submission).
- The exact serving profile passed an end-to-end request through llama-swap in
  the published image, using the persistent drafter (`published-profile-check.json`).
- The guarded image's CUDA, model, and vision shared libraries have exactly the
  same SHA256 values as v17; see `guard-library-hashes.json`.

Changing the drafter does change some deterministic continuations: only three of
eight short-session files matched the original byte for byte, although all passed
behavioral checks. These tests found no quality regression; they do not establish
universal equivalence. The deep paired run produced eight byte-identical responses. Target fine-tuning
or a smaller target quant was not accepted.

The 131k pool is shared, not 131k simultaneously per slot. Requests with large or
unspecified output budgets may serialize. This preserves requested budgets rather
than reducing them to force concurrency. Applications supplying bounded output
budgets can benefit from parallel execution when both requests fit.

## Experiments and research

The local, gitignored `decision.tsv` records eight hypotheses: three retained components and five rejected
configurations. Q8/two slots and Q4/two slots with GPU vision ran out of memory.
A third slot also failed to load (553.90 MiB draft compute allocation failed).
Draft length 4 was slower than 7. Raising ngram draft length from 64 to 96 did not
clear the 5% editing-session screening threshold.

Primary sources consulted:

- [Qwen's model card](https://huggingface.co/Qwen/Qwen3.8-27B): sampling and reasoning
  guidance; lower reasoning can increase agent retries.
- [DFlash2 3090 measurements](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/discussions/9):
  contributor reports motivated testing shorter drafts; our patched build measured
  the opposite outcome. GGUF metadata has block size 8, so lengths above 7 clamp.
- [DFlash2 GGUF distribution](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF):
  smaller proposal weights save about 850 MiB. Download/source metadata is saved
  in `drafter-source.json`; the same verified file exists in the existing pinned
  z-lab repository, so serving does not require another distribution dependency.
- [llama.cpp changes since the pinned base](https://github.com/ggml-org/llama.cpp/compare/0f3a71be1...427291b5b34c):
  reviewed 56 commits; no clear single-Ampere speed fix justified replacing the
  qualified CUDA patches wholesale. The accepted update changes server admission.
- [Pinned server documentation](https://raw.githubusercontent.com/ggml-org/llama.cpp/0f3a71be1/tools/server/README.md):
  unified KV and per-slot limits. Capacity was validated on the running server.
- [DFlash training implementation](https://github.com/z-lab/dflash): training a better
  drafter remains a future experiment, requiring an acceptance/latency win.

Prior campaigns in `../engine-trial-2026-09-02`,
`../qwen-vllm-hillclimb-2026-08-28`, and `../../docs/quant-selection.md` already
cover alternate engines, quantization, and kernels. Smaller target IQ4/Ridge
candidates worsened divergence. A prior fused GDN kernel passed isolated checks
but damaged MTP agreement. Neither is promoted on speed alone. The next substantial
experiment would target the roughly 78-second cold prefill at 71k context, with
full end-to-end quality checks; another small decode setting is unlikely to help it.

## Reproduce and deploy

Run from this directory, with the existing model volume available:

```sh
python3 run-arm.py control-new
python3 run-arm.py guarded-new --image llama:cuda-swap-v18 --q4-draft --parallel 2 --cpu-vision --qualify --context-check
python3 run-arm.py guarded-stress --image llama:cuda-swap-v18 --q4-draft --parallel 2 --cpu-vision --checks-only --stress-context-check
python3 check-files.py guarded-new-session-outputs.jsonl
python3 check-profile.py
```

The harness expects `/tmp/qwen-speed/DFlash2-Q4_K_M.gguf` (copy or link the verified
production drafter there). It starts a disposable server on loopback port 18089
and removes only its own container. Each arm has a fresh ngram history. Its four
concurrent prompts are distinct Python, TypeScript, Rust, and Go tasks; their
512-token capped outputs are throughput probes, not quality scores. Complete-file
sessions allow 4096 tokens and record finish reasons and full outputs.

`baseline.jsonl` is excluded: its original 1536-token cap truncated files despite
passing substring checks. Control-r1 concurrency is also excluded because identical
prompts exaggerated ngram reuse. `screening.jsonl` contains valid later measurements
as well as explicitly identified exploratory records; select tags as above.

The canonical patch is `../../patches-v15/0011-server-unified-kv-admission.patch`;
the normal lab Dockerfile applies it with the existing patches. For this experiment,
`Dockerfile.guard` used a build stage made from lab revision f5c696c (before patch
0011), then replaced only libllama-server-impl.so in the exact v17 runtime. Do not
apply the overlay again to a build stage that already contains 0011.

Published runtime: `127.0.0.1:5000/llama:cuda-swap-v18`.
Registry digest: `sha256:23da58507732c60e658bf21ddabb56e1fc9509387fc9730e0a3e39c852b63336`.
Local image ID: `sha256:4d7b15bd0bbdef496415363a5e9693757ae379d8d5106fad6d25373913ab5fc4`.
Drafter SHA256: `1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd`.

`candidate-profile.yaml` and `../../config/llama-swap-qwen38.yaml` contain the serving
profile. The parent repository's llama ConfigMap and deployment select it and v18.
At inspection, the live deployment was already at zero replicas and Flux apps
reconciliation suspended. These experiments leave that maintenance state intact;
activation awaits the user's answer. The image is published and the drafter is
already present in the persistent model volume. Rollback is the original v17 image
and profile from parent revision ac71891; the original Q8 drafter remains available.

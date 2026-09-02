# RTX 3090 LLM Lab

This repository collects the measured runtime, model, and benchmark work for
running coding models on one 24 GB RTX 3090. Qwen3.8-27B is the main deployment.
Tiel-Coder-35B-A3B is retained as a deployment-fit comparison and llama.cpp
alternative.

## Repository map

- [`benchmarks/tiel-vs-qwen/`](benchmarks/tiel-vs-qwen/) contains the public
  Tiel-versus-Qwen comparison, corrected harnesses, and later qualification
  evidence.
- [`benchmarks/qwen-vllm-hillclimb-2026-08-28/`](benchmarks/qwen-vllm-hillclimb-2026-08-28/)
  contains the frozen control, 16 optimization decisions, and measured results
  from the vLLM slowdown campaign.
- [`benchmarks/engine-trial-2026-09-02/`](benchmarks/engine-trial-2026-09-02/)
  is the trial that promoted llama.cpp v14 back over vLLM v10 in production:
  llama.cpp master + four drafters, SGLang, and the vLLM baseline, on one
  harness.
- [`patches-v14/`](patches-v14/) is the **current production** llama.cpp patch
  set (base `0f3a71be1`). [`patches/`](patches/) /
  [`patches-v9-v12-base-4df29be4/`](patches-v9-v12-base-4df29be4/) is the
  superseded set behind images v9–v12 (base `4df29be4f`); both names hold the
  same eight files, kept so the base commit is unambiguous.
- [`vllm/`](vllm/) contains the syv-ai overlay, the v9 image overlay once
  deployed ([`vllm/image-v9/`](vllm/image-v9/)), the exported v10 and v11
  branch diffs ([`vllm/image-v10/`](vllm/image-v10/),
  [`vllm/image-v11-vllm028/`](vllm/image-v11-vllm028/)), and the Club 3090
  bundle. vLLM v10 was production from 2026-08-20 to 2026-09-02; llama.cpp v14
  is production now.
- [`research/vllm/`](research/vllm/) records upstream findings that still need
  a local A/B before promotion.
- [`experiments/`](experiments/) contains rejected or unfinished implementation
  work. Nothing there is a production default.

## Where things live

This repo (`0x7067/rtx3090-llm-lab`) is canonical for benchmarks, llama.cpp
patches, and the write-up docs — everything above and below this section.
Two other repos hold adjacent pieces:

- [`0x7067/qwen38-27b-rtx3090`](https://github.com/0x7067/qwen38-27b-rtx3090)
  is a fork of [`syv-ai/qwen38-27b-rtx3090`](https://github.com/syv-ai/qwen38-27b-rtx3090),
  kept only as the vehicle for upstreaming PRs to syv-ai. Its deploy branches
  (`local/k8s-deploy-v10`, `local/k8s-deploy-v11-vllm028`, …) are not merged
  anywhere upstream; they're exported here as diffs under `vllm/image-v10/`
  and `vllm/image-v11-vllm028/` rather than lived in as a submodule.
- [`0x7067/tiel-bench-rtx3090`](https://github.com/0x7067/tiel-bench-rtx3090)
  is archived. Its content is already merged into this repo's
  `benchmarks/tiel-vs-qwen/`.
- `/data/docker-services` `k8s/workloads/apps/llama/` in the GitOps repo holds
  the actual Kubernetes deployment manifests (`deployment.yaml`,
  `configmap.yaml`, the built `image/`) and their running change log
  (`k8s/MIGRATION_LOG.md`). Those stay in the GitOps repo because Flux
  reconciles from there directly; this lab mirrors the parts worth keeping
  reproducible outside that repo (patches, benchmarks, design docs) and
  cross-links rather than duplicating the manifests wholesale.

## Qwen3.8-27B llama.cpp stack

The llama.cpp stack targets Qwen3.8-27B, a 27B hybrid Gated-DeltaNet model. The
full 131k context stays resident with the main model, MTP drafter, and vision
projector at about 22.9 GiB. Speculative decoding uses the model's native MTP
head. The vLLM and llama.cpp lanes use different model formats, caches, and
benchmark harnesses, so their results remain separate.

Results on the target workload (54k-token-deep agentic decode, temp 0). The
patch/drafter campaign used the former Q4_K_L target; the last promoted
llama.cpp profile uses UD-Q4_K_XL, selected in the later quant sweep below:

- The patch stack takes decode from **49.8 to 69.3 tok/s (+39%)**. Mid-depth
  (1k) goes from **63 to 84 tok/s (+33%)**.
- The truncated draft vocabulary adds another **+5–6% at mid to 54k depth**
  and frees **0.5 GiB of VRAM**. Output stays byte-identical, verified by
  output sha256 in every A/B pair.
- Patch 0008 adds **+7% at 1k depth and +5.4–6.4% at deep decode** on top of
  that profile when `GGML_CUDA_MMQ_SMALLN=3` is paired with the MMVQ cap.

The approach is inspired by
[syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090),
which uses vLLM. This is the llama.cpp counterpart. It has no multi-minute
cold start, it hot-swaps models through llama-swap, and decode holds up at
54k depth.

## Current deployment

**As of 2026-09-02, the home deployment is llama.cpp again**, image
`llama:cuda-swap-v14` (llama.cpp master `0f3a71be1` + the six
[`patches-v14/`](patches-v14/) patches), promoted over the vLLM v10 profile
below on the strength of
[`benchmarks/engine-trial-2026-09-02/`](benchmarks/engine-trial-2026-09-02/):
`draft-dflash,ngram-mod` is ~2x vLLM on the agentic edit workload at every
depth (230/194/219 vs 127/108/114 tok/s), quality 4/4. Serving config:
UD-Q4_K_XL main + mmproj, q4_0 KV, 131,072-token context, Q8_0 DFlash2 drafter
(`--spec-draft-n-max 7 --spec-ngram-mod-n-match 32`), `GGML_CUDA_MMVQ_NE11_MAX=3`
/ `GGML_CUDA_MMQ_SMALLN=3`. vLLM's own profile keeps single-request decode
(129 vs 105–108 tok/s), prefill, and 4-way concurrency (262 vs 160 tok/s) — see
the trial README for the full arm-by-arm table and the rollback path.

**Between 2026-08-20 and 2026-09-02 the home deployment ran vLLM.** Its
qualified profile used vLLM 0.27.1 with the syv-ai patch stack and this repo's
v4 overlay (deployed as image `qwen38-27b-3090:v9`, two patches further on),
prepared W4A16 weights, FP8 KV, prefix caching, vision, a 140,000-token limit,
three-token MTP speculation, `GPU_UTIL=0.94` (177,282 GPU KV tokens), and a
24 GiB CPU KV tier. The retained MTP-3 / 2,048-token batch arm measured 104.02
tok/s shallow, 95.69 tok/s at 60k, and 70.41 tok/s at 100k. Four concurrent
requests reached 358.41 tok/s aggregate. By 2026-09-01 the deployed source had
moved on to v10 (rebased onto syv-ai `453104e`, vision-tower CPU offload on by
default) — see [`vllm/image-v10/`](vllm/image-v10/) for that branch's diff and
the last-deployed manifest; [`vllm/image-v11-vllm028/`](vllm/image-v11-vllm028/)
is a vLLM-0.28 candidate branch that was exported but never deployed.

The exact overlay and installable Club 3090 local bundle are under
[`vllm/`](vllm/). The wrapper passed a target-3090 boot and generation canary
on 2026-08-22. See [`docs/vllm-companion.md`](docs/vllm-companion.md) for the
model assembly, matched arms, quality position, environment, and qualification
evidence.

The rest of this README documents the reproducible llama.cpp project. "Prod"
in its historical notes means the last promoted llama.cpp profile, not the
current vLLM service.

## OBLITERATED standalone variant

The Apache-2.0 `OBLITERATUS/Qwen3.8-27B-OBLITERATED` Q4_K_M checkpoint is a
standalone llama.cpp benchmark option, not a production deployment. On the
patched image, three-token MTP raised its controlled median decode from 42.4
to 64.1 tok/s (+51%). Do not compare 64.1 tok/s with this repo's stock-model
results or the vLLM lane; only the MTP-off to MTP-on pair is controlled.

Use [`config/llama-swap-qwen38-obliterated.yaml`](config/llama-swap-qwen38-obliterated.yaml)
for the backend fragment. See
[`docs/obliterated-variant.md`](docs/obliterated-variant.md) for the mainline
MTP trap, A/B conditions, correctness control, VRAM limit, and open leads.

## llama.cpp results

Conditions: temp 0, warm, `/completion` timings, `--spec-draft-n-max 5`,
`GGML_CUDA_MMVQ_NE11_MAX=3`. "Full drafter" and "d48k drafter" run the same
image (patches 0001–0007); only the drafter file differs.

| Payload (prompt depth) | Full drafter | d48k drafter | Delta | Acceptance (full → d48k) |
|---|---|---|---|---|
| short (98 tok) | 75.3 tok/s | 75.7 | +0.5% | .527 → .488 |
| mid (870 tok, code) | 75.0 | **79.6** | **+6.1%** | .532 → .532 (identical) |
| prose (6.8k tok) | 64.7 | **68.2** | **+5.4%** | .435 → .431 |
| agentic (53.4k tok) | 59.4 | **62.9** | **+6.0%** | .439 → .439 (identical) |
| VRAM at 131k ctx | 23,438 MiB | **22,892 MiB** | −546 MiB | |

Output content is byte-identical between the two drafters for every payload,
compared by sha256. This is expected from the code: llama.cpp's verify step
samples from the target distribution and keeps a draft token only when it
matches exactly. A smaller draft vocabulary can therefore change speed, but it
cannot change output. See `docs/truncated-draft-vocab-design.md`, section 4.

Patches 0001–0006 produced the earlier 49.8 → 69.3 tok/s campaign result.
`docs/PERFORMANCE.md` is the full write-up.

## llama.cpp main quant: UD-Q4_K_XL

The last promoted llama.cpp main-model quant moved from bartowski Q4_K_L to
**unsloth UD-Q4_K_XL**, chosen for quality: a KLD-to-Q6K sweep found
UD-Q4_K_XL at ratio **0.787** vs Q4_K_L's own 1.0 (lower is closer to the
Q6_K reference), at a ~3.4% pooled-step-time cost and a small VRAM *saving*. A faster
candidate (bartowski IQ4_XS, −7% pooled step time) was measured and rejected
because its KLD ratio (1.415) is a real quality regression, not a free lunch.
Full sweep, a rejected 3.7bpw extreme, and the corrected
weight-bytes cost model: `docs/quant-selection.md`.

## The patches

**Which set is current:** [`patches-v14/`](patches-v14/) is the set running in
production today (image `llama:cuda-swap-v14`, promoted 2026-09-02), rebased
onto llama.cpp master `0f3a71be1`. `patches/` (duplicated verbatim as
[`patches-v9-v12-base-4df29be4/`](patches-v9-v12-base-4df29be4/) so the
historical base is unambiguous) is the **superseded** set behind images v9–v12,
against base `4df29be4f`. Every reference to `patches/` or `image/patches/`
below this line, and everywhere in `docs/`, describes that historical
4df29be4-based campaign, not the running v14 set — see
[`patches-v14/REBASE-2026-09-02.md`](patches-v14/REBASE-2026-09-02.md) for what
changed in the rebase (0001/0006 dropped as superseded by upstream backend
draft sampling; 0003/0005/0008 rebased with behavioural notes; 0002/0004/0007
applied clean).

Apply the patches onto llama.cpp commit `4df29be4f`, in order, with
`git apply`. Do not use `git am`: patches 0002 and 0006 are plain diffs
without format-patch headers. The `Dockerfile` does all of this for you.

| Patch | What it does |
|---|---|
| 0001 | DFlash draft-side greedy fast path, plus `LLAMA_SPEC_PROF` instrumentation |
| 0002 | Ampere MMQ small-batch (J=16) tile config: 128×64 tiles for Q4_K/Q5_K |
| 0003 | GQA-batched FlashAttention vector kernel for quantized KV, batch ≤2, cc <8.9. Reads each KV block once per GQA group instead of once per Q head, with no F16 scratch |
| 0004 | `GGML_CUDA_MMVQ_NE11_MAX` env var. Caps the MMVQ→MMQ crossover per model. The measured crossover is N≈3.4 on this GPU and model; upstream's flat 8 overpays 24–38% on speculative verify batches |
| 0005 | MMA FlashAttention reads q4_0 K/V inline (cp.async → smem → dequant in place), so it no longer dequantizes the full cache to F16 every step. Batches 2–16, D=256, GQA >4 |
| 0006 | DFlash greedy fast path requires host logits. Fixes silent acceptance=0 with GPU draft sampling |
| 0007 | Truncated draft vocabulary for MTP drafters, using the EAGLE3 `d2t` idiom. A drafter GGUF can carry a reduced LM head plus a `d2t` I64 tensor that maps head rows to target token ids. Logits scatter back to full width with −inf elsewhere. The patch is a no-op for GGUFs without `d2t` |
| 0008 | MMQ small-batch composite, env-gated `GGML_CUDA_MMQ_SMALLN` (1=stream-k grid multiplier, 2=+y-tile double buffer, 3=+m=1024 shape gate). Verify at n=5: −3.0 ms/forward → +7% mid / +5.4–6.4% deep decode. Caveat: the config-table half is compile-time and unconditional — env-off is not stock; see `docs/journal-2026-08-17.md` |

## Truncated draft vocabulary (patch 0007 and tools)

The Qwen3.8 MTP drafter has about 3.0B parameters. 2.5B of them sit in two
248,320-row embedding matrices. The input embedding is a row gather, which is
cheap. The output head is a full 0.7 GB matrix-vector product on every draft
step: **74% of the draft pass**. Truncating the head to the 49,152 tokens that
cover ~98.5% of real traffic cuts per-step draft traffic by ~62%. That becomes
the +5–6% end-to-end decode gain in the table above.

The pipeline has four steps:

1. `tools/build_draft_vocab.py` ranks token ids by frequency over a corpus
   shaped like your traffic. Ours was English technical text and code, with
   ~10% Brazilian Portuguese. The script force-includes all control and
   special tokens and the byte alphabet, then emits keep-set JSONs
   (32k/40k/48k).
2. `tools/eval_coverage.py` measures held-out and out-of-distribution
   coverage per slice. Pick the smallest set whose OOD coverage you can
   accept. We shipped 48k: 98.5% held-out, 96.5% OOD.
3. `tools/truncate_drafter.py` performs the GGUF surgery. It copies kept
   Q4_0 head rows byte-for-byte (a 5120-wide Q4_0 row is exactly 2880 B),
   keeps `token_embd` at full width (inputs are target-vocab ids), appends
   the `d2t` I64 tensor, and validates range, uniqueness, and sort order.
   `tools/validate_drafter.py` re-checks everything with 26 assertions
   against the source file.
4. Run the patched build with `--model-draft <truncated>.gguf`. To roll
   back, point `--model-draft` at the original file. The patched binary is a
   no-op without `d2t`.

## Reproduce on a fresh machine with one RTX 3090

Requirements: an NVIDIA driver compatible with CUDA 12.8 (R570 or later),
Docker with nvidia-container-toolkit, `jq`, and Python 3.10+. The GGUF surgery
uses only the standard library. Payload and vocabulary building also need
`pip install tokenizers`.

```bash
# 1. Build the image. No GPU is needed for this step.
#    It clones llama.cpp @4df29be4f, applies patches 0001-0008, and builds
#    for sm_86. The v11 tag is retained because the validation script uses it.
docker build -t llama:cuda-swap-v11 .

# 2. Download the models (~19.9 GiB total, links below) into a $MODELS dir:
#    Qwen3.8-27B-UD-Q4_K_XL.gguf, mmproj-Qwen3.8-27B-Q8_0.gguf,
#    and mtp-Qwen3.8-27B-Q4_0.gguf.
#    Also download tokenizer.json from any Qwen3.8-27B HF repo (a few MB).

# 3. Rebuild the truncated drafter byte-for-byte, then verify it:
python3 tools/truncate_drafter.py $MODELS/mtp-Qwen3.8-27B-Q4_0.gguf \
        data/draft_vocab_48k.json $MODELS/mtp-Qwen3.8-27B-Q4_0-d48k.gguf
python3 tools/validate_drafter.py $MODELS/mtp-Qwen3.8-27B-Q4_0.gguf \
        $MODELS/mtp-Qwen3.8-27B-Q4_0-d48k.gguf data/draft_vocab_48k.json

# 4. Build the bench payloads. Any large code+docs tree works as the corpus;
#    a llama.cpp checkout is a good choice.
QWEN_TOKENIZER_JSON=/path/tokenizer.json CORPUS_DIR=/path/llama.cpp \
  python3 tools/make_payloads.py tools/

# 5. Run the A/B: four depths, VRAM, acceptance, and a sha256 output check.
MODELS=$MODELS tools/run_validation.sh A mtp-Qwen3.8-27B-Q4_0.gguf
MODELS=$MODELS tools/run_validation.sh B mtp-Qwen3.8-27B-Q4_0-d48k.gguf
```

The published Q4_K_L run had arm B beat arm A by ~5–6% at mid and deep depth,
with matching output hashes. `tools/run_validation.sh` now defaults to the
promoted llama.cpp UD-Q4_K_XL target; set
`MAIN_MODEL=Qwen3.8-27B-Q4_K_L.gguf` to reproduce the historical table exactly.
On another target or corpus, hash equality remains required, but re-measure the
A/B delta instead of assuming it transfers.

Models:
[Unsloth UD-Q4_K_XL main](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
(17,559,178,144 bytes; embeds an unused MTP block), plus
[ggml-org's](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF)
`mtp-*-Q4_0` drafter and Q8_0 mmproj.

To serve, embed `config/llama-swap-qwen38.yaml` in a llama-swap config (the
binary ships in the image), or run the `llama-server` command from
`tools/run_validation.sh` directly. One coupling matters: draft-n 5 wins only
**with** the MMVQ cap env var. Without it, n4 is slower than n2 and nothing
fails loudly. The docs have the measurements.

## Stacked n-gram drafting (config-only, the biggest single win)

Agentic sessions spend much of their decode re-emitting files the model was
shown or just wrote. `--spec-type draft-mtp,ngram-mod` adds llama.cpp's
n-gram drafter on top of MTP: it keeps a host-RAM table of the session's
tokens and, when the last 32 tokens exactly match a stored chain
(`--spec-ngram-mod-n-match 32`; the default 24 fires on incidental quotes),
drafts up to 64 tokens of verbatim copy in one round. It preempts the MTP
drafter only on those rounds; everything else is unchanged. Same lossless
verify contract as MTP.

Measured (temp 0, v11): 8-turn cumulative file-editing session **108.8 →
260.8 tok/s** overall (warm turns 237–288); repeated queries +~50%; prose-7k
+7%; worst case −1.7% on a deliberately pathological self-repeating 54k
prompt. Zero VRAM. `tools/session_bench.py` is the workload — one-shot
benchmarks cannot see this technique, which is why early reports dismissed it.

The 200+ numbers are possible because the bandwidth ceiling applies per
verify pass, not per token: one ~30 ms pass can carry 64 draft tokens, and a
copy round drafts them for free.

## MMQ small-batch composite (wave 8, shipped as patch 0008)

The largest remaining lever is the quantized-weight GEMM (MMQ) on the
speculative verify path. The measurements behind this
(`docs/mmq-small-batch-analysis.md`):

- At batch ≥2, MMQ costs a fixed ~31 µs per call, almost independent of how
  many columns you need. ~20 µs of that is the x-tile global→shared load
  phase streaming at 618 GB/s where the direct-to-register path reaches 825.
  The load is fenced by `__syncthreads` against the compute phase, so the
  kernel waits instead of overlapping.
- A tile-and-grid reconfiguration (`experiments/armG-*.patch`) recovered part
  of this and measured +4.4% end-to-end: below the ship bar, preserved for
  reference.
- An independent SGLang profile of the same model confirms the target:
  quantized-weight GEMM is ~77% of a single-card decode step, while the
  Gated-DeltaNet path is ~2% and already fused
  (`docs/sglang-claim-check.md` — the same document checks and rejects the
  "SGLang goes beyond 100 tok/s" claim for this GPU: the numbers are real but
  Blackwell/NVFP4-only, and SGLang cannot currently load 4-bit weights for
  this architecture on Ampere).

Resolution 2026-08-17 (full story in `docs/journal-2026-08-17.md`): the
pipelining premise was **falsified by an ncu stall breakdown** — the stock
kernel sits at 56% of DRAM peak because of SM residency, not load scheduling,
and `cp.async` is structurally impossible for the x tile (nibbles unpack on
the way into shared memory). Full register pipelining recovered only ~3 µs of
a ~31 µs step: NO-GO. What survived and **shipped as `patches/0008`** is a
composite of the arm-G grid change, a type-agnostic y-tile double buffer, and
an m=1024 shape gate (one env var, `GGML_CUDA_MMQ_SMALLN`): verify at n=5
drops 38.23 → 35.24 ms at mid-1k and 41.73 → 38.63 ms at 54k — **+7% mid and
+5.4–6.4% deep decode at constant acceptance**, KLD 0.00617, correctness
1267/1267 at every level. Blast radius is structurally clean: only a backend
that both caps MMVQ and drafts ≤7 reaches the changed path. Two reviewer
notes: env-off in this binary is *not* stock (compile-time table change), and
KLD is nonzero because both changes alter float accumulation order.

## Measured dead ends

These were measured so you do not have to repeat them:

- Embedded-MTP mode (dropping `--model-draft`) routes draft logits through
  the target's Q8_0 head. Measured: −1.6% speed for −0.9 GiB VRAM. Keep the
  separate drafter.
- One draft step costs 2.25 ms marginal (measured at n=3→5), against a
  ~1.2 ms byte floor. Fixed overhead is ~45% of a draft step, which caps what
  head-shrinking can buy.
- q8_0 KV does not fit at 131k. DeltaNet leaves only 16 attention layers, so
  q4_0 KV costs ~18 KB/token. f16 KV forces a worse kernel path.
- Chain speculation (PR #27173) loses 6–12% on a single GPU.
- GDN chunked prefill (PR #26001) is lossy (KLD 0.0075).
- `docs/PERFORMANCE.md` has the full kernel-level analysis, the dead-end
  list, and the per-wave benches.

## What is reproducible from this repo alone

These parts are fully standalone and need no access to the original host:

- The image. The `Dockerfile` clones llama.cpp at the pinned ref from GitHub
  and applies `patches/`. Nothing is local.
- The truncated drafter. `data/draft_vocab_48k.json` is the exact keep-set
  that produced the shipped GGUF, and the surgery tools use only the Python
  standard library. The 32k/40k variants and the ranked frequency table
  (`data/token_freq.tsv.gz`) let you slice other set sizes without redoing
  the corpus pass. `data/coverage.json` and `data/ood_coverage.json` are the
  acceptance-risk evidence behind the 48k choice.

These parts work anywhere but default to the original machine. Override the
environment variables or edit the paths before reuse:

- `tools/build_draft_vocab.py` re-derives the keep-set from a corpus. Corpus
  roots are local paths by nature: point them at your own traffic-shaped
  text, and set the tokenizer with `QWEN_TOKENIZER_JSON`. You only need this
  script to build a different vocabulary.
- `tools/run_validation.sh` takes the models dir from `MODELS=`.
  `tools/make_payloads.py` takes `QWEN_TOKENIZER_JSON` and `CORPUS_DIR`.
  Avoid chat transcripts as corpus: at temp 0 they can make the model emit
  EOS at position 0.
- `config/llama-swap-qwen38.yaml` assumes llama-swap and this image. It is kept
  byte-identical to the `qwen3.8-27b` block in the cluster's retained
  llama-swap ConfigMap, which is the rollback path off vLLM — not what serves
  today.

## Layout

```
Dockerfile                  build llama.cpp @4df29be4f + patches (CUDA sm_86) + llama-swap
patches/0001..0008          historical patch stack (apply with git apply, in order); superseded by patches-v14/
patches-v9-v12-base-4df29be4/  verbatim copy of patches/, named for the base commit and the image
                            range it shipped in (v9-v12), kept alongside patches/ for clarity
patches-v14/                the CURRENT production patch set (six patches, base 0f3a71be1,
                            promoted 2026-09-02); see patches-v14/REBASE-2026-09-02.md
tools/                      draft-vocab pipeline, GGUF surgery + validation, bench harnesses
                            (one-shot A/B, cumulative session, 12-turn episode metric)
tools/bench/                shared GPU lock, quiesce, health, and result-posting helpers
data/                       keep-sets (32k/40k/48k), ranked token frequencies, coverage evidence
config/                     llama-swap model block (flags + env couplings, commented)
                            mirrors the cluster's retained llama-swap rollback block
config/llama-swap-qwen38-obliterated.yaml  standalone OBLITERATED Q4_K_M backend fragment
experiments/                measured but unshipped patches (see experiments/README.md)
docs/PERFORMANCE.md         full campaign write-up (waves, kernels, rejects, methodology)
docs/mmq-small-batch-analysis.md       MMQ verify-path analysis behind the wave-8 work
docs/journal-2026-08-17.md  day journal: what shipped, what was rejected, and why
docs/sglang-claim-check.md  the "SGLang >100 tok/s" claim, checked against this GPU
docs/truncated-draft-vocab-design.md   design doc for patch 0007 (data flow, correctness proof)
docs/thermals-and-oc.md     fan-curve dead zone, NVML offset convention, OC ladder, thermal ceiling
docs/dflash2-findings.md    DFlash2 (llama.cpp PR #27342) vs our MTP drafter: env-cap effect, caveats
docs/quant-selection.md     main-quant sweep: why UD-Q4_K_XL, the rejected 3.7bpw extreme, cost model
docs/obliterated-variant.md OBLITERATED Q4_K_M MTP findings, controls, and open leads
docs/vllm-companion.md      the vLLM profile that served 2026-08-20 to 2026-09-02, Club 3090 bundle, arms, quality gates
vllm/                       reproducible local overlay + Club 3090 local-layer bundle
vllm/image-v9/              the v8->v9 overlay actually deployed (2 vLLM patches + lineage)
vllm/image-v10/             exported k8s-deploy-v10 branch diff + last-deployed k8s manifest
vllm/image-v11-vllm028/     exported k8s-deploy-v11-vllm028 branch diff (vLLM 0.28 candidate)
benchmarks/engine-trial-2026-09-02/  llama.cpp master vs vLLM v10 vs SGLang trial that
                            promoted llama.cpp v14 to production; see its README.md
```

## License and credits

The `patches/` modify [llama.cpp](https://github.com/ggml-org/llama.cpp)
(MIT) and follow its license. Everything else here is MIT.

Prior art and sources this work builds on:

- [syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090) —
  the truncated-draft-vocabulary idea (their `build_draft_vocab.py`, for vLLM);
  patch 0007 and `tools/` are the llama.cpp counterpart.
- [EAGLE](https://github.com/SafeAILab/EAGLE) (Li et al.) — the `d2t`
  draft-to-target token-map mechanism, which patch 0007 reuses via llama.cpp's
  in-tree EAGLE3 implementation (`src/models/eagle3.cpp`).
- [llama-swap](https://github.com/mostlygeek/llama-swap) — the model-swapping
  proxy the `config/` block targets; its per-model `env:` lists are what make
  the env-gated kernel caps (patches 0004/0008) deployable per-backend.
- Model files: [Unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
  (promoted llama.cpp main GGUF),
  [bartowski](https://huggingface.co/bartowski) (the prior Q4_K_L and rejected
  IQ4_XS comparison),
  [ggml-org](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF) (MTP drafter
  and vision projector GGUFs), and [Qwen](https://huggingface.co/Qwen) for
  Qwen3.8-27B itself.
- The `ngram-mod` stacking defaults in `config/` were informed by a community
  ablation posted to r/LocalLLaMA (u/lukaLLM); tuned values are our own
  measurements.
- [Anbeeld/BeeLlama.cpp](https://github.com/Anbeeld/beellama.cpp) — a
  llama.cpp fork carrying its own low-bit KV-cache/quant work; evaluating a
  hybrid of our patch stack against it (see `docs/quant-selection.md`)
  informed how we think about weight-byte-vs-compute tradeoffs at very low
  bit widths.
- [KVarN](https://arxiv.org/abs/2606.03458) — the exact-tail KV-cache
  quantization approach BeeLlama implements; its "always retain an intrinsic
  exact-precision tail" design is the reference point our own quant-cost
  reasoning was checked against.
- [HoltYoung/vram-thermal-guard](https://github.com/HoltYoung/vram-thermal-guard)
  and ThomasBaruzier's `gputemps` approach — the thermal-watchdog and
  BAR-register GDDR6X-junction-reading approaches that made the fan-curve
  dead-zone finding in `docs/thermals-and-oc.md` possible to measure and fix.
- [club-3090](https://github.com/noonghunna/club-3090) — a cross-rig
  benchmark harness for this hardware class; its pinned bench prompts and
  report format shaped how we validated numbers in `docs/dflash2-findings.md`
  and `docs/quant-selection.md` against other rigs.

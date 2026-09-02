# SGLang trial: Qwen3.8-27B on one RTX 3090 (sm86, 24 GiB)

Prepared 2026-09-02, entirely off-GPU. The card was busy the whole time
(`VLLM::EngineCore`, 23764 MiB), so **nothing here has been executed against a
GPU**. Everything below is either read out of the SGLang source at the exact
commit the image ships, measured from the checkpoint's safetensors headers, or
quoted from an upstream issue. Where I am predicting rather than reporting, I
say so.

Everything lives in this directory:

| file | what it is |
| --- | --- |
| `run-sglang.sh` | launcher, three modes: `stock`, `hybrid`, `nospec` |
| `smoke.sh` | waits for `/health`, one chat completion, one tool call |
| `Dockerfile.sglang-trial` | the PR #36783 overlay |
| `build-hybrid.sh` | builds that image (done: `sglang-hybrid-mtp-ngram:pr36783`) |
| `vram-budget.py` | reimplements SGLang's pool sizing to predict boots/failures |
| `fork/`, `pr/` | blobless clone of the fork + a worktree at the PR head |
| `issue36048.json`, `iss358*.json`, `cookbook.txt` | the fetched primary sources |

## 1. Image choice

**`lmsysorg/sglang:nightly-dev-cu13-20260828-daf63171`** — pulled, 33.4 GB on
disk. Verified inside it:

```
sglang      0.0.0.dev1+gdaf631719      (tree == commit daf631719)
torch       2.13.0+cu130
flashinfer  0.6.17
python      3.12.3
g++         13.3.0        nvcc present
install     /sgl-workspace/sglang/python/sglang   (editable, not site-packages)
```

`torch==2.13.0` and `flashinfer_python[cu13]==0.6.17` are exactly what the PR
base's `python/pyproject.toml` pins, so the overlay does not fight the image's
wheels.

I did **not** use `v0.5.18`: no `v0.5.18*` tag appears in the 300 most recently
updated tags on Docker Hub (release-style tags there stop at `v0.5.13`), and it
would be the wrong base anyway — see below. I also did not use the cookbook's
`dev-qwen38-27b-dflash2`, which is pinned to `1cf2b8c5`, **460 commits behind**
the PR base. It is a reasonable alternative for `stock` mode only.

## 2. PR #36783 overlay: feasible, and built

```
PR          sgl-project/sglang#36783  "[Speculative Decoding] Add hybrid MTP and N-gram retrieval"
state       OPEN, DRAFT, 1 commit, 25 files, +4488 -68
head        63ad1158953dd92a665eda71e21a45f072d2bfd5   (shl518/sglang, feature/hybrid-mtp-ngram)
merge-base  7088f21922edc633dbb38e7543ef2b33252dfcbf   2026-08-28 04:22 UTC
```

`git merge-base` confirms the base is exactly `7088f2192`, and the PR is a
single commit on top of it. The image tree (`daf631719`) is **base~7**.

**It does not touch sgl-kernel and ships no CUDA.** `git diff --name-only`
against `sgl-kernel/`, `*.cu`, `*.cuh` returns nothing. Its 6 C++ files are all
under `python/sglang/kernels/jit/csrc/ngram_corpus/`, which the runtime JIT/FFI
layer compiles on demand — that directory is already present in the image, and
g++ 13.3.0 is installed. So no kernel build is required and the fallback to
"stock MTP only" is **not** needed.

**The overlay is clean, not a rebase.** The 7 commits between the image tree and
the merge-base change 18 files, all in `multimodal_gen`/diffusion, AMD MoRI
dispatch, HiCache mmap, DP-attention metadata, and `docker/Dockerfile.cu134`.
Intersection with the PR's 25 files is **zero**. Replacing the image's
`python/sglang` tree with the PR head's therefore yields exactly merge-base + PR
for every path this model touches. `COPY` merges, so the generated
`python/sglang/_version.py` (absent from a git checkout) survives.

Build verified — `build-hybrid.sh` asserts at build time and passed:

```
sglang 0.0.0.dev1+gdaf631719 at /sgl-workspace/sglang/python/sglang/__init__.py
hybrid server args present: 9
tau vector: [0.0, 0.4, 0.55]
OVERLAY OK
ngram corpus import OK
```

Two caveats worth carrying: the PR is a **draft whose CI is red** on all three
runs (Base, Extra, AMD ROCm), and its own description says it was GPU-tested on
v0.5.16 and then ported to main with "current-main GPU CI and benchmark reruns
still required". There are zero review comments. Nobody has run this code on
current main, on any GPU.

## 3. Checkpoint choice, and a correction

`Qwen3.8-27B-INT4-RedHatAI` is the right choice, but **not for the reason we
assumed.** I checked all 400 quantized Linear layers by reading the safetensors
headers directly:

| layer shape (out x in) | count | `out % 64` |
| --- | --- | --- |
| 17408 x 5120 | 128 | 0 |
| 5120 x 6144 | 64 | 0 |
| 5120 x 17408 | 64 | 0 |
| 10240 x 5120 | 48 | 0 |
| 6144 x 5120 | 48 | 0 |
| 1024 x 5120 | 32 | 0 |
| 12288 x 5120 | 16 | 0 |

All 400 pass Marlin's `output_size_per_partition % GPTQ_MARLIN_MIN_THREAD_N`
check (`marlin_utils.py:186`, constant = 64), and group size 128 divides every
input dim. The 96 narrow GDN projections that *would* fail —
`linear_attn.in_proj_a` / `in_proj_b`, both `[48, 5120]`, and `48 % 64 = 48` —
are held in BF16 by the checkpoint's `ignore` list. Confirmed: they are BF16 in
the file.

**The correction:** `Qwen3.8-27B-W4A16-AutoRound` does *not* quantize all Linear
layers. Its `ignore` list also has 96 `linear_attn` entries covering the same
`in_proj_a/b`, and all 402 of its quantized layers pass the same shape check. So
the predicted Marlin shape failure is not a real differentiator. The actual
difference is that AutoRound puts `lm_head`, `embed_tokens` and `re:^mtp\..*`
into their own quantization groups (`group_1/2/3`) instead of leaving them
dense, i.e. **it quantizes the MTP head we intend to use as the draft model**.
RedHatAI keeps all three in BF16. That is the reason to prefer RedHatAI, and it
is a better reason.

Measured from the headers, not estimated:

```
model.safetensors        17.326 GiB   (1184 BF16 + 400 I32 packed + 400 I64 shape)
model_mtp.safetensors     0.791 GiB   (15 BF16 tensors, the MTP draft head)
total                    18.117 GiB   = 75.5% of a 24 GiB card
```

## 4. The binding constraint is memory, and it is severe

The architecture, from `config.json` and confirmed against the cookbook: 64
layers, `full_attention_interval` 4, so 48 GDN linear-attention layers and 16
GQA full-attention layers; GDN is 48 value heads x 16 QK heads at head_dim 128;
attention is GQA 24/4 at head_dim 256; `mamba_ssm_dtype: float32`.

That geometry gives (cookbook, *Mamba ratio calculator*):

- one GDN state slot = **153.9 MB** at fp32, **78.4 MB** at bf16
- KV = **32.8 KB/token** at fp8, 65.5 KB/token at bf16

With 18.117 GiB of weights, a 24 GiB card has under 4 GiB for everything else,
and a single GDN state slot costs as much as 4,700 KV tokens. `vram-budget.py`
reimplements `_calculate_mamba_ratio()` and `_resolve_memory_pool_config()` from
`mem_cache/kv_cache_configurator.py` at the PR base:

```
config                                              R   K reqs  state  inter  KV GiB   KV tok  verdict
A  stock MTP, default flags (no tuning)             5   0    0    154M   616M    1.22    39824  FAILS TO BOOT
B  stock MTP, --disable-radix-cache --max-run 1     1   1    1    308M  1231M    1.70    55647  caps context at 54k
C  = B at --mem-fraction-static 0.92                1   1    1    308M  1231M    2.18    71360  64k fits
D  = C + --mamba-ssm-dtype bfloat16                 1   1    1    157M   627M    2.88    94378  64k fits
E  = C + --enable-linear-replayssm-spec             1   1    1    308M     0M    3.33   108897  64k fits
F  no-spec baseline (no MTP weights at all)         1   1    1    308M     0M    4.12   134793  64k fits
G  hybrid, L=9, fp32                                1   1    1    308M  2770M    0.75    24439  caps context at 23k
H  hybrid, L=9, bfloat16                            1   1    1    157M  1411M    2.15    70476  64k fits
I  hybrid, L=9, fp32 + replayssm                    1   1    1    308M     0M    3.33   108897  64k fits
J  stock MTP, radix kept: no_buffer + SKIP_LOCK     3   2    0    462M   616M    2.61    85436  FAILS TO BOOT
```

Row A is the headline. With default flags the solver lands on
`max_mamba_cache_size=0` at `mamba_ratio=5` and the server dies with

```
RuntimeError: Hybrid (mamba/linear-attention) state cache is too small to serve
any requests. max_mamba_cache_size=K, mamba_ratio=R, resulting max_num_reqs=0.
```

That is **the same arithmetic that produced issue #36048's `max_mamba_cache_size=3,
mamba_ratio=5, max_num_reqs=0`** on a 32 GB card. We are worse off, not better.
Row J shows the obvious middle setting (`no_buffer`) still fails.

`--disable-radix-cache` is therefore **not optional** — it is the only setting
that takes the state-slot multiplier to 1 (`_calculate_mamba_ratio` returns 1
immediately when radix cache is off; otherwise base 3, +2 for the default
`extra_buffer` under overlap, +1 lazy, +1 `no_buffer`). The cost is real: no
prefix caching, so every turn of a multi-turn conversation re-prefills. On this
card that is the price of booting at all.

The task asked for `--mem-fraction-static 0.85`. At 0.85 the post-weight budget
is 2.28 GiB and row B-equivalent caps KV around 24k tokens, so **0.85 cannot
serve a 64k context**. The scripts default to **0.92** and `MFS` is an env var.

`extra_buffer_lazy` is worth knowing about but unusable in `hybrid` mode: it
asserts "Lazy extra buffer requires overlap schedule", and the PR's own hook
forces `--disable-overlap-schedule` unless `--speculative-hybrid-overlap`.

## 5. Flags, and why each one

Common to all modes:

| flag | reason |
| --- | --- |
| `--model-path /models/qwen38` | checkpoint mounted read-only |
| `--dtype bfloat16` | compressed-tensors permits it. AWQ does not — issue #36048 hit `torch.bfloat16 is not supported for quantization method awq`. Choosing compressed-tensors over AWQ avoids that whole fp16 branch |
| `--quantization compressed-tensors` | matches `quant_method`; routes to `CompressedTensorsWNA16`, i.e. the Marlin kernel path |
| `--kv-cache-dtype fp8_e5m2` | halves KV to 32.8 KB/token, which the budget above needs. No compute-capability gate exists in the source (help text says CUDA 11.8+), and issue #35822 ran fp8_e5m2 on sm86 A2 cards with an 81,272-token pool, so this is confirmed on our exact architecture |
| `--mamba-ssm-dtype float32` | matches the checkpoint. Issue #35150 reports GDN recurrent-state drift under speculative verify that fp32 "delays substantially" but does not eliminate; bf16 halves the state cost and doubles the exposure |
| `--linear-attn-backend triton` | already the default. FlashInfer GDN decode is auto-selected only under `is_sm100_supported()`, and its CuteDSL prefill kernel is SM100-only. Set explicitly so it cannot drift |
| `--disable-radix-cache --max-running-requests 1` | see section 4; without these it does not boot |
| `--disable-prefill-cuda-graph` | prefill graph capture hangs on this hybrid-GDN architecture (#36048 item 4, #35437) |
| `--chunked-prefill-size 1024` | same value the #36048 reporter needed for a stable path |
| `--reasoning-parser qwen3`, `--tool-call-parser qwen3_coder` | `Qwen3CoderDetector` is registered under exactly `qwen3_coder` in `function_call_parser.py:92` |
| `SGLANG_MAMBA_CONV_DTYPE`, `SGLANG_MAMBA_SSM_DTYPE` | #36048 item 6: with a mismatched conv-state dtype the first request dies at `gdn_backend.py:559` with `Index put requires the source and destination dtypes match`. Set to bfloat16/float32 to match a BF16 model with an fp32 SSM state |

MTP operating point, from the cookbook: `--speculative-algorithm EAGLE
--speculative-num-steps 3 --speculative-eagle-topk 1
--speculative-num-draft-tokens 4`.

Hybrid adds, per the PR's own `docs/.../speculative_decoding.mdx`:
`--speculative-num-draft-tokens 9` (L must exceed `num_steps+1` so retrieval has
slots), `--speculative-hybrid-retrieval`, `--speculative-hybrid-tau-per-pos
off,off,0.40,0.55`, and `--speculative-hybrid-index-prompt`. `hybrid` also
defaults `REPLAYSSM=1`, because at L=9 the verify-intermediate reservation is
2770 MB and row G leaves only 24k KV tokens.

Deliberately **not** set: `--speculative-hybrid-ragged`. Its validator asserts
`decode_backend == "dsv4"`, so on any other attention backend it fails at
startup. It is a DeepSeek-V4 feature and irrelevant here.

I validated the tau vector against the PR's real parser off-GPU:

```
ACCEPTED  steps=3 'off,off,0.40,0.55'  -> [0.0, 0.4, 0.55]
REJECTED  steps=3 'off,0.3,0.40,0.55'  -> column 1 must be disabled
REJECTED  steps=3 'off,off,0.40'       -> wrong arity for num_steps
REJECTED  steps=4 'off,off,0.40,0.55'  -> wrong arity for num_steps
```

Both mode's full flag sets pass argparse inside their images. Note that
`prepare_server_args()` runs argparse **only** — the semantic checks
(`check_hybrid_retrieval_server_args`) run later, at engine init, so a
successful off-GPU parse is weaker evidence than it looks. `run-sglang.sh`
re-checks the tau arity and the two mandatory-off columns itself, before docker
is invoked, because the engine-init version of that error costs an 18 GiB weight
load first.

## 6. What to watch in the log, in the order it will bite

1. `Hybrid (mamba/linear-attention) state cache is too small to serve any
   requests` — sizing. Predicted for default flags; the scripts avoid it.
2. `Loaded weights leave no GPU memory for the KV cache ... If using speculative
   decoding, draft weights are now counted` — lower `CTX`, or `REPLAYSSM=1`.
3. `max_running_requests is capped to N by the mamba state cache` — a warning,
   not fatal, but it means concurrency is 1 whatever you asked for.
4. `Capture target decode CUDA graph begin` followed by 0% GPU utilisation and
   no progress — the #36048 decode-graph hang. Re-run with `EAGER=1`.
5. `cuda_graph_config decode='tc_piecewise' is not yet implemented; falling back
   to 'full'` — then the `full` capture hangs too (#36048 item 3).
6. `Index put requires the source and destination dtypes match` at
   `gdn_backend.py` — conv-state dtype; the env vars above prevent it.
7. Both ranks at 100% utilisation with ~25 W and no output, `py-spy` showing
   `tree_speculative_sampling_target_only` — **this is the Ampere MTP hang, see
   below.** Watchdog kills the scheduler at 300 s.
8. `gptq_marlin_fp16` sitting in NVCC for minutes on first request — #36048 item
   7 saw `awq_marlin` JIT never complete on SM89. We are on compressed-tensors
   W4A16, a different scheme, but the Marlin JIT is shared. The first request
   may be very slow; the JIT cache is persisted to `./cache`.

## 7. Honest risk assessment

**I expect `stock` mode to hang, and I expect `hybrid` to hang in the same
place.** The evidence is specific rather than general:

Issue **#35822** is the one that matters most, and it was not in the original
brief's framing. Title: *"EAGLE speculative decoding (native Qwen3.5/3.8 MTP)
hangs in `tree_speculative_sampling_target_only` on Ampere"*. The reporter runs
**Qwen3.8-27B AWQ on sm_86** — our exact model family and our exact compute
capability. Startup succeeds, graphs capture, a short request returns 200, and
then the first request long enough to reach EAGLE verify pins the GPU at 100%
utilisation and ~25 W forever. Same config with speculation off runs fine at
~17.5 tok/s. Zero comments, still open.

That failure is in `sgl_kernel`'s tree sampling kernel, reached through
`eagle_sample` → `run_eagle_verify` → `verify`. **PR #36783 changes none of
that.** It splices a retrieval tail onto the draft chain and hands the result to
the same target-verify and the same sampling kernel. If MTP verify hangs on
Ampere, hybrid MTP+retrieval hangs on Ampere, and it arrives there with a
*longer* chain (L=9 vs 4), which is if anything more exposure. The PR is not a
fix for our problem; it is a throughput optimisation layered above it.

Issue **#36048** adds that on a *newer*, larger card (SM89, 32 GB) the only
stable configuration is fully eager at **5.9–6.0 tok/s**, against ~64–66 tok/s
for llama.cpp on the same GPU, and that native MTP could not allocate a request
slot at all. Its author asked five direct questions and got no reply.

The cookbook is the third signal, by omission. Its validated matrix is H200, RTX
PRO 6000, RTX 5090 and DGX Spark — **SM90 and SM120/121 only, no Ampere entry,
and no W4A16/INT4 checkpoint recipe at all** (BF16, FP8, NVFP4). Nothing about
this configuration is on a supported path.

So, ranked by what I actually expect:

1. **`nospec` boots and serves.** Highest confidence. Row F has 134k KV tokens
   of headroom, and #35822 explicitly reports the no-speculation arm working at
   ~17.5 tok/s on sm86. Run this first — it establishes whether the checkpoint,
   Marlin path, chat template and parsers are sound, independent of MTP.
2. **`stock` boots** (the sizing is solved) **and then hangs on the first
   substantial request**, per #35822. If it survives, expect single-digit to low
   tens of tok/s, not a speedup.
3. **`hybrid` boots and hangs identically**, plus it is unreviewed draft code
   with red CI that no one has run on current main on any GPU.

If the goal is tokens per second on this card, the honest read is that SGLang is
not the right engine for Qwen3.8-27B on Ampere today, and the 10x gap #36048
measured against llama.cpp on a *better* card is the number to weigh. If the
goal is to characterise the PR, the useful experiment is narrow: run `nospec`,
then `stock`, and see whether #35822 reproduces here. There is no point
evaluating hybrid retrieval's acceptance-length gain until MTP verify completes
a single long request on this GPU.

Cheap thing worth trying if `stock` hangs: `EAGER=1` (which is what #36048
needed) plus `--cuda-graph-max-bs-decode 2`, since #35822's hang was in the
verify sampling kernel rather than in capture and its reporter had graphs
captured successfully. Also `--sampling-backend pytorch`, which that reporter
used and which routes around the flashinfer sampling path.

## 8. Reproducing the research

```bash
# issue bodies + comments (all five referenced issues, and the PR)
curl -A "OpenAI File Downloader, XaiImageApiFetch/1.0" \
  https://api.github.com/repos/sgl-project/sglang/issues/36048

# merge-base
cd fork && git merge-base 63ad1158953dd92a665eda71e21a45f072d2bfd5 upstream/main
#   -> 7088f21922edc633dbb38e7543ef2b33252dfcbf

# the sizing prediction
python3 vram-budget.py
```

Sudo was never needed. Nothing in `/data/docker-services`, `k8s/`, or the model
directories was modified; the checkpoints are mounted read-only and nothing was
deleted. The two images added are
`lmsysorg/sglang:nightly-dev-cu13-20260828-daf63171` (33.4 GB) and the derived
`sglang-hybrid-mtp-ngram:pr36783` (33.5 GB, sharing all but one layer). Disk
went from 238 GB free to ~205 GB free.

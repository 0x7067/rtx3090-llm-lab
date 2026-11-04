# Rebase of the vendored llama.cpp patches onto master

- **Old base**: `4df29be4f4c3673f428170fda944a5b19f743bb8` (2026-08-16)
- **New base**: `0f3a71be1` — "mtmd: Fix Qwen3-tts-0.6b (#28231)" (2026-09-02)
- **Branch**: `trial-2026-09-02` in `./llama.cpp`
- **Trial image**: `llama:trial-2026-09-02` (`./Dockerfile.trial`)

All six exported patches apply to a pristine `0f3a71be1` checkout with plain
`git apply` (no `--3way`, no fuzz). Verified in a throwaway worktree before the
image build.

## Build result

`docker build -f Dockerfile.trial -t llama:trial-2026-09-02 .` succeeded on the
first attempt after the rebase edits below. Image `89f7b893cecb`, 3.57 GB,
compile step 317 s wall on 20 cores. The build log confirms the intended base:
`HEAD is now at 0f3a71be1 mtmd: Fix Qwen3-tts-0.6b (#28231)`.

Targets built: `ggml-cuda` (`-DCMAKE_CUDA_ARCHITECTURES=86`), `llama-server`,
`llama-bench`, `llama-perplexity`, `test-backend-ops`. Zero compiler errors and
zero warnings attributable to the patches.

Post-build checks in the image:

- all three env gates are present in `libggml-cuda.so`:
  `GGML_CUDA_MMVQ_NE11_MAX`, `GGML_CUDA_MMQ_SMALLN`, `GGML_CUDA_FATTN_MMA_Q`
- 0007's loader log string `QWEN35 using d2t mapping` is present in `libllama.so`
- `ldd` leaves only `libcuda.so.1` unresolved, which is the host NVIDIA driver
  injected by the container toolkit at runtime. Same as the production image;
  not a linking defect. It does mean the binaries cannot even print `--version`
  without a GPU-enabled runtime.

**Not imported into k3s.** No `k3s ctr images import` was run.

**What compiling does and does not prove.** It proves every patch's code is
syntactically valid and type-correct against the new base for sm86, and that all
template instances the patches touch actually instantiate. It proves nothing
about numerics or speed. No kernel in this image has been executed. See the
unverified note under 0005 in particular.

Numbering is preserved from `k8s/workloads/apps/llama/image/patches/`, so the
gaps at 0001 and 0006 are intentional. `git apply patches/*.patch` applies them
in glob order, which is the order they were rebased in.

| # | Status | Summary |
|---|--------|---------|
| 0001 | **dropped** | DFlash greedy fast path superseded by upstream backend sampling |
| 0002 | applied clean | Ampere MMQ small-batch tiles |
| 0003 | rebased | GQA-batched small-batch FA vector kernel |
| 0004 | applied clean | env-gated dense MMVQ batch cap |
| 0005 | rebased | inline-q4-dequant FA MMA path |
| 0006 | **dropped** | only existed to disable 0001 |
| 0007 | applied clean | qwen35 MTP truncated draft vocab via d2t |
| 0008 | rebased | env-gated small-batch MMQ grid + y-tile double buffer |
| PR26 | **skipped** | hybrid/recurrent checkpoint fix is upstream already |

Both env gates the production ConfigMap depends on are intact:
`GGML_CUDA_MMVQ_NE11_MAX` (`ggml/src/ggml-cuda/mmvq.cu:381`) and
`GGML_CUDA_MMQ_SMALLN` (`ggml/src/ggml-cuda/mmq.cuh:1459`).

---

## 0001 — DFlash draft greedy fast path + LLAMA_SPEC_PROF — DROPPED

Two things travelled in this patch and both lost their reason to exist.

**The greedy fast path is superseded by upstream #26958.** Master defaults
`common_params_speculative::draft.backend_sampling` to `true`
(`common/common.h:332`) and builds a backend top-k chain for non-DFlash2
drafters (`common/speculative.cpp:1031`). Patch 0006 existed precisely to make
the fast path stand down whenever backend sampling is on, so with the upstream
default the fast path is unreachable.

The production ConfigMap has already moved to the upstream path. The
`muse-glimmer-30b` backend in `k8s/workloads/apps/llama/configmap.yaml:200-217`
does **not** pass `--no-spec-draft-backend-sampling`, and its comment records
the reason: GPU draft sampling measured +3.2%/+2.7%/+3.0% (short/1k/7k prose)
with byte-identical output. So upstream's sampling is not merely equivalent, it
beat the vendored fast path on the exact workload the patch was written for.
Keeping 0001 would reintroduce dead code plus the acceptance-drops-to-zero
landmine that 0006 was papering over.

Note the README at `k8s/workloads/apps/llama/README.md:149-156` still describes
the older v9 arrangement where the flag was set. It is stale relative to the
ConfigMap; the ConfigMap is what runs.

**The `LLAMA_SPEC_PROF` instrumentation no longer has its measurement sites.**
Upstream fused the DFlash feature-gather, encoder pass and K/V injection into a
single decode. The separate `llama_encode(ctx_dft, enc_batch)` +
`llama_get_embeddings_nextn()` + `llama_decode(ctx_dft, batch_inject)` sequence
that the `encode`, `embd_nextn_copy` and `decode_inject` timers wrapped is gone
from master. Three of the patch's four conflict hunks in `common/speculative.cpp`
were timers around deleted code.

**Consequence to be aware of:** this build has no `LLAMA_SPEC_PROF` per-phase
profiling. Nothing in the production config reads it, but if the lab wants that
instrumentation back it needs re-authoring against the fused decode, not a
rebase. That was not attempted here.

## 0002 — Ampere MMQ small-batch tiles — APPLIED CLEAN

`ggml/src/ggml-cuda/mmq-config-ampere.cuh`. Four `CASE` rows for `J = 16` move
from 256 threads / I=128 to 128 threads / I=64: Q4_K at lines 158 and 163, Q5_K
at lines 175 and 180. No upstream drift around them.

## 0003 — GQA-batched small-batch FA vector kernel — REBASED

Applied clean to `fattn-common.cuh`, `fattn-vec.cuh` and `fattn.cu`. One
conflict, in `tests/test-backend-ops.cpp`.

**Still needed.** Master's `gqa_opt` machinery in `fattn.cu` is the pre-existing
*MMA* `ncols2` batching, not this patch's intent. Master's vector kernel is
still `template<int D, int ncols, ggml_type type_K, ggml_type type_V, bool
use_logit_softcap>` (`ggml/src/ggml-cuda/fattn-vec.cuh:19`) with no GQA-group
parameter, so nothing upstream serves a GQA group from one block in the vector
kernel.

**What changed.** The patch hoists master's inline `gqa_opt_applies` computation
out of `ggml_cuda_get_best_fattn_kernel` into a shared
`ggml_cuda_fattn_gqa_opt_applies(dst)` helper in `fattn-common.cuh`, because the
new `ggml_cuda_fattn_vec_use_gqa` needs the same predicate. Upstream had since
added a `gqa_ratio >= 2` term to that predicate. I checked the extracted helper
term by term against master's deleted inline version: `gqa_ratio >= 2 && mask &&
max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0` plus the 16-byte stride
loop over `{Q, K, V, mask}` skipping quantized tensors. Identical, so the
refactor is behaviour-preserving. The other `use_gqa_opt` site in
`ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2` deliberately omits the
`gqa_ratio >= 2` term and is untouched.

The new VEC route lands in the `cc < GGML_CUDA_CC_ADA_LOVELACE` arm of the
quantized-KV branch, which is where it was aimed; in the final tree that call
sits at `ggml/src/ggml-cuda/fattn.cu:471`, wrapped by 0005's `mma_q_claims`
guard at line 467, and the extracted helper is called at line 374. The SM-fill
parallelism floor that fixed the shallow-depth regression survived:
`ntiles_dst*ntiles_KV >= nsm` in `ggml_cuda_fattn_vec_use_gqa`
(`ggml/src/ggml-cuda/fattn-vec.cuh`).

**Conflict resolution.** Both sides append cases to the same spot in
`test_flash_attn_ext`'s list, over an empty merge base. Kept both blocks;
upstream's large-KV F16 and MLA-view cases first, then the patch's quantized-KV
GQA sweep. Verified the patch's 13-argument calls still match master's
constructor, which has defaults from `permute` onward
(`tests/test-backend-ops.cpp`, `test_flash_attn_ext` ctor).

## 0004 — env-gated dense MMVQ batch cap — APPLIED CLEAN

`ggml/src/ggml-cuda/mmvq.cu:372-389`. `GGML_CUDA_MMVQ_NE11_MAX` read at line
381. This is the gate the `muse-glimmer-30b` and `qwen3.8` backends set to 3.

## 0005 — inline-q4-dequant FA MMA path — REBASED (behavioural fix required)

Applied clean to `fattn.cu` and `tests/test-backend-ops.cpp`. Five conflicts in
`fattn-mma-f16.cuh`, all from one upstream change, plus one real correctness
problem that a textual merge would have hidden.

**Still needed.** Master has no `GGML_CUDA_FATTN_MMA_Q` and no inline dequant in
the MMA kernel; nothing upstream implements this.

**Upstream drift: shared-memory swizzling.** Master added
`ggml/src/ggml-cuda/fattn-swizzle.cuh` and threaded a swizzle flag through the
tile loads: `swz_K = ggml_cuda_fattn_smem_swizzle::enabled(nbatch_K2)` and
`swz_V` likewise (`fattn-mma-f16.cuh:674-675`). Every
`flash_attn_ext_f16_load_tile` call gained a `swz` template argument, and the
`ldmatrix` reads in the MMA loop now go through
`ggml_cuda_fattn_smem_swizzle::load_ldmatrix<stride_tile_K, swz_K>`. The tile
stride changed too: `tile_stride(n)` returns `n` when swizzled and `n + 4`
otherwise. All five conflicts are the same shape, the patch wrapping a call the
upstream signature had grown a parameter on.

**The correctness problem.** `enabled(n)` is `n >= 32 && n % 32 == 0` on
Turing and newer, and multi-stage configs pin `nbatch_K2 == DKQ/2`, so at the
head sizes this patch targets (DKQ 128, DV 128) `nbatch_K2 == nbatch_V2 == 64`
and swizzling is **on** for exactly the instances the patch claims. The patch's
`flash_attn_ext_f16_dequant_tile_q` wrote the expanded F16 tile in linear
layout, at `(half *)(tile_KV + i*stride_tile) + kb*QK4_0`. Merged as written it
would compile and produce silently wrong attention output, because the
subsequent swizzled `ldmatrix` reads would look elsewhere.

**Fix applied.** Made the dequant swizzle-aware rather than forcing swizzling
off for these instances, so upstream's optimization is kept and the address math
is upstream's own rather than something I invented.
`flash_attn_ext_f16_dequant_tile_q` gained a `bool swz` template parameter
(`fattn-mma-f16.cuh:511-512`) and now addresses each `half2` through
`ggml_cuda_fattn_smem_swizzle::bytes_rc<stride_tile>(i, col_h2)` when `swz`,
which is the same map `flash_attn_ext_f16_load_tile` uses for its 16-byte
`cp.async` chunks. The two call sites pass `swz_K` (line 712) and `swz_V`
(line 1084).

Upstream only ever calls `bytes_rc` on chunk-aligned columns, so the extension
to a single `half2` needs an argument. `bytes_rc(row, c) = ((row*S + c)*4) ^
((row & 7) << 4)`. The XOR occupies bits 4-6. A bank-aligned `S` is a multiple
of 32, so `row*S*4` contributes only bits 7 and up. Writing `c = c0 + r` with
`c0 = c & ~3` and `r = c & 3`, the term `c0*4` is a multiple of 16 and `4r`
occupies only bits 2-3. The three parts are disjoint, therefore
`bytes_rc(row, c) == bytes_rc(row, c & ~3) + (c & 3)*sizeof(half2)` and indexing
by absolute column is exact. That is why the swizzled branch can be a plain
address computation with no chunk bookkeeping.

**Unverified.** This is a numerics change inside a CUDA kernel and it compiles,
but no GPU was available in this task, so neither `test-backend-ops` nor a
perplexity gate was run. The swizzled write path of 0005 has never executed.
Treat `GGML_CUDA_FATTN_MMA_Q` as unvalidated on this base until
`test-backend-ops -o FLASH_ATTN_EXT` passes on the 3090.

Two conflicts resolved as `if constexpr (mma_q) { NO_DEVICE_CODE; } else { ... }`
are in the `nstages <= 1` arms. Those are unreachable for `mma_q`, which asserts
`nstages > 1` (`fattn-mma-f16.cuh:2206`), and they were unreachable in the
original patch too.

## 0006 — greedy fast path requires host logits — DROPPED

A one-line guard adding `&& !params.backend_sampling` to 0001's
`greedy_fast_path`. With 0001 dropped there is no fast path to guard and the
patch has no target. Upstream's backend sampling, which this guard existed to
defer to, is now the only path.

## 0007 — qwen35 MTP truncated draft vocab via d2t — APPLIED CLEAN

`src/models/qwen35.cpp`. Loader reads the draft vocab size from the `d2t` tensor
and sizes `output` accordingly (lines 45-55); `graph_mtp` scatters reduced
logits back into full target-vocab space. `LLM_TENSOR_D2T` already exists on
master for EAGLE3, so the enum the patch reuses is present. The normal decoder
path keeps its assertion that `d2t` is absent (lines 229-230).

## 0008 — env-gated small-batch MMQ grid + y-tile double buffer — REBASED

Applied clean to `mmq-config-ampere.cuh` and `mmq.cuh` in sequence, with no
conflicts and no hand edits.

Worth recording because it is misleading in isolation: `git apply --check`
reports 0008 as conflicting against bare master. It is not stale. 0008 was
authored on top of 0002 and edits the same `CASE` rows in
`mmq-config-ampere.cuh`, so it only applies once 0002 is in the tree. Applied in
glob order it is clean.

`GGML_CUDA_MMQ_SMALLN` is intact: the level is read at
`ggml/src/ggml-cuda/mmq.cuh:1459` and gated to NVIDIA Ampere with `J <= 8`
(lines 1452-1458), which still matches the RTX 3090. The three levels
(`_GRID` 1, `_YBUF` 2, `_GATE` 3) and the `_MIN_ITERS` grid walk-back survive.

The double buffer merged onto upstream's current loop without changing the
default path: the `pipeline < GGML_CUDA_MMQ_SMALLN_YBUF` branch
(`mmq.cuh:931-965`) is upstream's loop verbatim, four `__syncthreads` and all,
and the two-`__syncthreads` variant is confined to the `else`
(`mmq.cuh:966-990`). So a level-0 or non-Ampere build runs upstream's code.

Note the caveat the patch header already carries: the small-J rows of the config
table are changed unconditionally by 0002, so `GGML_CUDA_MMQ_SMALLN=0` in this
build is not the upstream kernel.

## buun PR26 — hybrid/recurrent context checkpoint restore — SKIPPED

Master already implements both halves. Not ported.

**Half 1, the restore predicate.** PR26 wanted `cur.pos_max <= pos_next` for
recurrent and hybrid models instead of the SWA-oriented `cur.pos_min <
pos_min_thold`. Master's predicate is now:

```cpp
// workaround for [TAG_CHECKPOINTS_FIX_POS_MIN]
if (cur.pos_max > pos_next) {
    return false;
}
return cur.pos_min < pos_min_thold || cur.pos_min == 0;
```

at `tools/server/server-context.cpp:3330-3334`. The `pos_max > pos_next`
rejection is PR26's condition, landed upstream as
`db94854ff` "server : skip checkpoints beyond pos_next (#24411)" (2026-06-11).
The `cur.pos_min == 0` escape came from `e8f508269` "server : fix restore for
checkpoints with pos_min == 0 (#21510)" (2026-04-07).

For a recurrent memory, `seq_pos_min` is the min `pos` over the cells holding
the sequence (`src/llama-memory-recurrent.cpp:378`), and a recurrent sequence
occupies one cell, so `pos_min == pos_max`. Master's predicate therefore reduces
to `p < pos_next` where PR26 asked for `p <= pos_next`. The only divergence is a
checkpoint sitting exactly at `pos_next`, which would leave zero tokens to
process and violate the `[TAG_PROMPT_LOGITS]` invariant that at least one token
be evaluated. Master's strict inequality is the more correct of the two.

Upstream also grew dedicated plumbing for these models that did not exist at the
old base: `COMMON_CONTEXT_SEQ_RM_TYPE_RS` ("can seq_rm partial sequences,
bounded by `n_rs_seq`", `common/common.h:991`), which the checkpoint-creation
gate now tests (`server-context.cpp:3443-3446`), and a reset log line that names
hybrid/recurrent memory explicitly.

**Half 2, the 64-token minimum.** PR26 lowered a `slot.prompt.n_tokens() >= 64`
checkpoint-creation floor to 4 for these models. That constant no longer exists
anywhere in `tools/server`. The creation logic was rewritten: checkpoints are
now placed at `checkpoint_offsets[] = {4 + n_ubatch, 4}` tokens before the end
of the prompt (`server-context.cpp:3538`, ref PR #20288), at user-message starts
(#24176), and pruned by `checkpoint_min_step` (#25472). The only remaining
floor is `pos_min < 0`, i.e. nothing cached yet
(`server-context.cpp:3590-3592`). Upstream reaches 4 tokens from the prompt end
for any prompt length, which is what PR26 was asking for.

**Residual gap, stated for honesty.** One case is not equivalent. For a model
that is hybrid *and* uses SWA, `llama_memory_hybrid::seq_pos_min` returns the max
of the two caches' minima (`src/llama-memory-hybrid.cpp`), so the recurrent cell
dominates and `pos_min` tracks the last position. `pos_min_thold` is then
`pos_next - n_swa`, so a checkpoint newer than `n_swa` tokens back fails
`cur.pos_min < pos_min_thold` and falls through to a full reprocess unless
`pos_min == 0`. PR26's unconditional `pos_max <= pos_next` would accept it.

Neither production target model reaches that case, and the two miss it for
different reasons, so this is worth stating rather than waving at:

- **qwen3.8** loads as `LLM_ARCH_QWEN35`, which *is* in `llm_arch_is_hybrid`
  (`src/llama-arch.cpp`). But `src/models/qwen35.cpp` never touches
  `hparams.n_swa` or `hparams.swa_type`, so `n_swa` keeps its default of 0
  (`src/llama-hparams.h:162`). With `n_swa == 0` the threshold collapses to
  `pos_next` and master's predicate is correct for it.
- **muse-glimmer** does set `n_swa` from GGUF and `swa_type =
  LLAMA_SWA_TYPE_STANDARD` (`src/models/muse-glimmer.cpp:5-12`), but
  `LLM_ARCH_MUSE_GLIMMER` is in neither `llm_arch_is_hybrid` nor
  `llm_arch_is_recurrent`, so PR26's branch would never have fired for it.

`n_swa` reaches the server through `llama_model_n_swa(model_tgt)`
(`server-context.cpp:1204`). If a model that is both hybrid and SWA is ever
adopted here, revisit this. Porting PR26 verbatim would still be the wrong
shape, since master's `pos_max > pos_next` rejection is already in place; the fix
would be to relax the `pos_min` term for recurrent-backed memory only.

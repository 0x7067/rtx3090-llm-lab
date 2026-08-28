# vLLM PR #51812 backport onto our 0.27.1 tree

New files only, in `patches-night/`: `51812-upstream.diff`, `51812-backport.patch`, `pr51812.json`.

## 1. Applies cleanly

Yes. Verified the way the image build invokes `patch` (`-p1 -d $SP`), against copies in `scratch/`,
never the venv (venv file md5 unchanged, `0d6c83b9…`): `-F0 --dry-run` clean with zero offset and
zero fuzz; real apply byte-identical to applying the upstream diff (`-p2`) to the same source;
`py_compile` passes; `patch -p1 -R --dry-run` succeeds, so `verify.sh` sees it as applied.

Two path corrections. The brief's `df2-repo/venv/...` does not exist — the tree is at
`/data/buttercup_6tb/k3s/vllm-trial/venv/lib/python3.12/site-packages/vllm`, a **pristine** 0.27.1
install with no repo patch applied (`qwen3_dflash2.py` absent). And the upstream diff needed its
`vllm/` path component stripped: the build runs `-d $SP` with `$SP` already inside the package.

## 2. The exact hunk

Target `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`, same path as upstream. No
local patch touches it — `hybrid-kv-groups-v2-cudagraph.patch` and
`vllm-pr50021-gdn-spec-bounds.patch` match "gdn" but hit `v1/core/`, `v1/worker/gpu/`,
`mamba/ops/` and `third_party/flash_linear_attention/` — so glob order is irrelevant here.

```diff
--- a/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
+++ b/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
@@ -1249,9 +1249,13 @@
         if spec_sequence_masks is not None:
             if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                 mixed_qkv_spec = mixed_qkv
+                a_spec = a
+                b_spec = b
                 mixed_qkv_non_spec = None
             else:
                 mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
+                a_spec = a.index_select(0, spec_token_indx)
+                b_spec = b.index_select(0, spec_token_indx)
                 mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
         else:
             mixed_qkv_spec = None
@@ -1376,8 +1380,8 @@
             core_attn_out_spec, last_recurrent_state = (
                 fused_sigmoid_gating_delta_rule_update(
                     A_log=self.A_log,
-                    a=a,
-                    b=b,
+                    a=a_spec,
+                    b=b_spec,
                     dt_bias=self.dt_bias,
                     q=query_spec,
                     k=key_spec,
```

`a_spec`/`b_spec` are unbound in the `spec_sequence_masks is None` branch, but their only use site
sits inside `if spec_sequence_masks is not None:`, so there is no `NameError` path. As upstream,
`_forward_core_rocm` and the kimi/olmo GDN variants are left alone.

## 3. Reachability for our config: yes, on SPEC=mtp

- `SPEC=mtp` is the default (`start_qwen.sh:70`) and runs the **V1** runner.
  `VLLM_USE_V2_MODEL_RUNNER` is unset anywhere in the repo, so `config/vllm.py:578` falls
  through (method is `mtp`, not `dspark`/`dflash`) to `_is_default_v2_model_runner_model()`,
  which returns `False`: `Qwen3_5ForCausalLM` declares `IsHybrid` (`models/qwen3_5.py:288-291`)
  and is not in `DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES` (`config/vllm.py:69-79`). Only
  `SPEC=dflash2` reaches V2. `models/qwen3_5.py:143` builds the GDN layer from that module.
- Mixed batches are routine, not a corner case: chunked prefill with `--max-num-batched-tokens
  2048` (`start_qwen.sh:380`) and `max_num_seqs 8` lets a prefill chunk share a step with decoding
  requests, and a non-speculative decode row does it too. vLLM treats the layout as expected
  (`v1/worker/gpu/warmup.py:354`, `.../model_states/mamba_hybrid.py:249`). Every such step takes
  the `index_select` branch: `num_decodes > 0 and num_spec_decodes > 0` reclassifies non-spec
  decodes as prefills (`gdn_attn.py:247-251`), so `num_prefills != 0` and the fast path is skipped.
- The gates are then wrong **whenever a non-spec token occupies a lower batch position than a
  spec token**. If the spec request sits at positions 0..k the gather is the identity and that
  step is fine — the same coincidence that spares V2. Which holds depends on request slot
  ordering, so corruption is intermittent, not every mixed step. Upstream saw it on V1 with our
  shape: `spec_token_indx = (1,2,3)`, `non_spec_token_indx = (0,)`.

## 4. Expected user-visible effect

Upstream measured, on **Qwen3.5-2B, k=2, max_model_len=128, eager** — not our config, so the
mechanism transfers and the magnitudes do not: mean absolute chosen-logprob error 0.002755 →
0.000208, max 0.020539 → 0.001690. The residual after the fix is kernel nondeterminism, not
leftover bug. Greedy token IDs were identical there because the drift did
not cross an argmax boundary — a property of that run, not a guarantee. Our traffic samples at
temperature 1.0, so it **is** affected: corrupted gates perturb the speculative rows' logprobs,
shifting sampled tokens and MTP acceptance. Expect a small quality and acceptance-rate change,
not a crash — silent numeric corruption a greedy eval will not show.

## 5. Folding it into the image build

1. Copy to `df2-repo/patches/vllm-pr51812-gdn-spec-gates.patch`, matching
   `vllm-pr50021-gdn-spec-bounds.patch`. The patch header already cites that filename.
2. Nothing else to register: `Dockerfile:29` globs `patches/*.patch` and runs `patch -p1 -d
   "$SP"` over each, `verify.sh:43-49` iterates the same glob, `_check_applied.py` has no registry.
3. Cosmetic caveat: `_check_applied.py` needs 3 added lines over 24 characters and this patch has
   2, so its content fallback returns 1 — never reached, since the primary reverse dry-run succeeds
   (no other patch touches this file). Both verified.
4. Rebuild, run `verify.sh` (expect `vllm-pr51812-gdn-spec-gates.patch applied`), then measure a
   sampled (temp 1.0) eval and the MTP acceptance rate before/after.

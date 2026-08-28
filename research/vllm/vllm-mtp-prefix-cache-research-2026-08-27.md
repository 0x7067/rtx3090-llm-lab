# vLLM MTP + prefix-cache research — 2026-08-27

## Headline corrections

1. **#51113 is already in our 0.27.1.** Merged 2026-08-06; v0.27.0 shipped 08-10, v0.27.1 on 08-11. I diffed the patch against the installed tree — `v1/core/sched/scheduler.py:383-420` already carries `prefill_end`, the new invariant comment, and the unconditional `next_block_boundary`. Our 08-26 comment on #47194 ("after #51113 shipped in 0.28.0…") rests on a version delta that does not exist. **Upgrading to 0.28.0 buys nothing for this bug.** That comment is being cited as one of "2 independent hardware confirmations" — it needs a correction.
2. **The clean A/B did not test the production code path.** `MambaManager.find_longest_cache_hit` has two branches (`single_type_kv_cache_manager.py:1309-1356`): a fine-grained per-hash-unit loop when `alignment_tokens < block_size`, else a coarse `max_num_blocks` loop. **Production does not set `--prefix-match-unit`:** the v9 image entrypoint is `docker/entrypoint.sh single` → `df2-repo/single-user/start_qwen.sh`, which adds `--prefix-match-unit 128` only under `CTX="huge"` (`:271`), and `deployment.yaml:27` pins `CTX=long`. So `hash_block_size == block_size` and production takes the coarse branch. The posted A/B comment describes arm C as `--prefix-match-unit 16` (fine-grained) — a different finder. Either the comment's config line is wrong or that arm ran a script other than `run_qwen_v028.sh`, which sets no such flag; either way the clean result does not clear production.
3. **The bug is open on 0.28.0** — #53912 (08-26, H100, Qwen3.5 27B FP8, MTP k=2, align) reproduces there and states that #51113 fixed the *write* path while the *read* path is untouched.

## 1. #47194 — open, updated 2026-08-26

**Citation:** github.com/vllm-project/vllm/issues/47194. No fix merged after 0.28.0. Closed-unmerged attempt: #47861. Siblings: #47087 (closed), #43559 (closed 08-06 by #51113), superseded by #53912.

**Measured** (2×RTX 2080 Ti SM75, TP2, GPTQ-marlin, fp8_e5m2 KV, sha256_cbor, `qwen3_coder` + `qwen3` parsers, 58K static prefix): MTP off → 90-98% hit rate, all suites pass. MTP3 on, everything else identical → 83.9% hit rate, cold PP 23.35s → hot 2.03s, but tool call 2/10, multicase 1/18, needle recall 0/10, multi-turn tool 0/5, `<tool_call>` leaking as plain text.

**Triage:** no maintainer mechanism in-thread. The load-bearing comment is @dmih (2×3090, TP2, Qwen3.6 INT4, `qwen3_next_mtp` n=3, fp8 KV): *"cache of the common prefix seems to get poisoned… while the cache is hot, close to zero success running tools… when this cache portion goes away or after a vLLM restart everything is back to normal."* That is **producer-side contamination**, not a decode-time fault.

**Mechanism, named precisely by #53912:** `MambaManager.find_longest_cache_hit` accepts `drop_eagle_block` and never acts on it. Confirmed in our tree — declared at `single_type_kv_cache_manager.py:1287`, absent from the body; `FullAttentionManager` honours it at `:768`. `kv_cache_coordinator.py:755-756` compounds it by skipping the eagle margin for `MambaSpec` ("its finder never drops"). The last matched mamba block — whose recurrent state can hold draft positions that verification later rejected — stays reachable and is served to every later request on that prefix. Fix candidates: **#43650** (open, 6 lines, `max_num_blocks -= 1` when `use_eagle`) and **#48375** ("Honor drop_eagle_block in MambaManager"). #43650 has two field confirmations (18/39 → 39/39 on a private benchmark; `content: null` at ~1-in-8 fixed outright) and starts clean on 0.28.0 with no accuracy regression.

**Symptom match:** strong on software — align mode, both parsers, MTP k=3, fp8 KV, prefix caching, hybrid GDN. Deltas: reporter is SM75 GPTQ-marlin, #53912 is H100 FP8 TP1, and neither has our CPU KV offload connector. Because we set no `--prefix-match-unit`, we sit on the coarse branch — the exact branch #43650 patches.

**Most discriminating probe.** The mechanism requires a dirty *producer*, so split warm from read:

- **Arm 1 (clean producer):** warm the shared prefix with MTP **off**, then serve the hits with MTP **on**.
- **Arm 2 (dirty producer):** warm with MTP **on**, then serve the hits with MTP **off**.

Only Arm 2 failing ⇒ the poison is in the cached block ⇒ #43650/#48375 is the right fix class. Arm 1 also failing ⇒ not the cached block, and the decode path is implicated. Controls: `temperature=0` + fixed seed so first-divergence position is meaningful; prefix lengths at an exact mamba-page multiple and at ±1 hash unit (#52244 shows block multiples behave differently); score `<tool_call>` leakage and needle recall separately; watch `vllm:prefix_cache_hits_total`. Make the shared prefix several multiples of the derived attention block — #53749 reports zero hits below one block, so a short prefix makes both arms pass vacuously. Prerequisite: read the deployed **mamba page and derived attention block size** off the startup log — fp8 KV raises the attention block to cover one mamba page (#53912 saw 800 → 1600), and that number sets the probe's prefix lengths and the cost in §3.

**Action:** run the probe on 0.27.1 before any upgrade work, then A/B #43650 on v9. Post the #51113 correction.

## 2. #51599 — open, not merged, not released

**Citation:** pull/51599, updated 2026-08-23, still in review (njhill re-review; "MRV1 is low priority right now"; a reviewer notes a larger-scoped fix is needed). Fixes #51571 (open): async align mode gathers previous-iteration accepted counts from `InputBatch` rows that `condense()` has already mutated, giving a request a state-copy offset that is too small.

**Does it make async safe on hybrids? No.** #53726 (08-26; RTX 3090 SM86, Qwen3.5 hybrid GDN 27B W4A16, MTP k=3, fp8 KV, FlashInfer, PIECEWISE cudagraphs — the closest config match in the corpus) reports a silent CUDA IMA that persists through #50021, #45100, #53613 *and* a partial #51599 consumer-side backport, and says it **"crashes with it off too"**. Upstream's async gain is small-batch only, and we run `max_num_seqs 8`.

**Action:** keep `--no-async-scheduling`; do not backport #51599. Separately, #53726 notes the API server exits with **status 0** after the IMA, so `Restart=on-failure` leaves the unit down silently — audit our restart policy regardless.

## 3. #52244 — open, contested, do not backport

**Citation:** pull/52244, open with merge conflicts, updated 2026-08-27. Restores hybrid GDN prefix-cache **hit depth** under MTP by writing mamba state one hash unit below the deepest boundary a replay can reach. Author's numbers (Qwen3.5-122B-A10B, TP4, MTP 3, `--prefix-match-unit 67`): every length now hits `(P-1-unit)//unit*unit` — 3000 → 1072 becomes 2881, 2144 → 0 becomes 2010.

**Two blockers.** (a) A reviewer objects that it adds a prefill chunk and hurts TTFT (`block=8, unit=4, len=18`: `0→16→18` becomes `0→12→16→18`). (b) An independent GB10 measurement in-thread confirms the defect (6400-token prompt caches 3200 where the back-off predicts 4800) but finds **the PR does not move it** — `last_cache_position` is unchanged; #50897 does move it (→ 4800/6400). The author points at **#50897** and **#53479** as the real direction.

**Match:** every gate is conditioned on fine-grained hashing being active. With `CTX=long` we set no `--prefix-match-unit`, so **this PR is a no-op on our config as deployed** (it would matter on the `CTX=huge` KVarN path, which sets unit 128). Backport size ~3 commits across the prefill splitter, `MambaManager`, `FullAttentionManager`; gain is warm-TTFT only, and only after adopting a `prefix_match_unit`.

**Action:** skip; track #50897 and #53479. The perf item that *does* hit us is **#53670** — the EAGLE last-block drop costs a 1,648-token recompute per hit on a Qwen3.8-27B GDN layout (15 MambaSpec groups, scheduler/hash block 1648): 97.5% → 86.9-91.7% hit rate, **−37% throughput at c=8**, recovered in an ablation that disables the drop. The issue itself argues at length that disabling it is not the remedy — treat the ablation as a measurement of the cost, not a proposed patch.

**Tension to hold:** #43650 says mamba must drop *one more* block (correctness); #53670 says the existing drop already costs ~100× more on hybrid layouts than dense ones. On our layout the correctness fix has a real throughput price — measure it in the same A/B, don't treat it as free.

## 4. New since 2026-08-26 (filtered to our config)

| Item | Status | Why it matters |
|---|---|---|
| **#53912** prefix cache + MTP still corrupts on 0.28.0 | open 08-26 | Names the read-path mechanism; the citation to use instead of #47194. Shows `--kv-cache-dtype fp8` only *masks* it by raising the block size until nothing hits. |
| **#53505** hybrid Mamba align corrupts under spec decode **with a KV connector attached, even at zero retrieved tokens** | open | 2×RTX 3090. **We run a 24 GiB CPU KV offload connector**; our A/B rig had none, so this vector is untested on our stack. Give it its own arm. |
| **#53726** silent CUDA IMA, exit 0, RTX 3090 + GDN + MTP k=3 + fp8 KV + FlashInfer | open 08-26 | Closest config match; uses `draft_sample_method: probabilistic` as we do. Async off does not help. |
| **#53887** MTP draft allocates a second full vocab embedding, OOMs a 27B INT4 target | open 08-26 | Exactly what our `0007-qwen35-mtp-d2t-draft-vocab` patch works around. Worth commenting with our 40k-head result. |
| **#53670** last-block drop → 1,648-token recompute, −37% at c=8 | open | Quantifies the cost side of the #43650 tradeoff on a Qwen3.8 GDN layout. |
| **#53749** zero prefix-cache hits for shared prefixes shorter than one attention block on hybrids | open | Explains dead short prefixes; no log or metric names the minimum. |

**FlashInfer fp8 KV on Ampere** and **acceptance improvements:** nothing surfaced in an 80-item sweep of items updated since 08-25 — not an exhaustive search. The only acceptance issue in range, #54011 (DSpark adaptive verification collapse on SM90), is not our method.

**`use_heterogeneous_vocab`** builds a token-level intersection of the two vocabularies at init and constrains draft logits to shared tokens (TLI). **It cannot help our 40k draft-vocab head:** `speculative.py:1324` raises unless `method == "draft_model"`, and we are `method="mtp"`; it also forces `draft_sample_method="greedy"` (`:1329`). Our patch stays the only route on the MTP path.

## 5. SpeculativeConfig, vLLM 0.27.1 — every tunable field

Source: `vllm/config/speculative.py`. ✗ = we do not set it.

| Field | Default | Notes |
|---|---|---|
| `num_speculative_tokens` | required, >0 | we set 3 |
| `method` | None | we set `"mtp"` |
| `model` | None | ✗ MTP head resolves from the checkpoint |
| `draft_sample_method` | `"greedy"` | **the temp>0 lever, and we already set it.** Deployed JSON (`start_qwen.sh:259`) is `{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}`. Probabilistic samples from the draft distribution and uses full draft logits in the ratio test, at the cost of extra GPU memory. (`run_qwen_v028.sh` omits it — that A/B arm ran greedy.) |
| `rejection_sample_method` | `"standard"` | ✗ `"block"` = joint block verification (Sun et al.), the one genuine acceptance lever to A/B at temp>0. `"synthetic"` is **not** a speedup — it fakes acceptance at a calibrated rate and degrades output. |
| `synthetic_acceptance_rates` / `_length` | None | ✗ only valid with `"synthetic"`, mutually exclusive |
| `num_speculative_tokens_per_batch_size` | None | ✗ **the dynamic-k / acceptance-aware-k lever, present in 0.27.1**: `list[(range_start, range_end, k)]`, inclusive ranges. `skip_draft_when_k0` (seen in #53670) is 0.28+, not in our tree. |
| `disable_padded_drafter_batch` | False | ✗ EAGLE path only |
| `parallel_drafting` | False | ✗ auto-set for dflash/dspark; needs a drafter trained for it |
| `use_local_argmax_reduction` | False | ✗ TP-communication saving, greedy non-tree drafting only |
| `use_heterogeneous_vocab` | False | ✗ unreachable on MTP (§4) |
| `kv_cache_dtype` | None | ✗ draft inherits target fp8 |
| `attention_backend` | None | ✗ draft backend override |
| `moe_backend` | None | ✗ |
| `quantization` | None | ✗ draft-model method only |
| `enforce_eager` | None | ✗ overrides the target's setting for the draft |
| `draft_tensor_parallel_size` | None | ✗ must be 1 or target TP |
| `tensor_parallel_size` | None | ✗ trap field — exists only to warn on the wrong argument name |
| `max_model_len` | None | ✗ used to test skipping speculation |
| `revision` / `code_revision` | None | ✗ |
| `draft_load_config` | None | ✗ |
| `prompt_lookup_max` / `prompt_lookup_min` | None | ✗ ngram only |
| `suffix_decoding_max_tree_depth` | 24 | ✗ suffix method only |
| `suffix_decoding_max_cached_requests` | 10000 | ✗ |
| `suffix_decoding_max_spec_factor` | 1.0 | ✗ |
| `suffix_decoding_min_token_prob` | 0.1 | ✗ |

No draft-temperature or draft top-k truncation field exists in 0.27.1. The documented acceptance levers are exactly three: `draft_sample_method`, `rejection_sample_method="block"`, and `num_speculative_tokens_per_batch_size`.

## Recommended order

1. Read the mamba page and derived attention block size off the startup log (the deployed spec JSON is settled: mtp, k=3, probabilistic).
2. Run the two-arm clean-vs-dirty-producer probe on 0.27.1 (§1), plus a third arm with the KV offload connector detached (#53505).
3. If Arm 2 fails, A/B #43650 on v9 and measure the #53670 hit-depth cost in the same run.
4. Post the #51113 correction to #47194.

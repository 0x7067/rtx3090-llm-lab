# vLLM 0.27.1 findings — Qwen3.8-27B GDN hybrid on RTX 3090

Release timeline: v0.27.1 = 2026-08-11, **v0.28.0 = 2026-08-26 (one day old)**.
Containment for every PR below was verified with `compare/<merge_sha>...<tag>`.

## 1. Empty content + empty reasoning_content at finish_reason "length" — NOT A BUG

Two independent causes, both confirmed in source. Neither is a truncation defect.

**a) `reasoning_content` no longer exists on output.** Renamed to `reasoning` (#27752,
2025-11-08), then removed from output entirely in
[#33402](https://github.com/vllm-project/vllm/pull/33402) (merged 2026-01-30) —
**verified present in your 0.27.1**. `vllm/entrypoints/openai/chat_completion/protocol.py:71`
declares `reasoning: str | None`. The input side still normalizes `reasoning_content`
→ `reasoning` (#42664, also in 0.27.1, protocol.py:529-531), so the break is output-only
and silent. Documented as a breaking client change in
[#50624](https://github.com/vllm-project/vllm/pull/50624) (merged 2026-08-03, in 0.28.0).

**b) Empty `content` on truncation is correct by construction.**
`vllm/parser/engine/parser_engine.py:493-517` (non-streaming `extract_reasoning`):

```python
raw_reasoning = "".join(reasoning_parts)
reasoning = raw_reasoning or None
content = "".join(content_parts) or None
```

Truncation mid-`<think>` yields only REASONING_CHUNK events and zero TEXT_CHUNK events,
so `content` is `None`. `incremental_lexer.flush()` (line 128-134) emits the buffered tail and `_engine.finish()`
is called at line 500, so the partial CoT should land in `reasoning` — but I did not read
the lexer-token → REASONING_CHUNK mapping, so verify rather than assume:

> **One-request check:** send a request that hits `max_tokens` and inspect the raw JSON for a
> `reasoning` field. If it holds the partial CoT, this is purely the field rename — done.
> If `reasoning` is *also* empty, the parser is dropping the unterminated buffer. That would
> be a real defect and worth filing, since nothing upstream reports it.

The recommendation below is correct either way, because the rename is independently verified.

**Action (config/client only, no patch, no upgrade):** read `reasoning`, falling back
defensively as #50624 recommends:
`getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)`.
Also confirm no client sends `include_reasoning: false` — it defaults `True`
(protocol.py:258) and setting it false nulls the field.

## 2. Empty reply, finish_reason "stop", on prefix-cache hits — NO EXACT MATCH

**No issue found matching "empty output with finish_reason stop on a prefix-cache hit."**
Searched: `mamba prefix caching empty output`, `hybrid empty output`, `GDN prefix cache`,
`gated deltanet prefix cache corruption`, `mamba_cache_mode`, `mamba cache align`.
Same-class issues, none reporting empty output:

- [#47194](https://github.com/vllm-project/vllm/issues/47194) — **OPEN**, 2026-06-30.
  Closest by stack: Qwen3.6 hybrid + prefix caching + MTP3 + `qwen3_coder` tool parser +
  `qwen3` reasoning parser + fp8 KV. Cache-hit path gives tool-call leakage and needle-recall
  failure; the no-MTP path is correct. Symptom is corruption, not empty output. No fix.
- [#51812](https://github.com/vllm-project/vllm/pull/51812) — **MERGED 2026-08-11T15:35Z,
  ~5h after 0.27.1 shipped. In 0.28.0 only, NOT in your build.** Real GDN bug: in a mixed
  batch where a non-speculative row precedes speculative rows, `mixed_qkv` is gathered with
  `spec_token_indx` but the `a`/`b` gates are not, so gates pair with the wrong tokens.
  Measured effect is logprob drift (max error 0.0205 → 0.0017), greedy IDs unchanged in
  their repro — so it does not by itself explain an empty reply. **One file**
  (`qwen_gdn_linear_attn.py`) — the cleanest backport candidate you have.
  **Precondition: V1 model runner only.** The PR states the V2 runner ordered speculative
  tokens first and was already unaffected. On V2 the backport is a no-op.
  **Applies to your model:** Qwen3.8 has no registry entry of its own, so it runs as
  `Qwen3_5ForCausalLM` or `Qwen3NextForCausalLM`; both import `QwenGatedDeltaNetAttention`
  from the exact file this PR patches.
- [#52244](https://github.com/vllm-project/vllm/pull/52244) — OPEN. GDN prefix-cache hit
  *depth* under MTP (hits collapse to 0 at page multiples). Performance, not correctness.
- [#51571](https://github.com/vllm-project/vllm/issues/51571) — OPEN. Wrong accepted counts
  → "repeated, dropped, or garbled tokens" on hybrid GDN + align mode. **Requires async
  scheduling**, which you have off. See item 4 — this is why `--no-async-scheduling` matters.

**Action:** backport #51812. To isolate, re-run a failing prompt with
`--no-enable-prefix-caching` and separately with spec decode off; that splits #47194's
class from everything else. If it reproduces, #47194 is the issue to comment on with your
repro — it is open and under-evidenced.

## 3. MTP + FlashInfer EngineDeadError at num_speculative_tokens=4 — NOT FOUND

**No upstream issue, PR, or fix matches this.** Searched: `EngineDeadError`,
`flashinfer mtp`, `flashinfer spec decode`, `mtp num_speculative_tokens 4`,
`spec decode request finishes crash`, `flashinfer wrapper plan crash decode`.
Nearest, all non-matching:

- [#37754](https://github.com/vllm-project/vllm/issues/37754) — OPEN. FlashInfer+MTP crash,
  but SM121 (DGX Spark) with GQA=16. Different arch.
- [#40756](https://github.com/vllm-project/vllm/issues/40756) — OPEN. MTP illegal memory
  access on long sequences, Qwen3.6-27B-FP8, v0.19.1. Long-context trigger, not batch churn.
- [#36613](https://github.com/vllm-project/vllm/issues/36613) — CLOSED. MTP ILM under high
  concurrency, Qwen3.5-397B.

**Action: stay at `num_speculative_tokens: 3`.** You are not in an outage — this is a
"can we raise it" question, and there is no upstream fix to raise it with. If you want it
addressed, file an issue: your trigger (one request finishing while another is mid-generation
at spec_tokens=4, on Ampere + fp8 KV FlashInfer) is not represented upstream.

## 4. Async scheduling + spec decode on hybrids — STANDING hazard, not an upgrade regression

**Correction worth stating plainly: this is not new in 0.28.0.** I compared the
default-resolution block in `vllm/config/vllm.py` at both tags. **v0.27.1 already resolves
`async_scheduling` to `True`** when the spec method is in `EagleModelTypes`, and `"mtp"` and
`"qwen3_next_mtp"` are both in that list (`vllm/config/speculative.py:49-55`). 0.28.0 only
adds `draft_model` to the allowed set (#48341). Neither version has a Mamba/hybrid exclusion
in that path.

So `--no-async-scheduling` is already doing real work on your current build. It is not
belt-and-braces.

Why it matters: [#51571](https://github.com/vllm-project/vllm/issues/51571) (OPEN) is exactly
async scheduling + MTP + hybrid Mamba/GDN + `mamba_cache_mode="align"`. Accepted counts are
read from mutable `InputBatch` rows after `condense()`, giving a wrong Mamba/GDN state-copy
offset and "repeated, dropped, or garbled tokens." The proposed fix
[#51599](https://github.com/vllm-project/vllm/pull/51599) is **open, not merged, not in 0.28.0**.
#51571 also notes align mode is already the default for hybrid models with prefix caching on.

**Action: keep `--no-async-scheduling` explicitly, on this version and any future one.**
Treat it as required. Do not let it get dropped during an upgrade.

## 5. Ampere decode wins after 0.27.1 — one strong candidate

[**#51674**](https://github.com/vllm-project/vllm/pull/51674) — MERGED 2026-08-14,
**in 0.28.0 only**. Fused CUDA post-conv MTP decode kernel for Qwen3.5 GDN. Replaces the
per-step chain of small Triton kernels (gating, delta-rule recurrence, state rewind/update,
gated RMSNorm) that leaves MTP decode latency-bound at small batch — exactly your regime
(single 3090, MTP=3). **It applies to your GPU**: the CMake arch list is
`"8.0;8.6;8.9;9.0a;10.0f;12.0f"` and the test gate is compute capability 8.0+; RTX 3090 is
8.6. Controlled by `VLLM_GDN_DECODE_KERNEL`, default `"cuda"`, with `"triton"` as fallback.
The PR description frames the speedup on Blackwell, so **the Ampere gain is unquantified
upstream — benchmark before and after rather than assuming it.**
Backport cost is real: new `.cu` source, CMakeLists, torch bindings, `_custom_ops.py`,
`compilation.py`, envs. Heavier than #51812.

Also in 0.28.0, lower value for you:
- #49436 — 3D-grid tiling of the Mamba state-copy Triton kernels.
- #50991 — prefix caching on by default for Mamba models.
- #51726 — `max_num_batched_tokens` default 8192 → 16384 (you set chunked prefill at 2048).

**No fp8-KV FlashInfer plan-reuse change was found** in this window. Not stretching to fit one.

## Recommended sequence

1. **Now, zero risk:** switch the client to read `reasoning`. Closes item 1 outright.
2. **Now:** backport #51812 (single file). Correctness, cheap, no behavior surface.
3. **Measure:** split item 2 with `--no-enable-prefix-caching` vs. spec-decode-off runs.
4. **Hold** `num_speculative_tokens: 3` — no upstream fix exists for item 3.
5. **Do not blanket-upgrade to 0.28.0.** It is one day old. If you do upgrade, keep
   `--no-async-scheduling` (open bug #51571 is unfixed in 0.28.0) and re-verify the
   mamba/prefix-cache flags — #50991 and #51726 changed defaults.

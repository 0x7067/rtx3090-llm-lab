# Empty replies in the 2026-08-26 qwen3.8-27b re-run — evidence

Pure file analysis. No server, GPU, docker, or kubectl was touched.

## 1. Request shape (read from the harnesses)

- Both harnesses POST `/v1/chat/completions` **non-streaming** (`httpx.AsyncClient.post`, no
  `stream` field), `temperature: 0`, `max_tokens: 12288`, no `tools`, no `chat_template_kwargs`,
  no `reasoning_effort`. `timeout=900`, 3 retries.
- Both read `choices[0].message.content`, `.reasoning_content`, and `choices[0].finish_reason`.
  **Neither harness reads `usage`.**
- `bench_quality.py:60-68` records `raw_len=len(content)` and `reasoning_len=len(reasoning)`.
- `bench_multiturn.py:71-73` returns `finish_reason + "/reasoning-only"` when content is empty
  *and* reasoning_content is non-empty. **No recorded turn carries that suffix**, so
  `reasoning_content` was empty on all 24 no-code turns too.
- Server flags (`run_qwen.sh`): vLLM 0.27.1, `PREFIX_CACHE=1`, `DRAFT_TOKENS=3` (MTP),
  `--enable-auto-tool-choice --tool-call-parser qwen3_coder`,
  `--chat-template /app/config/chat_template.jinja`. **No `--chat-template-kwargs`**, so the
  template's `reasoning_effort` default of **`xhigh`** applied. Production
  (`configmap.yaml:120`) pins `medium`, and its own comment puts xhigh at "22k reasoning
  tokens on trivial tasks".

## 2. The 15 single-shot defects (cand_qwen_v2.jsonl)

All 15: `finish_reason: "length"`, `raw_len: 0`, `reasoning_len: 0`, `error: null`. Token counts are the templated prompt (chat_template.jinja, `add_generation_prompt`, tokenizer.json md5 `4010c9c0…`, identical across all three model dirs), at the xhigh default.

| task | user chars | tokens | mod 128 | | task | user chars | tokens | mod 128 |
|---|---|---|---|---|---|---|---|---|
| HumanEval/10 | 785 | 243 | 115 | | HumanEval/116 | 667 | 267 | 11 |
| HumanEval/24 | 377 | 144 | 16 | | HumanEval/129 | 1565 | 500 | 116 |
| HumanEval/32 | 964 | 359 | 103 | | HumanEval/134 | 725 | 231 | 103 |
| HumanEval/39 | 465 | 196 | 68 | | HumanEval/137 | 659 | 237 | 109 |
| HumanEval/47 | 373 | 174 | 46 | | HumanEval/145 | 619 | 218 | 90 |
| HumanEval/56 | 553 | 188 | 60 | | HumanEval/157 | 588 | 203 | 75 |
| HumanEval/76 | 631 | 234 | 106 | | | | | |
| HumanEval/91 | 577 | 203 | 75 | | | | | |
| HumanEval/99 | 818 | 260 | **4** | | | | | |

Residue 4 hits 1 of these 15 and **0 of the 149 non-defective** prompts. Prompt length barely separates the groups: defective median 231 tok (144-500) vs 218 tok (136-429).

**`finish_reason: length` does not by itself produce an empty body in this stack.** Two
counterexamples: single-shot HumanEval/160 (`length`, 1343 chars of unfenced truncated code) and multi-turn HumanEval/145 turn 1 (`length`, 381 chars ending mid-docstring). Both hit the budget and returned partial text. The split is: reasoning block closed → truncated answer survives; reasoning block never closed → nothing survives.

## 3. The 24 multi-turn no-code turns

22 `length`, 2 `stop`. 23 of 24 have `raw_len: 0`; the exception is HumanEval/145 turn 1 (`length`, 381). Tasks 76/124/132/146 burned all three turns; 75/145 turns 1-2; 99 turns 1 (len) / 2 (**stop**) / 3 (len); 134 turn 1 (**stop**); 39/71/102/105 turn 1 only.

Templated turn-1 token counts for the 12 affected tasks: 336-703 (median 428) vs 195-784 (median 390) for the other 113. No separation.

**The two empty-stop cases, exactly reconstructed:**

| case | chars | xhigh | medium | low | enable_thinking=false |
|---|---|---|---|---|---|
| HumanEval/134 t1 | 1090 | 352 → **r96** | 314 → r58 | 340 → r84 | 316 → r60 |
| HumanEval/99 t2 | 1458 | 460 → **r76** | 422 → r38 | 448 → r64 | 424 → r40 |

HumanEval/99 t2 is deterministic: turn-1 messages + `{"role":"assistant","content":""}` + the literal nudge. That empty assistant turn renders as `<|im_start|>assistant\n<think>\n\n</think>\n\n<|im_end|>\n`.

**Neither case lands on residue 4 under any template variant, and the two residues differ by
20 mod 128 — so no constant rendering offset can put both at 4.** That negative does not depend on my rendering being byte-exact with vLLM's. It refutes the residue rule as a *common* explanation for the two empty-stop replies; it does not refute it for either one alone. Across all 125 mutant turn-1 prompts the residues are uniform (largest bucket 4 of 125). Open caveat: nothing here shows 128 is special in this stack — I did not verify vLLM's block size from source.

## 4. What the installed vLLM 0.27.1 source says

Read from `/data/buttercup_6tb/k3s/vllm-trial/venv/lib/python3.12/site-packages/vllm`. `--reasoning-parser qwen3` resolves to `vllm/parser/qwen3.py:qwen3_config` via `Qwen3ParserReasoningAdapter`.

- Non-streaming `ParserEngine.extract_reasoning` (parser_engine.py:493-518) returns
  `(reasoning or None, content or None)`, built from `REASONING_CHUNK` and `TEXT_CHUNK` events.
- `StreamingParserEngine._emit_for_state` (streaming_parser_engine.py:359-374) emits
  `config.content_events[self.state]` for accumulated text; qwen3 does not override the
  default map, so `REASONING → REASONING_CHUNK`. **Verified, not inferred:** text accumulated
  in the REASONING state is emitted on the non-streaming path.
- `strip_trailing_reasoning_whitespace=False` for qwen3, so nothing is rstripped away.
- `include_reasoning` defaults to `True` (protocol.py:258); serving only nulls reasoning when a
  client sets it false, which neither harness does.
- Non-streaming `parse()` never calls `adjust_initial_state_from_prompt`, so `initial_state`
  comes from `qwen3_config(thinking=…)`. **Both values give the same prediction**: `REASONING`
  → 12k of `REASONING_CHUNK`; `CONTENT` → 12k of `TEXT_CHUNK` (the `(CONTENT, THINK_END)`
  absorb transition makes good replies look identical either way). So this gap does not matter.

**Therefore: for any non-empty `output.text`, at least one of `reasoning_content` / `content`
must be non-empty.** An unclosed `<think>` yields a large `reasoning_content`, not an empty pair. The recorded empty pair implies `output.text` was empty.

Corroborating oddity: `reasoning_len == 0` on **all 164** rows including the 148 good ones, whose content starts with `"\n\n```python"`. That is only consistent with the model emitting `</think>` as its first generated token — zero reasoning — despite the xhigh "think carefully" system prompt.

## 5. Cross-run and cross-model comparison

- v1 (6144) vs v2 (12288) single-shot: 22 empties → 15. **13 of the 15 recur** (32, 39, 47, 56,
  76, 91, 99, 116, 129, 134, 137, 145, 157). New in v2: 10, 24. Fixed by the bigger budget:
  22, 94, 124, 125, 127, 132, 146, 147, 160.
- v1 vs v2 multi-turn (derived from `work_mt_qwen*/turnN/candidates.jsonl` against the active
  set): turn-1 no-code 18 → 12, overlap 9 (39, 75, 76, 99, 105, 124, 132, 145, 146). Unsolved
  sets differ by one task (v1 has 129, v2 has 146).
- **This does not discriminate.** At temperature 0, both a >12k reasoning overrun and a
  prompt-deterministic serving bug predict the same tasks recurring.
- **Tiel ran on a different stack.** `swap_to_tiel.sh` starts `ghcr.io/ggml-org/llama.cpp:full-cuda`
  with `--jinja` and the model's own template — llama.cpp, not vLLM. The 15-vs-1 headline is
  cross-stack. Notably llama.cpp *also* produced one empty-on-length reply (HumanEval/145,
  `length`, raw_len 0) plus one partial (HumanEval/32, `length`, 1160 chars).
- **Not recoverable from these files:** completion order and server uptime.
  `asyncio.as_completed` plus the by-`task_id` rebuild destroys ordering, and no timestamps
  were recorded. Any "clusters late in the run" claim cannot be tested offline.

## 6. Ranked hypotheses

**H1 (concluded in §4) — the engine returned an empty string on a length-capped generation.**
The parser source cannot produce `(None, None)` from non-empty text; two other `length` replies *did* return partial text, so the empty body is not a property of hitting the cap. `usage` was never recorded, so nothing in the data establishes that ~12k tokens were actually generated — that is an inference from `finish_reason` alone. Candidate mechanisms, ranked:

- **H1a — MTP speculative decoding.** `DRAFT_TOKENS=3` with `PREFIX_CACHE=1`. This repo already
  ships one MTP-triggered vLLM patch (`vllm-patches/xgrammar-terminated-batch.patch`: "MTP
  speculative decoding can return a token batch whose grammar stop token is followed by unused
  draft tokens"). Against: that patch is scoped to xgrammar structured output, which neither
  harness requests.
- **H1b — detokenizer or output assembly drops the text at the cap.** Fits the clean
  length-vs-partial split; no direct evidence either way.
- **H1c — scheduler returns the request with zero generated tokens.** Would make
  `finish_reason: length` spurious; `usage` settles it in one request.

**H2 — reasoning overruns 12,288 tokens and the unclosed block is lost.** For: `reasoning_effort`
fell back to `xhigh` because `run_qwen.sh` omits the kwargs production sets; the configmap puts xhigh at ~22k reasoning tokens, above both 6144 and 12288; llama.cpp shows the same shape once. Against: §4 shows unclosed reasoning is routed to `reasoning_content`, which was empty, and the model emitted zero reasoning on the other 148. Still the cheapest thing to rule out.

**H3 — templated length residue ≡ 4 (mod 128) plus a prefix-cache hit.** For: 1 of 15
single-shot defects hits residue 4 (HumanEval/99) against 0 of 149 controls. Against: §3 — the two empty-stop replies it was proposed for both miss, under every variant, and no offset rescues both; the 125 multi-turn residues are uniform.

**H4 — the empty assistant turn poisons the next prompt.** For: HumanEval/99 turn 2, one of the
two empty-stop cases, carries exactly that degenerate `<think>\n\n</think>` assistant turn, and failing tasks tend to keep failing (76, 124, 132, 146 burn all three turns). Against: HumanEval/134's empty-stop is turn **1**, with no prior assistant turn. Confounded regardless.

## 7. Discriminating experiments (one request each)

1. **Did the server generate tokens it did not return?** Re-send HumanEval/129 single-shot
   verbatim (500 tok, residue 116 — the longest prompt, and unconfounded by H3), **twice**, so
   the second request is a prefix-cache hit:
   ```
   curl -s localhost:8094/v1/chat/completions -H 'Content-Type: application/json' -d '{
     "model":"qwen3.8-27b","temperature":0,"max_tokens":12288,
     "messages":[{"role":"user","content":"<PROMPT_TMPL for HumanEval/129>"}]}' \
   | jq '{fr:.choices[0].finish_reason, c:(.choices[0].message.content|length),
          r:(.choices[0].message.reasoning_content|length), u:.usage}'
   ```
   `completion_tokens ≈ 12288` with both lengths 0 confirms H1 and refutes "the model answered
   normally". `completion_tokens ≈ 0` selects H1c. A cold/warm difference also tests H3.
2. **Same request with `"stream": true`.** Streaming uses `parse_delta`, a different path that
   does call `adjust_initial_state_from_prompt`. Deltas arriving while the non-streaming call
   returns empty puts the defect in `extract_reasoning`; nothing streaming either confirms H1.
3. **Same request plus `"chat_template_kwargs":{"reasoning_effort":"medium"}`.** A clean reply
   confirms H2 as the trigger and makes the fix one line in `run_qwen.sh`.
4. **Isolate MTP:** repeat request 1 against a server started with `DRAFT_TOKENS=0`. A clean
   reply isolates H1a.

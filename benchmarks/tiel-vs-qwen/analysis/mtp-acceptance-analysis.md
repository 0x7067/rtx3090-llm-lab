# MTP acceptance analysis — Qwen3.8-27B, vLLM 0.27.1, k=3

## 0. The 0.595 figure is a slot ratio, not a chainable probability

`a+a²+a³` needs a **conditional** per-position acceptance. `num_accepted/num_draft` is already averaged over positions where the chain had broken, so chaining it again double-counts breaks. Arithmetic rules the other readings out, using no repo number:

| reading of 0.595 | tok/step | implied step | verdict |
|---|---|---|---|
| conditional, `1+a+a²+a³` | 2.160 | 22.6 ms | **impossible** — 0.86 ms over the 21.74 ms no-spec step; 3 draft passes alone cost ~1.5–3 ms (optimizations.md item 5) |
| accepted per *step* (`/num_drafts_total`) | 1.595 | 16.7 ms | **impossible** — under the no-spec floor |
| slot ratio (`/num_draft_tokens_total`, k×steps) | **2.785** | **29.13 ms** | consistent |

I assume the third: accepted/step = 3 × 0.595 = 1.785, tok/step = 2.785 (the repo counts the bonus token — "no speculation → 1.0 tokens/step", README:246). Derived step **29.13 ms**; no-spec 21.74 ms; speculation overhead 7.39 ms. Equivalent geometric conditional **p ≈ 0.762**, which matches the repo's independent "~75–77% of first drafts" (README:299).

## 1. Acceptance → tok/s (k=3, step fixed at 29.13 ms)

Reading A — p as conditional per-position acceptance (your formula):

| p | tok/step | tok/s | vs today |
|---|---|---|---|
| 0.595 | 2.160 | 74.1 | −22% |
| 0.65 | 2.347 | 80.6 | −16% |
| 0.70 | 2.533 | 86.9 | −9% |
| 0.75 | 2.734 | 93.9 | −2% |
| **0.762 (today)** | **2.785** | **95.6** | — |
| 0.80 | 2.952 | 101.3 | +6% |
| 0.85 | 3.187 | 109.4 | +14% |
| 0.90 | 3.439 | 118.1 | +24% |
| 1.00 | 4.000 | 137.3 | +44% (k=3 ceiling) |

**Do not read this table as "your targets are regressions."** If 0.595 is a slot ratio, so are your 0.65/0.70/0.75 targets — and those are real gains. The correction is that today's *conditional* is 0.762, not 0.595. Bridge (solve `p+p²+p³ = 3a`):

| slot ratio (what /metrics reports) | conditional p | tok/step | tok/s | vs today |
|---|---|---|---|---|
| **0.595 (today)** | **0.762** | 2.785 | **95.6** | — |
| 0.65 | 0.800 | 2.95 | 101.3 | +6% |
| 0.70 | 0.832 | 3.10 | 106.4 | +11% |
| 0.75 | 0.863 | 3.25 | 111.6 | +17% |

So slot 0.70 needs a conditional 0.832 — seven points of per-position acceptance above today's 0.762 — and is worth +11%. Reading A below is what the same p values give under the literal `a+a²+a³` formula, to show why the two must not be mixed.

Reading B — a as the slot ratio you measured (accepted = 3a): 0.595 → 95.6 (today); 0.65 → 101.3; 0.70 → 106.4; 0.75 → 111.6; 0.80 → 116.7.

Non-geometric check, from the repo's measured per-position profile (README:31, C1 T=1.0, cumulative 74/50/34/24): first three sum to 1.58 → 2.58 tok/step → 88.6 tok/s. Its conditionals are 0.74 then a **plateau at ~0.68** (0.676/0.680/0.706). That plateau is why k=5 and k=6 measured slower — not the head position.

## 2. Vocab misses are ~8% of the rejection, and the headroom is unreachable

A miss is a certain rejection, so coverage `c` caps per-position acceptance at `c`. (c=0.975 is held-out coverage on `gen_data.py` outputs; the repo does not state that run's sampling, so if it differs from production T=1.0/top-k 20 the true c shifts — this does not move the verdict, since the 49k measurement independently closes the bigger-vocab route.) With c = 0.975 and p = 0.762, quality-limited acceptance q = p/c = 0.7815.

- Misses cost Δp = 0.0195 of the 0.238 total rejection = **8.2%**.
- Perfect coverage: 2.785 → 2.869 tok/step = **+3.0% (98.5 tok/s)**. That is the *entire* vocab ceiling, and it is unreachable: c=1 means the full 248k head, which costs more per draft than it returns (ladder row "k=4, full 248k head" 74%/76% pos-0 vs 74%/74% for the shipped 40k).
- Code (c=0.96, p=0.766 from 96.1 tok/s): q=0.798, Δp=0.032 of the 0.234 rejection = **13.6%** (same multiplicative method as the overall figure).

**Bigger heads are measured dead.** drafter/README.md:21-22, gotchas.md:59-61: coverage saturates near 40k (49k → 98.2%), the model only ever emits ~54k distinct tokens, and the 49k head **measured no faster** (109/115 vs 114/124). 60k/80k asymptote at ~98.2–98.5% — ≤1.3 points, ~+1.5% tok/step, against a bigger head read on every draft. Net ≤ 0.

**No coverage curve is stored.** build_draft_vocab.py:84-87 prints coverage at N=16384/32768/49152/65536, but only on the `--corpus` path (`if not ids_file`, line 82). No log is committed and the own-output corpus is not in the repo, so the curve needs the 2.2 h GPU generation step to reproduce.

## 3. Rebuild recipe — the expensive part does not exist

**Deployed vocab verified, not assumed.** `mtp_draft_vocab_ids.pt` is md5 `25270169e123eb83…`, identical across all three model dirs, and its 40,960 int64 ids are **byte-identical** to `prepare/draft_vocab_ids.json` (0 ids differ either way). `git log --follow` shows the list was *replaced* in e964409 (2026-08-18), the "Single-user: 114 tok/s / 124 greedy" commit. So the deployment carries the 97.5% own-outputs list, per optimizations.md:81 and README:266. → **build_draft_vocab.py's docstring (lines 20-23: "Danish web text … 8.8M tokens, held-out coverage 95%") is stale**, describing the superseded 92% list. Worth a one-line fix.

Shipped list: 40,960 ids counted over 5.4M tokens of the model's own outputs (`drafter/gen_data.py`, 2.2 h GPU) over 6.8k prompts from `collect_prompts.py` (UltraChat, Magicoder, syvai/da-instruction, syvai/reasoning-v1, skolegpt-instruct, GSM8K; 45% thinking on).

Storage: `mtp.draft_lm_head.{weight_packed,weight_scale,weight_shape}` appended to `model_extra_tensors.safetensors` + index entries, ids as `mtp_draft_vocab_ids.pt` (int64 [40960]).

Consumer: **`patches/qwen3_5-mtp-draft-vocab.patch`** on `models/qwen3_5_mtp.py`. Builds a `ParallelLMHead(len(ids))`, scores only those rows, `index_copy_`s them into a `-inf` full-vocab row. Guarded by `MTP_DRAFT_VOCAB` (0 → shared full head). Applied by `Dockerfile:29` (`for p in patches/*.patch`), so "patch 0007" is its glob position, not a numbered series.

**The GPTQ draft lm_head does NOT need requantizing when the vocab changes.** build_draft_vocab.py does `wp.index_select(0, ids_t)` / `ws.index_select(0, ids_t)` — a row gather from the already-quantized head. Group-128 quantization runs along K, so each kept row carries its own scales unchanged. Changing the vocab is a **CPU-only reslice (seconds) plus one restart**. Low risk: it backs up to `…safetensors.bak-draft`, and `MTP_DRAFT_VOCAB=0` reverts behaviour without touching files. The only expensive step is producing a *differently counted* list: 2.2 h GPU for `gen_data.py`.

## 4. Prose gap is entropy, not vocab — coverage inverts the hypothesis

Implied per class at 29.13 ms: code 2.800 tok/step (p≈0.766), prose 2.604 (p≈0.718), repair 2.884 (p≈0.785) — a 5-point per-position gap between code and prose.

**Code coverage is 96%, *below* the 97.5% overall, yet code is the fastest class.** Coverage predicts the opposite of the measured ranking, so a code-tuned vocab cannot explain the prose gap. Prose is intrinsic entropy: code and repair carry low-entropy structure (indentation, syntax, repeated identifiers, copied spans) a single-layer chain drafter predicts well. Repair's 99.0 is not lookup drafting — that patch fires only on the DFlash2 path (optimizations.md item 8), so MTP mode gets none of it.

Caveat: this holds step time fixed across classes. Minor at C1, but if prose prompts carry longer contexts, verify attention inflates their step and part of the 89.4 is step time rather than acceptance — confirm with per-class per-pos metrics (#5) before acting. No per-domain coverage data exists in the repo beyond the single "96% on code" figure.

## 5. Every other acceptance lever, and what the repo measured

| lever | state | measured |
|---|---|---|
| `draft_sample_method=probabilistic` | ON | +15% at T>0 (ladder 97→114); identical at greedy (README:365) |
| `VLLM_DRAFT_TOPK_TOPP=1` (drafts truncated to target's top-k/top-p) | ON | part of a "+4% at default sampling" bundle with split-KV verify. Matters here: production runs top-k 20 |
| `VLLM_DRAFT_TEMP_SCALE` (<1 sharpens draft) | 1.0 | "measured no gain here" (sampler patch header:16-17). Not stated to have been retested at k=3/CTX=long |
| `k` (`DRAFT_TOKENS`) | 3 | k=4 fastest (114/124) but crashes on FlashInfer+fp8 KV; k=2 −5%; k=5 106; k=6 76 |
| `MTP_DRAFT_VOCAB=0` (full 248k head) | 1 | "more acceptance, slower per draft"; 74%/76% pos-0 vs 74%/74% |
| 49k vocab | 40k | 109/115 vs 114/124 — no gain |
| MTP head fine-tuning | not shipped | **no gain**: top-1 agreement 0.685 → 0.685 |
| DFlash2 drafter | MTP | 3.14–3.34 tok/step vs 2.8–2.9, 117.8/125.7 at CTX=fast; *loses* at 12–36k context (2.3–2.6 vs 2.6–3.0) and from C8 |
| `mtp.fc` in bf16, rest int8 | int8 | ladder: 88/96 tok/s, 2.6/2.6 tok/step, **67%/70% pos-0** vs 69%/70% all-int8 — flat to slightly worse, costs memory |
| prefix caching (`PREFIX_CACHE=1`) | — | acceptance **unaffected**: 2.23/2.03/2.28 tok/step with vs without (README:129). Big prefill win, not an acceptance lever |
| `DRAFT_ATTN_BACKEND` / `DRAFT_KV_DTYPE` (start_qwen.sh:254-259) | unset | knob routes only the drafter off FlashInfer; **no measurement reported** |

## 6. Verdict: step time binds first, drafter quality second, vocab not at all

The 29.13 ms step is **11–17% slower** than the repo's own MTP step (~24.8 ms at k=4 CTX=fast, ~26.3 ms implied for k=3 fast variant). That gap is what `CTX=long` buys 150k context with: it loses the split-KV verify attention (`VLLM_SPEC_DECODE_ATTN` is bf16-KV only, gotcha 13) and pays FlashInfer/fp8. Recoverable headroom, ranked: **step time and k (config, ~+15%) > drafter quality (needs a new drafter, ~+14% at p=0.85) > vocab (≤+3%, unreachable)**. Vocab was the binding constraint one list ago; that fight is won and the repo's own numbers say so.

## 7. Ranked next experiments

1. **`MTP_DRAFT_VOCAB=0`, one restart.** Turns my 8.2% estimate into a measurement of the whole vocab-miss contribution, per class. Zero build cost. Expect ~+2 points pos-0 and a net tok/s loss. If prose gains materially, section 4 inverts — run this first.
2. **`CTX=fast` (k=4, bf16 KV, split-KV verify)** if 64k context suffices: repo-measured 114/124 vs 95/100. Biggest available win, config-only.
3. **`DRAFT_ATTN_BACKEND=FLASH_ATTN` (+`DRAFT_KV_DTYPE`) with `CTX=long` k=4.** Untested route to k=4 keeping the 150k fp8 target KV — the crash is in the FlashInfer spec-decode path and the repo built this knob for it. Soak C2/C4/C8 with staggered finishes first.
4. **`VLLM_DRAFT_TEMP_SCALE=0.8/0.9`** at production sampling. One env var; retests a "no gain" never measured at this k, and prose is where a sharper draft should help most.
5. **Per-class `spec_decode_num_accepted_tokens_per_pos_total`** (`bench/labd_accept.py`) to split prose's acceptance from its step time and confirm the ~0.68 plateau.
6. **Only if #1 shows a real prose vocab gap:** recount the list with a prose-weighted corpus at the **same 40k**. Reslice is CPU-seconds; the cost is 2.2 h GPU for `gen_data.py`.

Do **not** build a bigger vocab (49k measured no faster, ~54k emitted types) and do not revisit MTP fine-tuning (measured flat). The repo's own data closes both.

All repo tok/s figures are single-stream 3090 fast-variant; gotcha 14 and README:69 warn of ±3–5% run-to-run, so treat anything under ~5% as noise and repeat it.

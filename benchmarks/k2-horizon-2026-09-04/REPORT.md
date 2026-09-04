# K2 Horizon 7B on the RTX 3090 (2026-09-03/04)

IFM's K2-Horizon-7B (Apache 2.0, released 2026-09-03; dense 36 layers, GQA
32/8, 250k vocab, native 512k context, ~9B parameters with embeddings) brought
up as an optional llama-swap backend `k2-horizon-7b` next to the production
`qwen3.8-27b`. No client default changed.

## What had to be built

- llama.cpp upstream had no K2 Horizon support. `patches-v15/0009` is the
  squashed MBZUAI-IFM fork branch `model/K2Horizon` (5 commits) cherry-picked
  onto the v14 base `0f3a71be1`; it composes with the six kernel patches.
- The publisher ships a BF16 GGUF only (18.01 GB, revision `835e1323`, LFS SHA
  verified). Local quants: Q8_0 (served), Q6_K, Q5_K_M, Q4_K_M, the last three
  with an imatrix from `calib-code-chat.txt` (2.5 MB: agentic coding chats +
  C++/Python/YAML/shell/Markdown). Recipe: `make-quants.sh`.
- The chat template opens a different thinking tag per `reasoning_effort`
  (`<ifm|think>` / `<ifm|think_fast>` / `<ifm|think_faster>`) and the model
  closes `<ifm|think_fast>` with `</ifm|think>`. llama.cpp's differential
  auto-parser only learns the default tag, so medium/low leaked reasoning into
  `content` (v15) or swallowed the answer (v16). `patches-v15/0010` (rev 2,
  image v17) adopts the request's opener and accepts both closers. Verified
  live: high/medium/low/thinking-off all return the answer in `content` and the
  thinking in `reasoning_content`; tool calls parse in all three template
  formats (xml default, json, xml_typed) at every effort.

## Serving profile (k2-horizon-7b)

Q8_0, ctx 131072, q8_0 KV (74 KB/token -> ~9.7 GB at 131k), `--spec-type
ngram-mod --spec-ngram-mod-n-match 32`, no drafter (none exists for this
model yet), `reasoning_effort` default high per the publisher, temp 1.0 /
top_p 0.95 / top_k 0. Resident VRAM 18,858 MiB.

## Speed and quality (same harness as the 2026-09-02 engine trial)

Medium effort, `results-medium-v17.jsonl`. Qwen column = production
`qwen3.8-27b` (UD-Q4_K_XL + DFlash2-Q8 n7 + ngram-mod) from that trial.

| metric | K2 7B Q8_0 | Qwen3.8-27B prod |
|---|---|---|
| quality battery | 4/4 | 4/4 |
| decode tok/s (fresh server, honest) | 73 | 122 |
| decode tok/s (repeated prompt, ngram-inflated) | 102 | 383 |
| sustained 6k-token generation tok/s | 70 | 113 |
| cold prefill tok/s (~14.2k tokens) | 3381 | 1175 |
| edit session 8 turns / 0 preamble | 212 | 240 |
| edit session 6 turns / 50k preamble | 187 | 210 |
| edit session 20 turns / 20k preamble | 287 | 210 |
| 4-way concurrent aggregate tok/s | 71 | 160 |

Raw decode without a drafter is ~67-73 tok/s at every effort level (direct
probes with nonced prompts, `results-high-v15.jsonl` and the battery warm-up
run). The earlier 102 tok/s decode median is the documented ngram-mod
self-memorisation artefact on repeated identical prompts.

Caveat observed at high effort on v15 (`results-high-v15.jsonl`, sustained):
windows 1-9 ran at 60-66 tok/s, windows 10-12 jumped to 200-230 tok/s with
`finish_reason=length` at 6000 tokens, which is the signature of ngram-mod
copying a repeating reasoning loop. At medium effort the same test is flat
(74 -> 67 tok/s) with no jump. Three nonced 4000-token high-effort generations
showed no loop (duplicate-line ratio 0.14). Treat long high-effort
generations as something to watch, not a confirmed defect.

## Quant ladder (llama-bench pp4096/tg256, and KLD vs BF16 on 48x2048 tokens of held-out code)

| quant | GiB | prefill tok/s | decode tok/s | PPL ratio vs BF16 | same top-1 |
|---|---|---|---|---|---|
| BF16 | 16.8 | - | - | 1.000 | 100% |
| Q8_0 | 8.9 | 4831 | 91 | 1.0012 | 97.5% |
| Q6_K | 6.9 | 4198 | 101 | 1.0093 | 94.6% |
| Q5_K_M | 6.0 | 4509 | 118 | 1.0337 (KLD 0.045) | 93.1% |
| Q4_K_M | 5.2 | 4637 | 131 | 1.0280 (KLD 0.042) | 91.9% |

Held-out text: `heldout-code.txt` (syv-ai Python/Markdown + MIGRATION_LOG),
disjoint from the imatrix corpus. Base logits `heldout-bf16.kld` (24.6 GB) are
kept under `k2-horizon/local/` on Buttercup.

## Where the speed is

The 27B's 122 tok/s is ~55 raw x a 2.2x drafter stack. K2 has no drafter and
a 250k-row lm_head (2 GB at Q8). Levers, in order of payoff:

1. **Drafter.** A DFlash2/EAGLE-style block drafter trained for K2 Horizon 7B
   would apply the same 2x the Qwen stack gets. Nothing published yet;
   IFM's Uno adapter (1.4 GB LoRA r=128) is the publisher's own speedup but
   runs only in their nano-vllm-uno runtime today.
2. **Smaller quant.** Q4_K_M is +44% decode over Q8_0 (KLD 0.042, top-1 91.9%
   vs 97.5%) and is served as the on-demand `k2-horizon-7b-fast` backend;
   Q5_K_M lands in the same quality band 11% slower, so it is not served.
   Live through llama-swap: `k2-horizon-7b-fast` answered `K2_FAST_OK` at
   medium effort and decoded a nonced 509-token low-effort coding answer at
   98 tok/s (Q8_0 profile: 67-73), 15,548 MiB resident at 131k.
3. **Reasoning effort.** medium/low are the daily-driver settings for the
   Pi/Prime agentic loop; high spends 4000+ tokens before answering.

## Files

- `results-high-v15.jsonl`, `results-medium-v17.jsonl`, `battery-medium-v17.log`
- `llama-bench-quants.md`, `kld-vs-bf16.txt`
- `qualify-k2.py` (probe script), `make-quants.sh` (imatrix + quant recipe)

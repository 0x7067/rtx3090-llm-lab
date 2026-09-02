# Engine trial, 2026-09-02: llama.cpp master vs vLLM v10 vs SGLang

## The question

By 2026-09-01 the home deployment had been on vLLM for twelve days (see the
root README's "Current deployment" history and
[`docs/vllm-companion.md`](../../docs/vllm-companion.md)). Upstream llama.cpp
had meanwhile landed DFlash2 (#27342), DSpark (#25173), EAGLE-3 for
qwen3.5/3.6 (#24593), the hybrid checkpoint-restore fixes, and backend draft
sampling — enough new speculative-decoding machinery that the original
patch/vLLM comparison was stale. The question this trial answers: **on the
production workload, does llama.cpp master (rebased patches, new drafters) or
SGLang beat the deployed vLLM v10 profile, and if so, with which drafter?**

## The arms

One 94-minute production window (15:03-16:37 UTC-3, 2026-09-02), production
scaled to 0 (Flux suspended) and restored after. Same harness for every arm —
[`bench/bench.py`](bench/bench.py), reasoning effort medium, temp 0 — running
decode (1k), sustained (6k-token generation), cold prefill (~14.8k), agentic
file-edit sessions (8 shallow turns, 6 turns after a 50k preamble, 20 turns
after a 20k preamble), 4-way concurrency, and a 4-task quality battery:

| Arm | Engine | Drafter |
|---|---|---|
| `vllm-prod-v10` | vLLM v10 (baseline) | 3-token MTP |
| `llama-mtp-only` | llama.cpp trial image | MTP only |
| `llama-mtp-ngram` | llama.cpp trial image | MTP + ngram-mod |
| `llama-mtp-ngram-ctrl` | llama.cpp trial image | MTP + ngram-mod (same-block control) |
| `llama-dflash2q8-n4` | llama.cpp trial image | DFlash2-Q8, n=4 |
| `llama-dflash2q8-n7-ngram` | llama.cpp trial image | DFlash2-Q8, n=7 + ngram-mod |
| `llama-dspark-q8` | llama.cpp trial image | DSpark-Q8 |
| `sglang-nospec` | SGLang (RedHatAI INT4, nightly 2026-08-28) | none — native MTP failed to start (below) |

Trial image `llama:trial-2026-09-02` = llama.cpp master `0f3a71be1` + the six
patches in [`llamacpp/patches-rebased/`](llamacpp/patches-rebased/) (0001 and
0006 dropped — superseded by upstream backend draft sampling; see
[`llamacpp/patches-rebased/NOTES.md`](llamacpp/patches-rebased/NOTES.md) and
the fuller [`../../patches-v14/REBASE-2026-09-02.md`](../../patches-v14/REBASE-2026-09-02.md),
the same rebase notes as the production patch set). It passed
`test-backend-ops` for `FLASH_ATTN_EXT` and `MUL_MAT` before serving traffic.
Not imported into k3s during the trial; no production change happened until
the separate promotion decision below. SGLang used the hybrid MTP+ngram
overlay from PR #36783 ([`sglang/`](sglang/)), but its native MTP arm never
ran: weight loading left no GPU memory for the KV cache under
`--mem-fraction-static=0.92`, so only the no-speculation arm served.

## Headline table

Copied verbatim from [`REPORT.md`](REPORT.md) (see that file for the prefill,
concurrency, sustained, and quality tables too):

### decode (single-request, streaming)

| tag | runs | median tok/s | median ttft (s) |
|---|---|---|---|
| llama-dflash2q8-n4 | 5 | 99.61 | 1.16 |
| llama-dflash2q8-n7-ngram | 5 | 291.54 | 1.16 |
| llama-dspark-q8 | 5 | 100.17 | 1.15 |
| llama-mtp-ngram | 5 | 383.03 | 1.09 |
| llama-mtp-ngram-ctrl | 5 | 306.97 | 1.10 |
| llama-mtp-only | 5 | 105.37 | 1.10 |
| sglang-nospec | 3 | 50.18 | 0.79 |
| vllm-prod-v10 | 5 | 128.62 | 0.77 |

### session (cumulative multi-turn file-editing)

| tag | configs (turns/preamble-tokens) | median cumulative tok/s | median applied rate |
|---|---|---|---|
| llama-dflash2q8-n4 | 20t/20000p, 6t/50000p, 8t/0p | 99.29 | 1.00 |
| llama-dflash2q8-n7-ngram | 20t/20000p, 6t/50000p, 8t/0p | 210.39 | 1.00 |
| llama-dspark-q8 | 20t/20000p, 6t/50000p, 8t/0p | 107.66 | 1.00 |
| llama-mtp-ngram | 20t/20000p, 6t/50000p, 8t/0p | 219.08 | 1.00 |
| llama-mtp-ngram-ctrl | 20t/20000p, 6t/50000p, 8t/0p | 219.16 | 1.00 |
| llama-mtp-only | 20t/20000p, 6t/50000p, 8t/0p | 115.53 | 1.00 |
| sglang-nospec | 6t/0p | 49.42 | 1.00 |
| vllm-prod-v10 | 20t/20000p, 6t/50000p, 8t/0p | 113.69 | 1.00 |

This is the workload the promotion decision was made on: for the agentic edit
sessions, llama.cpp with `draft-mtp,ngram-mod` is **~2x vLLM at every depth**
(230/194/219 tok/s vs 127/108/114 — the session table above reports cumulative
rate at three preamble depths, not identical to the per-turn figures quoted in
`MIGRATION_LOG.md`, but the same ordering and the same ~2x gap), reproduced by
the same-block `-ctrl` arm within 1%. vLLM keeps single-request decode (129 vs
105-108), prefill (1,215 vs 1,175 tok/s), and 4-way concurrency (262 vs 160).

## Methodology caveat: ngram-mod self-memorization

The decode-table numbers above for `llama-mtp-ngram` (383.03),
`llama-mtp-ngram-ctrl` (306.97), and `llama-dflash2q8-n7-ngram` (291.54) are
**inflated and not the honest per-engine speed**. The single-request decode
probe in this harness sends the same prompt across its five repeated runs.
`ngram-mod` builds a host-RAM table of recently-seen token chains and drafts
verbatim copy rounds whenever the last N tokens exactly match a stored chain
(`--spec-ngram-mod-n-match 32`) — so by the second and later runs against an
identical prompt, the ngram drafter is substantially reciting its own prior
output rather than drafting fresh continuations. That is a real technique
(see the "Stacked n-gram drafting" section of the root README, where it is
the intended behavior on repeated file content in agentic sessions), but it
is the wrong thing to measure as "decode speed on a fresh prompt."

**The honest number is the fresh-server warm-up run** — the first request
each arm serves after the server boots, before any repeated-prompt
self-memorization can accumulate:

| mtp-ngram | dflash2-n4 | dflash2-n7-ngram | dspark | mtp-only | ctrl |
|---|---|---|---|---|---|
| 108 | 99 | 122 | 102 | 101 | 103 |

On that clean number, DFlash2-Q8 n7 + ngram-mod (122 tok/s) is the fastest
llama.cpp single-request decode arm, consistent with it also winning the
sustained (113) and session (240/210/210) tables in `REPORT.md`; the
ngram-only MTP arms (108/103) are not meaningfully faster than plain MTP
(101) on a fresh prompt, and their 300-400 tok/s decode-table entries should
not be quoted as a general result. This caveat does not touch the session or
sustained tables' relative ranking, since those workloads genuinely do
re-touch recently-seen content by design.

## Outcome

**llama.cpp v14 was promoted to production on 2026-09-02**, per
[`k8s/MIGRATION_LOG.md`](/data/docker-services/k8s/MIGRATION_LOG.md) in the
docker-services GitOps repo (top two entries as of this writing, quoted here
verbatim for a self-contained record):

> 2026-09-02  llama               cutover      **Promoted llama.cpp (llama-swap, image `llama:cuda-swap-v14`) over vLLM v10 for `qwen3.8-27b`**, on the strength of the engine trial below: ~2x on the agentic edit workload at every depth, quality 4/4. Backend: UD-Q4_K_XL-v3 + mmproj, q4_0 KV, ctx 131072 (was 153600 on the Aug llama-swap config; the Q8 DFlash2 drafter is +0.95 GB over d48k, VRAM 23,600 MiB measured), `--spec-type draft-dflash,ngram-mod --spec-draft-n-max 7 --spec-ngram-mod-n-match 32`, reasoning_effort default medium, env caps `GGML_CUDA_MMVQ_NE11_MAX=3` / `GGML_CUDA_MMQ_SMALLN=3`. Image = master `0f3a71be1` + six rebased patches (0001/0006 dropped as superseded by upstream backend draft sampling); `image/patches/REBASE-2026-09-02.md` has the per-patch notes. Manifest is the pre-2026-08-20 llama-swap deployment plus the gpu-free-gate init container. The `qwen3.8-27b-262k` KVarN entry is removed from the ConfigMap (v14 has no BeeLlama binary). `llama-cache-canary` scaled to 0 (vLLM-only metric). `fetch-models.sh` gained the z-lab DFlash2 Q8_0 GGUF. Clients: Pi and Prime on home-server, mac-studio and the Ravn MacBook already pointed at `http://<home-server-overlay-ip>/v1` / `qwen3.8-27b`; their `contextWindow` was lowered 140000 -> 131072 to match the server. Rollback: revert this commit (v10 image still in containerd, previous deployment.yaml = vLLM).
>
> 2026-09-02  llama               tested       **Engine trial in a 94-minute production window (15:03–16:37 UTC-3): llama.cpp master + rebased patches, four drafters, and SGLang, against the v10 vLLM baseline.** Production scaled to 0 (Flux suspended), restored 16:37; verified 1/1, same 201,630-token KV pool, serving. Same harness for every arm (`engine-trial-2026-09-02/bench/bench.py`, reasoning=medium, temp 0): decode 1k, sustained 6k-token generation, cold prefill ~14.8k, agentic file-edit sessions (8 turns shallow, 6 turns after 50k preamble, 20 turns after 20k), 4-way concurrency, 4-task quality. **Result: for the agentic edit workload llama.cpp with `draft-mtp,ngram-mod` is ~2x vLLM at every depth** (230/194/219 tok/s vs 127/108/114), reproduced by a same-block control within 1%. vLLM keeps single-request decode (129 vs 105–108), prefill (1,215 vs 1,175) and concurrency (262 vs 160 at `--parallel 1`). **DFlash2-Q8 n7 + ngram-mod is the best llama.cpp config** (122 decode, 113 sustained, 240/210/210 sessions); DFlash2 alone at n4 is the slowest drafter (99); DSpark-Q8 n5 loses to MTP (102 decode, 79 sustained). No MTP or DFlash2 decay over 20 turns or 6k tokens on either engine. Quality 4/4 everywhere. **SGLang** (RedHatAI INT4, nightly 2026-08-28 image, hybrid MTP+ngram PR #36783 overlaid) serves without speculation at 50 tok/s; native MTP died in memory-pool sizing (`Loaded weights leave no GPU memory for the KV cache under --mem-fraction-static=0.92`), so the hybrid arm never ran. Trial llama.cpp image `llama:trial-2026-09-02` = master `0f3a71be1` + 6 rebased patches (0001/0006 dropped as superseded by upstream backend draft sampling; upstream already has the hybrid checkpoint-restore fix); passed `test-backend-ops` FLASH_ATTN_EXT + MUL_MAT. Not imported into k3s; no production change. Caveat: the ngram arms' repeated-identical-prompt decode medians (300–400) are the documented self-memorization artifact and were discarded; the clean number is the fresh-server run. Artifacts: `/data/buttercup_6tb/k3s/vllm-trial/engine-trial-2026-09-02/` (REPORT.md, results.jsonl, window.log, rebased patches, SGLang scripts and logs).

The production patch set actually shipped
([`../../patches-v14/`](../../patches-v14/)) is the same rebase as this
trial's `llamacpp/patches-rebased/`, byte-identical modulo the directory it
lives in.

## Layout

```
REPORT.md                      full benchmark tables (decode, prefill, session, concurrent, sustained, quality)
bench/bench.py                 the harness: all six workload types, shared across every arm
bench/run-arm.sh               drives one arm through the harness and appends to results.jsonl
bench/results.jsonl            raw per-run measurements
run-window.sh                  orchestrates the whole 94-minute window: takedown, validate, per-arm runs, restore
window.log                     full run log of the trial window
baseline.log                   vLLM v10 baseline capture, same harness
llamacpp/patches-rebased/      the six patches rebased onto 0f3a71be1 for the trial image (== patches-v14/)
llamacpp/patches-rebased/NOTES.md   per-patch rebase notes written during the trial
llamacpp/Dockerfile.trial      trial image build (llama:trial-2026-09-02)
sglang/                        SGLang hybrid-MTP build script, run scripts, VRAM budget check, logs, NOTES.md
```

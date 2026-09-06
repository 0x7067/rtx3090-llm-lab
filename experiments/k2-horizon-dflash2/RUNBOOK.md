# K2 Horizon 7B DFlash2 drafter: training runbook

## Resume audit, September 5

The September 4 Claude session `e39adc1c-0fa6-4d11-85c2-93a34c34d655`
stopped during regeneration. At this audit, no training or capture process
is running, the old feature directory is empty, and the v18 local API is
healthy. `regen-output.jsonl` contains 4,427 distinct, parseable rows from
the 10,082-row input, but only **898** have a completed answer or tool call;
464 are truncated and 3,065 completed with neither. Of 3,345 public-prompt
rows, only 12 pass the usable-answer filter.

The installed SGLang K2 parser expects `</ifm|think_fast>` at medium and
`</ifm|think_faster>` at low. The 7B can emit `</ifm|think>` instead: a
non-streaming parser regression reproduces it swallowing the final answer.
The old regeneration script then removed that closing tag, so the affected
saved answers cannot be split reliably. Do not try to recover them using
prose heuristics. `sglang-k2-nonstream-reasoning.patch` is applied to the
isolated SGLang checkout and accepts this delimiter for non-streaming
regeneration; it does not change streaming parsing. The new tests execute
the actual installed parser, and failed before this patch.

`resume-regeneration.sh` preserves the original file and seeds
`regen-output-v2.jsonl` with the 898 usable and 464 truncated records. It
must generate the 5,655 missing plus 3,065 affected rows (8,720 total).
At the previous 10–15 rows/minute, allow roughly 10–15 hours for this stage;
the corrected live parser and throughput still need a GPU smoke test.
The job pauses Flux apps and llama, starts the isolated SGLang server,
checks a real medium-effort answer, resumes regeneration, filters the new
capture input, and restores the local API on success, failure, or TERM.
The user authorized taking the local-model server offline to continue on
September 5. The `--prepare-only` pass completed: the new
output contains 1,362 unique rows, all five local regression tests and nine
existing SGLang K2 parser tests pass, and the restart script passes
ShellCheck. The job is managed by the user systemd service
`k2-regeneration-20260905`. Startup exposed that user services inherit
neither the interactive `KUBECONFIG` nor its local-bin PATH; the launcher
now supplies both and checks required commands before pausing the API.
At 21:44 UTC, Flux apps was suspended, llama was scaled to zero, and
SGLang began loading K2 on the 3090. Inspect the service journal and
`resume-2026-09-05/server.log` under the training directory for live state.
The live medium-effort `2+2` smoke test passed at 21:47 UTC. By 21:48,
48 new responses had been persisted, all 48 with usable answers or tool
calls and no request errors; the regenerated output had 1,410 rows total.
This confirms the corrected regeneration stage is running, not that draft
training or deployment has completed. The job restores the API when this
regeneration/filtering stage ends; the existing maintenance authorization
also covers the subsequent capture and training work.

The user requested notifications in this same chat. At 22:05 UTC,
`k2-notify-20260905.service` started watching this regeneration invocation.
It polls every 30 seconds, checks invocation-scoped journal completion
(including restoration failures), and uses the installed `codex queue`
command to notify the originating thread. It also queues the existing
request to continue capture/training, so the completion callback is a
resume point, not a claim that all fine-tuning is finished. Its state is
`resume-2026-09-05/chat-notification.json` in the training directory; it
records the queue receipt and exits after notification. Failed queue
attempts retry through systemd. Six notification-state tests pass, the
watcher is confirmed active on the correct invocation, and a separate
delivery-test message was accepted by Codex for the originating thread.
This setup posts to the chat; it does not configure OS/mobile push settings.

The launch command, run on the host so it survives a client disconnect:

```bash
systemd-run --user --unit=k2-regeneration-20260905 \
  --property=KillMode=mixed --property=TimeoutStopSec=20min \
  bash /data/docker-services/rtx3090-llm-lab/experiments/k2-horizon-dflash2/resume-regeneration.sh
journalctl --user -u k2-regeneration-20260905 -f
```

Before capture, correct the legacy capture script/config: they still use
the discarded `k2-horizon-nothink` setup and original-answer dataset.
Use regenerated data, supervise only the last K2 assistant turn, preserve
the medium-effort wire format, and measure feature-store space first.
In a fixed-seed sample of 200 usable rows (198 agent contexts, two public),
last-turn masking at 2,048 tokens loses all supervision on 162 agent rows;
8,192 loses it on one. The old 2,048-token capture cannot simply be resumed.
There is about 856 GiB free on Buttercup. Capture, draft training, export,
and live acceptance/speed measurement remain unfinished; no new drafter
has been trained or deployed.

Local regression command:

```bash
TORCHINDUCTOR_CACHE_DIR=/tmp/k2-resume-inductor HF_HUB_OFFLINE=1 \
  /data/buttercup_6tb/specforge-work/venv/bin/python -m unittest discover \
  -s experiments/k2-horizon-dflash2 -p test_resume.py -v
```

Goal: a `draft-dflash` GGUF for `IFM/K2-Horizon-7B` so llama.cpp on the 3090
gets the same 1.6-2.2x speculative speedup the Qwen3.8-27B stack has
(measured raw decode today: ~70 tok/s Q8_0, ~98 tok/s Q4_K_M; target 110-150).

Everything on the llama.cpp side already works: our v15+ image is master
`0f3a71be1` (DFlash2 support #27342 included) plus the K2 Horizon arch; the
DFlash converter resolves the target class by architecture name and reuses
its vocab handler, and the runtime shares the target's embeddings/lm_head.
The only gap was the SGLang capture hook, provided here as a patch.

## Regeneration completed, September 6

The September 5 job finished its generation stage at 11:08 UTC after 13h24m:
8,717 new rows at a steady 10.9 rows/min, 3 errors (two responses the parser
read as having neither answer nor tool call, one HTTP 400). It then **exited 1
purely because `err > 0`**, which skipped `build_capture_set.py` and reported
the whole run as failed; the completion notification said `failed (exit 1)`.
The pause/restore harness worked correctly — Flux apps resumed, llama rolled
back to one healthy replica, GPU at 9 MiB — and no data was lost. The generated
data is intact and was audited: `regen-output-v2.jsonl` holds **10,079 of the
10,082 frozen input rows**, no duplicates, nothing outside the input, with
`finish_reason` 7,834 stop / 943 tool_calls / 1,302 length. The three missing
IDs are `69693a03-439`, `c913d1f1-13`, `pub-33c79514042d733fe78a4eda3ad43b08`.

`regenerate_sessions.py` now takes `--max-error-rate` (default 0.01) and fails
only above that fraction, so a handful of unparseable responses no longer
discards the downstream stage; a total failure still aborts. Both behaviours
have tests.

`build_capture_set.py` was then run by hand and produced the capture input:
**8,777 rows kept**, 1,302 truncated dropped, 0 empty, at
`cache/dataset/k2-eagle3-regenerated.jsonl` (75 MB).

### Capture length and feature-store size, measured

Measured with the **actual `ThinkingParser` and the registered
`k2-horizon-thinking` template** over a fixed-seed 500-row sample (not an
estimate from raw text): rendered length mean 2,126 tokens, median 1,850,
p90 4,000, max 9,136. Extrapolated to all 8,777 rows at the observed
32.8 KB/token:

| `max_length` | rows losing all supervision | feature store |
|---|---|---|
| 2,048 | 55/500 (11%) | 427 GB |
| 4,096 | 35/500 (7%) | 565 GB |
| 8,192 | 0/500 | 611 GB |
| 16,384 | 0/500 | 612 GB |

**8,192 is the setting**: the smallest cap that leaves every sampled row with
supervision, and only 8% more disk than 4,096, which still blanks 7% of rows.
Note 16,384 buys nothing — the distribution is exhausted by 8,192.

The extrapolation was then replaced with an **exact CPU-only count**: building
the full processed dataset at these settings gives **8,777 rows and 18,417,412
tokens, 604 GB**, with only **5 rows** losing all supervision and length mean
2,098 / median 1,838 / p90 3,934 / p99 7,255 / max 8,192. Buttercup had 855 GB
free, so this fits with about 250 GB to spare. That build is cached at
`cache/processed_dataset/48756f9976367db1d494ac5be86d7c7d*` under the same key
the capture script computes, so the real run does not re-tokenize.

### Smoke run before committing the disk

A 32-sample run through `with-local-api-paused.sh` validated the new flags end
to end and **confirmed 32,784 bytes/token exactly** — the 32.8 KB/token figure
used for every estimate above is measured, not assumed. Two things to know
about it:

- `--num-samples` selects `range(n)` from the raw file **before** the shuffle
  inside `build_eagle3_dataset` (EAGLE-3 registers no `loss_mask_filter`, so
  the early-select branch is taken). `build_regen_prompts.py` puts the 1,082
  mined agent contexts first, so a small `--num-samples` run captures only the
  longest, multi-turn rows: the smoke sample averaged 4,752 tokens against the
  full set's 2,098. Useful as a worst case, useless as a sample.
- One `CUDACachingAllocator` OOM retry appeared (398 MB request against
  266 MB free) and recovered. **That retry was the warning, and batch 2 did not
  survive the real run** — see below.

### batch 2 OOMed; capture runs at batch 1

The first full attempt died 54 rows in with a fatal
`torch.OutOfMemoryError: Tried to allocate 384.00 MiB. GPU 0 has a total
capacity of 23.55 GiB of which 246.12 MiB is free`. The arithmetic is exact:
the EAGLE-3 aux hidden state is 3 x 4,096 bf16 per token, so a batch of
2 x 8,192 tokens needs 402,653,184 bytes in one block — the number in the
allocator warning. Static occupancy was already 20.5 GB (18.1 GB bf16 weights
plus a 16,384-token KV pool at 144 KiB/token), leaving under 3 GB for
activations, and 1.23 GB of that was reserved-but-unallocated.

Capture now runs at **batch 1** with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: the KV pool halves to
1.2 GB and the aux block to 192 MiB, giving roughly 4.2 GB of activation
headroom. The throughput cost is small because this workload is prefill-bound
on sequences averaging 2,098 tokens, not decode-bound.

**Capture resumes.** `prepare_hidden_states.py` checks whether each sample
index already has an output file and skips it (the `skipped=` counter in the
progress bar), and the dataset shuffle seed is fixed, so index-to-sample
mapping is stable across batch sizes. The 54 rows written by the failed batch-2
attempt were kept, and a re-run after any interruption picks up where it left
off rather than rewriting 604 GB.

The wrapper restored the API correctly on this failure: Flux resumed and llama
rolled back to one replica within 11 seconds of the crash.

### `prepare_hidden_states.py` had no way to supervise only the last turn

`build_eagle3_dataset` and `preprocess_conversations` both accept
`train_only_last_turn`, but the capture entrypoint never exposed or passed it,
so capture defaulted to supervising **every** assistant turn. In the mined
agent contexts the history turns were written by Claude, not K2, so that is the
same defect that made the first training pass worthless — measured at 1,656
supervised tokens/row across all turns versus 1,461 for the last turn alone.
`specforge-sglang-main-compat.patch` now adds a `--train-only-last-turn` flag
and threads it into the dataset **cache key**, so an earlier cache cannot be
silently reused with the wrong masking.

`capture-eagle3.sh` was updated accordingly: regenerated dataset,
`k2-horizon-thinking`, `--max-length 8192`, `--train-only-last-turn`, output at
`cache/hidden_states/k2-7b-eagle3-regen`, and `triton` as the default attention
backend (what the successful September 4 run actually used; the old
`flashinfer` default never worked here). **Batch size dropped 4 -> 2**: the
script sizes the SGLang KV pool at `batch_size x max_length`, and K2 costs
144 KiB/token of KV (36 layers x 8 KV heads x 128 head-dim x 2 x bf16), so
batch 2 at 8,192 is 2.4 GB beside the 18 GB bf16 target — batch 4 would need
4.8 GB and not fit the 24 GB card.

`with-local-api-paused.sh` extracts the Flux-suspend / scale-to-zero /
GPU-free-check / restore-on-any-exit sequence that `resume-regeneration.sh`
proved over 13 hours, so the capture and training jobs reuse it and share its
lock instead of duplicating it.

Launch, on the host so it survives a client disconnect (`NUM_SAMPLES` and
`OUTPUT_DIR` are the smoke-run knobs):

```bash
E=/data/docker-services/rtx3090-llm-lab/experiments/k2-horizon-dflash2
systemd-run --user --unit=k2-capture-20260906 \
  --property=KillMode=mixed --property=TimeoutStopSec=20min \
  bash "$E/with-local-api-paused.sh" bash "$E/capture-eagle3.sh"
journalctl --user -u k2-capture-20260906 -f
```

The wrapper restores the API on success, failure, or TERM, so an abandoned
session cannot leave llama scaled to zero. tqdm writes progress with carriage
returns and no newlines, so **the journal looks frozen while the run is
healthy** — check `find <output-path> -name '*.ckpt' | wc -l` instead.

## Prerequisites (rented box, 1x H100 80 GB is enough; 2x makes it simpler)

- Python 3.10+, CUDA 12.8 driver.
- SGLang **from source at main** (K2 Horizon support merged 2026-09-03 via
  sgl-project/sglang#37654; 0.5.18 predates it), with
  `sglang-k2-capture.patch` applied (adds `set_dflash_layers_to_capture` /
  `set_eagle3_layers_to_capture` and aux hidden-state collection to
  `XllmForCausalLM`, mirroring `qwen3.py`). Note SpecForge pins
  `sglang==0.5.18`; install SpecForge with `--no-deps` on top of the newer
  SGLang and resolve any API drift in `specforge/offline_capture` /
  runtime capture code by hand. **This is the main integration risk.**
- SpecForge from source (`sgl-project/SpecForge`), `pip install -e . --no-deps`.
- `mooncake` transfer engine for the disaggregated mode (or use the
  colocated example layout if the SpecForge version offers one for dflash).
- HF cache with `IFM/K2-Horizon-7B` (18 GB) at revision
  `69ada542b68fe13d767479db2ab9421baff88681` (the SGLang cookbook pin).

## Step 1: prompts (CPU)

```bash
python scripts/prepare_data.py --dataset perfectblend --output-path cache/dataset   # or sharegpt + opc
python scripts/prepare_data.py --dataset opc --opc-subset <coding subset> --output-path cache/dataset
```
Aim for 100-200k conversations, code-heavy (the drafter must learn the
target's coding+reasoning distribution). 30-50k is the cheap EAGLE-3-scale
floor; DFlash2 recipes in the SpecForge repo use 100k+ samples x 6 epochs.

## Step 2: regenerate answers with the target (GPU, ~6-12 H100-hours for 150k)

```bash
python -m sglang.launch_server --model-path IFM/K2-Horizon-7B --trust-remote-code \
  --dtype bfloat16 --context-length 32768 --cuda-graph-max-bs 128 --mem-fraction-static 0.85 \
  --reasoning-parser k2_horizon --port 30000
python scripts/regenerate_train_data.py --model IFM/K2-Horizon-7B --server-address localhost:30000 \
  --concurrency 128 --max-tokens 16384 --temperature 1.0 --top-p 0.95 --reasoning save \
  --input-file-path cache/dataset/perfectblend_train.jsonl --output-file-path cache/dataset/k2-7b-regen.jsonl
python scripts/validate_regenerated_data.py --input cache/dataset/k2-7b-regen.jsonl
python scripts/expand_reasoning_conversations.py --input cache/dataset/k2-7b-regen.jsonl --output cache/dataset/k2-7b-regen.jsonl
```
Pass `chat_template_kwargs {"reasoning_effort":"high"}` if the script exposes
it (check `--extra-body`/`--chat-template-kwargs` flags in the current
version); the template registered in `specforge_k2_template.py` assumes the
assistant turn opens with `<ifm|think>\n`. Cap `--max-tokens` at 16k: high
effort spends 4k+ tokens thinking on simple prompts.

## Step 3: train (GPU, ~10-30 H100-hours)

```bash
export PYTHONPATH=$PWD/experiments/k2-horizon-dflash2:$PYTHONPATH   # for specforge_k2_template
python -c "import specforge_k2_template"                             # sanity
specforge train --config train-k2-7b-dflash2-online.yaml
```
Watch `selector_coverage` and acceptance-length metrics in TensorBoard.
Stop when the eval acceptance plateaus (Qwen3.6-27B DFlash2 recipe: ~6
epochs over 100k samples).

## Step 4: export and convert

```bash
specforge export --to hf --checkpoint outputs/k2-horizon-7b-dflash2/<step> --output-dir exports/k2-7b-dflash2-hf
# on the 3090 box, inside the scratch llama.cpp checkout (k2 branch, has conversion/k2_horizon.py):
python convert_hf_to_gguf.py exports/k2-7b-dflash2-hf --target-model-dir <hf cache>/IFM/K2-Horizon-7B \
  --outtype q8_0 --outfile /data/buttercup_6tb/k3s/llama-models/k2-horizon/local/K2-Horizon-7B-DFlash2-Q8_0.gguf
```
The converter maps `target_layer_ids` to llama.cpp's `dflash.target_layers`
(+1 offset, same convention SGLang uses), writes `mask_token_id`, block size,
conv/selector metadata (DFlash2 detected at runtime via `selector_top_k > 0`).

## Step 5: serve and measure

Add to the `k2-horizon-7b` stanza:
```
--model-draft /models/k2-horizon/local/K2-Horizon-7B-DFlash2-Q8_0.gguf
--spec-type draft-dflash,ngram-mod --spec-draft-n-max 7 -ngld 99
--cache-type-k-draft q8_0 --cache-type-v-draft q8_0
```
VRAM: the draft is ~1.1 B params (5 layers, no embeddings of its own) ≈ 1.2
GB at Q8 plus 20 KB/token KV (2.6 GB at 131k q8_0). Q8_0 target + 131k q8_0
KV + draft ≈ 22.7 GB: tight; drop the target KV to q4_0 or ctx to 100k if
the load fails. Then run the engine-trial harness
(`benchmarks/engine-trial-2026-09-02/bench/run-arm.sh`) with
`QWEN_MODEL=k2-horizon-7b` and compare with
`benchmarks/k2-horizon-2026-09-04/results-medium-v17.jsonl`.

## Cheap fallback: EAGLE-3

Same pipeline with `training.strategy: eagle3`, a `configs/*-eagle3.json`
draft (copy `configs/qwen3-8b-eagle3.json`, set hidden 4096, vocab 250624,
target layers from the 36-layer default `[2, 18, 33]`), and llama.cpp
`--spec-type draft-eagle3`. Thoughtworks trained a GLM-4.7-Flash EAGLE-3 head
on 54k samples in 1h26m on one H100. Expected 1.5-2x instead of 1.6-2.2x.

## Cost

H100 80 GB on-demand $2-3.5/h (RunPod/Lambda/Vast, 2026): regen 6-12 h +
training 10-30 h -> roughly $50-150 for a 50k-sample run, $150-400 for
150k+. The 3090 cannot train this (target bf16 alone is 18 GB); it could
only do a slow local regen (~1-1.5k tok/s aggregate, ~10 h per 30k samples)
while production is idle.

## Open risks (not yet verified)

1. SpecForge main vs SGLang main API drift (SpecForge pins 0.5.18).
2. SGLang K2 path refuses non-bf16 and quantized weights; fine on H100.
3. `regenerate_train_data.py` may not forward `chat_template_kwargs`; if not,
   the SGLang server default is already `reasoning_effort=high`.
4. Acceptance on reasoning-heavy outputs is lower than on plain chat; the
   Qwen3.8 DFlash2 stack still measured 1.8x in llama.cpp on this box.

## Local run log (2026-09-04, RTX 3090, EAGLE-3 offline)

Environment: `/data/buttercup_6tb/specforge-work/` (venv via uv, Python 3.12,
torch 2.13.0+cu130, SGLang main `2a0602c` from source with
`sglang-k2-capture.patch`, SpecForge main `e606d40` installed `--no-deps` with
`specforge-sglang-main-compat.patch`). Weights: `models/IFM/K2-Horizon-7B`
at main revision `586b03f0` (the cookbook's `69ada542` no longer exists).

Fixes needed to make SpecForge's offline capture run on SGLang main:
1. `runtime_context.publish(server_args, role="scheduler")` before the
   ModelRunner (`assert_published` is new).
2. `server_args.device` defaults to None until later; set `"cuda"`.
3. `CacheInitParams.page_size` must equal the allocator's page size (the
   runner sizes it from the schedule bag, not `server_args.page_size`);
   otherwise prefill hits `alloc_extend` on an unpaged allocator.
4. `require_mlp_sync()` / `require_mlp_tp_gather()` take no arguments now.
5. K2's HF template raises on assistant messages without a thinking field:
   the parser now defaults `reasoning_content=""` (and keeps that key through
   `_sanitize_message`) when the template mentions `ifm|think`.
6. Template registrations: with empty reasoning the template renders
   `<ifm|think>\n</ifm|think>` with NO trailing newline and `<|ifm|im_end|>`
   with none either; the first attempt (`\n` on both) matched nothing and
   produced an all-zero loss mask (symptom: "Total token frequency is zero").
7. CUDA toolchain in the venv: nvcc 13.3 vs runtime headers 13.0, so CCCL
   refuses to compile (`CUDA compiler and CUDA toolkit headers are
   incompatible`). `NVCC_PREPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK`
   plus `CUDA_HOME=<venv>/nvidia/cu13` with `lib64 -> lib` and
   `libcudart.so -> libcudart.so.13` symlinks (the JIT links `-lcudart` from
   `$CUDA_HOME/lib64`). FlashInfer's JIT prefill kernel was avoided by using
   `--sglang-attention-backend triton`; SGLang's own fused-RoPE JIT still
   compiles and works with the flags above.
8. `deep_ep` (sgl-deep-ep) asserts on import without CUDA_HOME; uninstalled
   (dense model, not needed).

Measured: bf16 target resident 19.5 GB at mem-fraction 0.88, batch 4,
ctx 2304. Features: `input_ids`, `loss_mask`, `aux_hidden_state`
(3 x 4096 bf16) and `hidden_state` (4096 bf16) = 32.8 KB/token; the 64-sample
smoke run averaged ~990 tokens/sample (32 MB/sample). Full run: 16,000 of
the 20,000 prepared samples (OPC realuser 8k + OPC largescale 6k +
PerfectBlend 6k, shuffled), ~525 GB, ~3.5 h. Training: `train-eagle3.sh`
(offline colocated, `train-k2-7b-eagle3-offline.yaml`).

Mined agent data for the next round: `extract_claude_sessions.py` turned 44
of the 68 Claude Code transcripts on this host into 1,082 tool-using
contexts (~3.4k tokens each, 3.7M tokens, 105 secret-like strings redacted)
at `sessions/denguinho-server/claude-code/extracted.jsonl`; assistant turns
are to be regenerated with K2 before use.

## Why the first training run was discarded (2026-09-04)

The first capture/train pass used the public datasets' own answers with
thinking disabled. It trained fine (loss 3.08 -> 1.9, position-0 accuracy
0 -> 21% by step 350) but it was teaching the drafter to predict text the
target never emits: K2 in production writes reasoning inside `<ifm|think>`
at medium effort. Acceptance is measured against the target's own output, so
mismatched data caps the achievable speedup no matter how long it trains.
Two secondary findings from that pass:

- `training.compact_teacher: true` is required here. Without it the 250,624-
  vocab teacher logits are materialised in full: 22.6/24 GB with 133
  allocator OOM retries. With it, 22.3 GB and no fatal errors.
- Throughput is ~1 step/s at `batch_size: 1`, so one epoch over 16,000
  samples is ~4.5 h. The upstream 10-epoch recipe would be ~44 h.

## Regeneration pass (the data that is actually worth training on)

1. `serve-k2-sglang.sh` serves the target with SGLang (bf16, triton
   attention, 28,299 tokens of KV at mem-fraction 0.90, both `k2_horizon`
   parsers). llama.cpp's single slot would take a day for this; SGLang
   batches.
2. `build_regen_prompts.py` assembles the input: the 1,082 mined agent
   contexts first (real tool schemas, repo content, multi-turn), then 9,000
   public coding prompts truncated to their first user turn so K2 answers
   from scratch instead of continuing another model's text.
3. `regenerate_sessions.py` calls the server at `--effort medium`
   (production's setting), concurrency 8, and keeps `reasoning_content` and
   `tool_calls` on the assistant turn. Measured 49 rows/min, so ~3.5 h for
   10,082 rows. It strips stray `</ifm|think>` tags: when the model emits no
   reasoning, SGLang's parser hands back a lone closing tag, and the chat
   template adds the tags itself at render time, so leaving them in would
   train the drafter to emit them mid-stream.
4. Capture with `chat_template: k2-horizon-thinking` (not `-nothink`) and
   delete the previous 504 GB feature set first.

### Regeneration throughput and truncation (measured mid-run)

Rates fall as the input shifts from mined contexts to public prompts:
87 rows/min for the first ~200, then 26, then a steady 15. Cause is output
length, not a leak: the mined agent contexts mostly end in a tool call
(mean 436 completion tokens, 2% truncated at the 4,096 cap), while the
public prompts write long reasoning at medium effort (mean 2,001 tokens,
14% truncated). Aggregate ~190 tok/s, which is what bf16 batching gives on
this card with 28k tokens of KV.

`build_capture_set.py` drops truncated rows before capture: a reasoning
block cut mid-thought teaches the drafter to predict a stop the target
would never emit. Expect to keep ~8,800 of 10,082 rows.

Feature-store planning at `max_length: 2048` and 32.8 KB/token: ~677 GB for
the full set. The stale 504 GB from the discarded first pass was deleted to
make room (878 GB free on Buttercup).

### Conversion path de-risked before training finished

`convert_hf_to_gguf.py` routes `LlamaForCausalLMEagle3` through the Llama
converter, whose `set_vocab` points `dir_model` at `--target-model-dir` and
runs the generic BPE path (unlike the DFlash converter, it does not delegate
to the target's own model class). That path identifies the pre-tokenizer by
a sha256 over a fixed check string, and an unregistered hash aborts the
conversion. `check_tokenizer_hash.py` confirms K2-Horizon-7B hashes to
`a9af07a8...`, the registered `k2-horizon` entry it shares with the 36B, so
the draft will convert. Note the check string is 350 characters; a truncated
copy produces a different hash and a false negative.

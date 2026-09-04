# K2 Horizon 7B DFlash2 drafter: training runbook

Goal: a `draft-dflash` GGUF for `IFM/K2-Horizon-7B` so llama.cpp on the 3090
gets the same 1.6-2.2x speculative speedup the Qwen3.8-27B stack has
(measured raw decode today: ~70 tok/s Q8_0, ~98 tok/s Q4_K_M; target 110-150).

Everything on the llama.cpp side already works: our v15+ image is master
`0f3a71be1` (DFlash2 support #27342 included) plus the K2 Horizon arch; the
DFlash converter resolves the target class by architecture name and reuses
its vocab handler, and the runtime shares the target's embeddings/lm_head.
The only gap was the SGLang capture hook, provided here as a patch.

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

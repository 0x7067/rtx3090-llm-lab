# qwen-image

OpenAI images API for Qwen-Image 2.1, served by llama-swap like the text models:
a request naming one of these models swaps the 3090 to it, and `ttl: 600` hands
the card back after ten idle minutes. No manual GPU window or Flux suspend.
Both are `unlisted`: absent from `/v1/models`, so chat model pickers skip them;
requests by name work as before.

| Model | Checkpoint | Steps | Notes |
| --- | --- | --- | --- |
| `qwen-image-2.1-turbo` (alias `qwen-image`) | Qwen/Qwen-Image-2.1-Turbo @ fc13ba1 | 8 (its own schedule) | default |
| `qwen-image-2.1` | Qwen/Qwen-Image-2.1 @ 790c926 | 40 | slower, for quality |

Both use the prompt enhancers Qwen-Image-2.1-PE-T2I @ f3ed798 (generation) and
PE-I2I @ 72927bc (edits), a 9B VLM that expands the brief and picks the canvas.
The server reads its weights from
`/srv/llama-nvme/production-models/qwen-image`, mounted in the container at
`/models/qwen-image`. The original checkpoints are archived on Buttercup.
During the prior HDD deployment the drive read at 167 MiB/s, and a cold load
could take up to seven minutes. The earlier NVMe measurement was 2.6 GiB/s.

## API

    POST /v1/images/generations  {"model", "prompt", "size": "WxH"|"auto",
                                  "enhance": true, "seed", "steps"}
    POST /v1/images/edits        multipart: image (or image[]; up to 4), prompt,
                                  size, enhance, seed, steps

Responses: `data[0].b64_json` (PNG), `revised_prompt` (the enhancer's prompt, or
yours with `enhance: false`), `seed`, plus `size` and `timings`. `n` must be 1;
only `b64_json`; edits take no mask. Without `size`, the enhancer's aspect ratio
(or the source image's, for edits) picks a native ~2K canvas.

## Measured on the 3090 (2026-11-10, weights on the NVMe, page cache warm)

| Request | Enhance | Render |
| --- | --- | --- |
| Turbo 1024² | off | 38 s |
| Turbo 2048² | off | 58 s |
| Turbo 2048² edit, one image | off | 60 s |
| Turbo 1696×2528 | 76 s | 60 s |

The process holds about 2 GiB of its own memory and at most 20.2 GiB of VRAM
(2048² edit). The enhancer, the 8B text encoder and the 7B DiT each load from
disk straight onto the GPU for their step and are freed after it, so the card is
empty between requests apart from the VAE (~1 GiB). Up to ~18 GiB of resident
memory shows while a model loads: that is the mmapped weight files, page cache
the kernel can reclaim.

Until 2026-11-10 the DiT and text encoder waited in host RAM under model CPU
offload (32 GiB), which took the host out of memory with the rest of its load.

## Deploy

The pod mounts `server.py` and `pe_core.py` from the `llama-qwen-image` ConfigMap
(`../kustomization.yaml`) over the copies baked into the image, so a change to
either file ships with a commit. Only a dependency change needs an image build.

## Build

    docker build -t 127.0.0.1:5000/home-llm-qwen-image:vN .   # this directory
    docker push 127.0.0.1:5000/home-llm-qwen-image:vN

The image is the production llama image plus llama-swap v265 and one uv env
(torch 2.11 cu128, transformers 5.17, diffusers main pinned in pyproject.toml).
`uv run --no-project --with pytest --with httpx --with fastapi --with python-multipart --with pillow --with json-repair -- python -m pytest -q`
runs the GPU-free request tests.

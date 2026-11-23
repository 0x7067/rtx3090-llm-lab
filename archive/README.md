# Archive

Superseded material, kept for provenance and moved here on 2026-11-04. Nothing
in this directory is built or served. Image tags, model paths and GitOps files
that these documents name may no longer exist; see "Reading old documents" in
the [root README](../README.md).

| Path | What it was |
| --- | --- |
| [`llama-cpp-stack-2026-09.md`](llama-cpp-stack-2026-09.md) | The repository README until 2026-11-04: the Qwen3.8 llama.cpp stack, patch table, truncated draft vocabulary, reproduction recipe, n-gram stacking, MMQ results, dead ends, and full credits |
| [`PROMOTE-llama-cpp-vllm-2026-09.md`](PROMOTE-llama-cpp-vllm-2026-09.md) | Build/promote/rollback runbook for llama.cpp v14/v15 and vLLM v9-v12 |
| [`swift-qualification-2026-09-15.md`](swift-qualification-2026-09-15.md) | Swift Q4_K_S production qualification (moved from the GitOps repo) |
| [`patches-v14/`](patches-v14/) | The six-patch llama.cpp set behind `cuda-swap-v14`, plus its rebase notes. Superseded by `llama-cpp-v15/patches-v15/` |
| [`patches-v9-v12-base-4df29be4/`](patches-v9-v12-base-4df29be4/) | The eight-patch set behind images v9-v12 (base `4df29be4f`) |
| [`experiments/`](experiments/) | Measured patches and arms that missed the ship bar (MMQ arm G, GDN fused kernel, MTP suffix lookup v1-v5, vLLM PR 51812, K2 DFlash2, OrcaSAQ latest, EXL16, GSQ llamampere); see its README |
| [`vllm/`](vllm/) | Exported overlays for vLLM images v9-v11, the club-3090 bundle and the v4-era syv-ai patch. The v12-era build trees are in `vllm-qwen38/` |
| [`research/`](research/) | vLLM 0.27.1 notes (2026-08) |
| [`config/`](config/) | llama-swap stanzas for the v18 Qwen3.8 profile and the OBLITERATED variant |
| [`llama-cpp-v15/`](llama-cpp-v15/) | The last llama.cpp build: `Dockerfile`, patches 0002-0011 on `0f3a71be1`, and `scripts/build-llama-image.sh` (recovered from GitOps `f13dde9d^`). Moved from the lab root on 2026-11-23 |
| [`vllm-qwen38/`](vllm-qwen38/) | The `qwen38-27b-3090` vLLM images: the squashed syv-ai stack, the locked `image-build/` base and overlay, and their build scripts. Moved from `vllm/` on 2026-11-23 |
| [`engines/spark/`](engines/spark/) | Spark X2.5 4B qualification (2026-09-10). Moved from `engines/` on 2026-11-23 |

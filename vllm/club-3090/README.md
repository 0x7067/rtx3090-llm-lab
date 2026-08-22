# Club 3090 local-layer bundle

This bundle registers the qualified single-card vLLM profile in a Club 3090
checkout without changing its curated catalog or opening a pull request.

Prerequisites:

- the prepared model directory `Qwen3.8-27B-W4A16-AutoRound-fast`;
- the local image `qwen38-27b-3090:v4`, built as described in
  [`../../docs/vllm-companion.md`](../../docs/vllm-companion.md);
- a Club 3090 checkout current enough to support `scripts/lib/profiles-local`.

Install from the Club 3090 checkout root:

```bash
src=/path/to/qwen38-27b-rtx3090-llamacpp/vllm/club-3090
mkdir -p scripts/lib/profiles-local/models.d
mkdir -p scripts/lib/profiles-local/composes/qwen3.8-27b-3090-fast/vllm/compose/single/compressed-tensors
cp "$src/models.d/qwen3.8-27b-3090-fast.yml" scripts/lib/profiles-local/models.d/
cp "$src/registry.local.json" scripts/lib/profiles-local/
cp "$src/composes/qwen3.8-27b-3090-fast/vllm/compose/single/compressed-tensors/mtp.yml" \
  scripts/lib/profiles-local/composes/qwen3.8-27b-3090-fast/vllm/compose/single/compressed-tensors/
```

Validate:

```bash
bash scripts/diagnose-profile.sh local/qwen38-27b-single-3090-fast
MODEL_DIR=/absolute/path/to/prepared/models \
QWEN38_CACHE_DIR=/absolute/path/to/cache \
docker compose \
  -f scripts/lib/profiles-local/composes/qwen3.8-27b-3090-fast/vllm/compose/single/compressed-tensors/mtp.yml \
  config --quiet
```

The entry is incubating and requires `--force`. Do not launch it until the RTX
3090 is free; it cannot coexist with the active llama service on the same GPU.

```bash
MODEL_DIR=/absolute/path/to/prepared/models \
QWEN38_CACHE_DIR=/absolute/path/to/cache \
bash scripts/launch.sh --variant local/qwen38-27b-single-3090-fast --force
```

Club's local layer is gitignored by design. This directory is the tracked
source of truth for reinstalling the personal entry.

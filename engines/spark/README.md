# Spark X2.5 4B on the RTX 3090

The qualified profile uses the publisher's Q8_0 GGUF, one slot, 131,072
tokens, FP16 K/V and full GPU offload. It runs through llama-swap with a
separate Spark-compatible binary; other models retain the v18 runtime.

## Pinned inputs

- Runtime: `XHToken/llama.cpp`, commit
  `4a3635c32fc9f044c2bde9ebeabf50c7e1ec5991`.
- Parent image: `llama:cuda-swap-v18`, published as
  `127.0.0.1:5000/llama@sha256:23da58507732c60e658bf21ddabb56e1fc9509387fc9730e0a3e39c852b63336`.
- Weights: [XHToken/Spark-X2.5-4B-GGUF](https://huggingface.co/XHToken/Spark-X2.5-4B-GGUF),
  revision `d313463f1ff7dc29db20193fddc4ad8386d16261`,
  `Spark-X2.5-4B-Q8_0.gguf` (4,375,021,152 bytes).
- Weight SHA-256:
  `5c2c3c190e4337e1016b8593ca8e26e8b18c972200b107385d4ec61a25d9dea2`.

## Build

Use a temporary build context so the source archive stays out of Git.
From this lab checkout, with the v18 parent image already present:

```bash
spark_context=$(mktemp -d)
cp engines/spark/Dockerfile "$spark_context/Dockerfile"
(cd "$spark_context" && gh-axi repo clone XHToken/llama.cpp)
git -C "$spark_context/llama.cpp" archive --format=tar \
  4a3635c32fc9f044c2bde9ebeabf50c7e1ec5991 -o "$spark_context/source.tar"
printf '%s\n' '*' '!source.tar' '!Dockerfile' > "$spark_context/.dockerignore"
BUILDX_CONFIG="$spark_context/buildx" docker build -t llama:cuda-swap-v19 "$spark_context"
# From the parent docker-services checkout:
scripts/publish-image-to-k3s.sh llama:cuda-swap-v19
```

CUDA targets sm_86. The Spark binary statically links its ggml/llama
libraries to avoid binding to the parent image's incompatible libraries.
The build disables both UI compilation and prebuilt UI downloads.

## Selection and memory budget

Q8_0 weighs 4.08 GiB; original BF16 weighs 7.66 GiB. BF16 also fits at
128k, but Q8 saves about 3.6 GiB and reduces weight traffic during decode.
This is a deployment-fit choice, not a measured speed or quality comparison.
Q4_K_M is unnecessary for fitting this model on a 24 GiB card.

From the [model configuration](https://huggingface.co/XHToken/Spark-X2.5-4B/blob/main/config.json),
the nine full-attention layers need 36 KiB/token in FP16 KV. The 27 sliding
layers use 512-token windows. With bounded SWA allocation, weights plus KV
are about 8.63 GiB at 128k and 13.13 GiB at 256k, before runtime workspace.
Full 1M context needs 36 GiB of FP16 full-attention cache alone.

The publisher supports SGLang and vLLM too. They are alternatives for a
dedicated batched service; this deployment uses the existing on-demand
llama-swap integration. The 3090 is Ampere, so native FP8 W8A8 is not its
fast path; weight-only fallback support is a separate engine/kernel question.
See [vLLM hardware support](https://docs.vllm.ai/en/latest/features/quantization/).

## Qualification

The [2026-09-10 qualification](qualification-2026-09-10.json) passed exact
text output, separate reasoning content, streamed output, a structured
`get_weather` call and tool-result follow-up, and retrieval of three values
from a 118,082-token prompt. The retrieval used repeated inventory-record
filler with values inserted at three depths; it is a smoke check, not a
general long-context quality benchmark.

Three distinct coding prompts with 256-token outputs and thinking disabled
measured 110.58, 107.50, and 108.41 decode tokens/s. At 118k depth, the
39-token retrieval answer decoded at 66.27 tokens/s; cold prompt processing
ran at 3,886 tokens/s (30.38 seconds). Total request time was 31.43 seconds.
GPU memory sampled every second peaked at 9,408 MiB of 24,576 MiB, leaving
15,168 MiB free. Thinking was enabled for the arithmetic reasoning check.

The published image digest is
`sha256:b2f307e0f21f073f16db7e7312fcbbd78cab11ea3c6a3546b77f288c7908d5a6`.
The server reports an unknown build commit because the build context uses
`git archive`; the exact source revision is recorded in the image labels.

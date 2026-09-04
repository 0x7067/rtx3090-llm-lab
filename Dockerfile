FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS build

# v14 (2026-09-02): master 0f3a71be1 "mtmd: Fix Qwen3-tts-0.6b (#28231)". Brings
# upstream DFlash2 (#27342), DSpark (#25173), EAGLE-3 for qwen3.5/3.6 (#24593),
# the hybrid checkpoint-restore fixes (#24411 et al.) and backend draft sampling.
# Patch rebase notes: patches-v14/REBASE-2026-09-02.md. Previous: 4df29be4 (2026-08-16).
ARG LLAMA_CPP_REF=0f3a71be1
# v15 (2026-09-03): same base, adds patches-v15/0009 (K2 Horizon arch from the
# MBZUAI-IFM fork) and ships llama-quantize + llama-imatrix for local quants.
# llama-swap: OpenAI-compatible proxy that hot-swaps llama-server backends so a
# single GPU can serve multiple models (one resident at a time).
ARG LLAMA_SWAP_VERSION=v230

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      cmake \
      curl \
      git \
      libcurl4-openssl-dev \
      libopenblas-dev \
      pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src/llama.cpp
RUN git clone https://github.com/ggml-org/llama.cpp.git . \
    && git checkout "${LLAMA_CPP_REF}"

# Vendored performance patches over the pinned ref (see patches-v14/*.patch):
# 0001/0006 dropped in v14: superseded by upstream backend draft sampling (#26958)
# 0002: Ampere MMQ small-batch (J=16) tile config — 128x64 tiles for Q4_K/Q5_K
# 0003: GQA-batched small-batch FA vector kernel
# 0004: env-gated dense MMVQ batch cap (GGML_CUDA_MMVQ_NE11_MAX)
# 0005: inline-q4-dequant FA MMA path (GGML_CUDA_FATTN_MMA_Q), swizzle-aware since v14
# 0007: qwen35 MTP truncated draft vocab via d2t tensor
# 0008: env-gated small-batch MMQ grid + y-tile double buffer (GGML_CUDA_MMQ_SMALLN)
# 0009: K2 Horizon architecture (see patches-v15/README.md)
# 0010: effort-specific reasoning tags
# 0011: queue requests that would exceed the shared KV pool
COPY patches-v15/ /src/llama.cpp/patches/
RUN git apply --stat --apply patches/*.patch

ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs

RUN ln -sf libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

RUN cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_NATIVE=ON \
      -DGGML_OPENMP=ON \
      -DGGML_BLAS=ON \
      -DGGML_BLAS_VENDOR=OpenBLAS \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=86 \
      -DLLAMA_BUILD_TESTS=ON \
      -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,/usr/local/cuda/lib64/stubs -L/usr/local/cuda/lib64/stubs" \
    && cmake --build build --target llama-server llama-bench llama-perplexity llama-quantize llama-imatrix llama-gguf-split test-backend-ops -j"$(nproc)"

# Fetch the llama-swap release binary (static Go binary, linux/amd64).
RUN curl -fL "https://github.com/mostlygeek/llama-swap/releases/download/${LLAMA_SWAP_VERSION}/llama-swap_${LLAMA_SWAP_VERSION#v}_linux_amd64.tar.gz" \
      -o /tmp/llama-swap.tar.gz \
    && tar -xzf /tmp/llama-swap.tar.gz -C /usr/local/bin llama-swap \
    && chmod +x /usr/local/bin/llama-swap

FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      libcurl4 \
      libgomp1 \
      libopenblas0-pthread \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /src/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=build /src/llama.cpp/build/bin/llama-bench /usr/local/bin/llama-bench
# Kernel-development harnesses: per-op CUDA-vs-CPU correctness/perf, and the
# KL-divergence quality gate used to validate hand-written kernels.
COPY --from=build /src/llama.cpp/build/bin/llama-perplexity /usr/local/bin/llama-perplexity
COPY --from=build /src/llama.cpp/build/bin/test-backend-ops /usr/local/bin/test-backend-ops
# Local quantization of BF16 releases (K2 Horizon ships BF16 GGUF only).
COPY --from=build /src/llama.cpp/build/bin/llama-quantize /usr/local/bin/llama-quantize
COPY --from=build /src/llama.cpp/build/bin/llama-imatrix /usr/local/bin/llama-imatrix
COPY --from=build /src/llama.cpp/build/bin/llama-gguf-split /usr/local/bin/llama-gguf-split
COPY --from=build /src/llama.cpp/build/bin/*.so /usr/local/lib/
COPY --from=build /usr/local/bin/llama-swap /usr/local/bin/llama-swap

RUN ldconfig

WORKDIR /app
EXPOSE 8080

FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS build

RUN apt-get -o Acquire::http::User-Agent="OpenAI File Downloader, XaiImageApiFetch/1.0" update \
    && apt-get -o Acquire::http::User-Agent="OpenAI File Downloader, XaiImageApiFetch/1.0" install -y --no-install-recommends \
      build-essential cmake ca-certificates libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Exported from XHToken/llama.cpp at 4a3635c32fc9f044c2bde9ebeabf50c7e1ec5991.
ADD source.tar /src/llama.cpp/
WORKDIR /src/llama.cpp
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs
RUN ln -sf libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
      -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_OPENSSL=OFF \
      -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,/usr/local/cuda/lib64/stubs -L/usr/local/cuda/lib64/stubs" \
    && cmake --build build --target llama-server llama-bench -j8

FROM llama:cuda-swap-v18
LABEL org.opencontainers.image.description="llama-swap v18 plus isolated Spark X2.5 runtime" \
      org.opencontainers.image.source="https://github.com/XHToken/llama.cpp" \
      org.opencontainers.image.revision="4a3635c32fc9f044c2bde9ebeabf50c7e1ec5991"
# Static ggml/llama linkage keeps Spark independent of v18's shared libraries.
COPY --from=build /src/llama.cpp/build/bin/llama-server /opt/spark/bin/llama-server
COPY --from=build /src/llama.cpp/build/bin/llama-bench /opt/spark/bin/llama-bench

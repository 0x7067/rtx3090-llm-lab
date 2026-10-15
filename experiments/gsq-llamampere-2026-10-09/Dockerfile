FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake ca-certificates libcurl4-openssl-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN ln -s libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && cmake -S . -B build-sm86 -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON -DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/usr/local/cuda/lib64/stubs && cmake --build build-sm86 -j4 --target llama-server
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends libcurl4t64 libgomp1 ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/build-sm86/bin/ /app/
ENV LD_LIBRARY_PATH=/app:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
WORKDIR /app
ENTRYPOINT ["/app/llama-server"]

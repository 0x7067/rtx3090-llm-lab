# Build context: Strata rc2 at 9ff02c9fa9c86a88088370925b3c9131cc5d6c26,
# with agentionai-group/llama.cpp 9aa51ce4fdd3435a91531ffe16a17414ccc6dc6f
# in gyro-llama/. CUDA 13.0, RTX 3090 sm_86.
FROM local/strata-bakeoff:v0.1.44.1 AS build
USER root
COPY . /opt/gyro-src/
ENV PATH=/opt/strata/.venv/bin:/usr/local/cuda/bin:${PATH}
RUN cmake -S /opt/gyro-src -B /opt/gyro-build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=86 -DSTRATA_GGML_DIR=/opt/gyro-src/gyro-llama \
    && cmake --build /opt/gyro-build -j 6 --target strata native_expert_parity \
    && mkdir -p /opt/gyro-runtime/bin \
    && cp /opt/gyro-build/strata /opt/gyro-build/native_expert_parity /opt/gyro-runtime/bin/ \
    && find /opt/gyro-build -name '*.so*' -type f -exec cp {} /opt/gyro-runtime/bin/ \; \
    && cp -a /opt/gyro-src/serve /opt/gyro-src/tools /opt/gyro-src/data /opt/gyro-runtime/ \
    && cp -a /opt/gyro-src/gyro-llama/gguf-py /opt/gyro-runtime/gguf-py
FROM 127.0.0.1:5000/home-llm-sglang@sha256:9807abc74b9126cd9c2a208b500896fedb110500fcbc2fc418abab8bb20b9642
COPY --from=build /opt/gyro-runtime /opt/gyro
COPY --from=build /opt/strata/.venv /opt/strata/.venv
ENV STRATA_GGUF_PY=/opt/gyro/gguf-py

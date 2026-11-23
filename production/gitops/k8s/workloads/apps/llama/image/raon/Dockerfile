# Build context includes the checksum-verified source archive named below.
# Krafton's latest raon-1-mini-preview revision, requested 2026-10-25.
FROM 127.0.0.1:5000/home-llm-ocr:photo-v12-acd0887666d8@sha256:8ca5524734dcc4ba45aca3da2b5449b566fd9a5dc24d0c1f975339f07504cc5b AS build
USER root
COPY llama.cpp-fb895662.tar.gz /tmp/raon-source.tar.gz
RUN echo 'ec9cdf3c4cceac7791c44bf3ccd027c78575dd3ff2e7273ed999ddf729f60f35  /tmp/raon-source.tar.gz' | sha256sum -c - \
    && mkdir /tmp/raon-source \
    && tar -xzf /tmp/raon-source.tar.gz -C /tmp/raon-source --strip-components=1 \
    && PATH=/usr/local/cuda/bin:/opt/sglang/bin:$PATH cmake -S /tmp/raon-source -B /tmp/raon-build -G Ninja \
       -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
       -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
       -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON \
    && cmake --build /tmp/raon-build --target llama-server -j 6 \
    && mkdir /opt/raon \
    && cp -a /tmp/raon-build/bin/. /opt/raon/

# Preserve the deployed text, image and OCR runtimes; add Raon's engine only.
FROM 127.0.0.1:5000/home-llm-ocr:photo-v12-acd0887666d8@sha256:8ca5524734dcc4ba45aca3da2b5449b566fd9a5dc24d0c1f975339f07504cc5b
COPY --from=build /opt/raon /opt/raon

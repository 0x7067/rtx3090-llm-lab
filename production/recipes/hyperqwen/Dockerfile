# Preserve the existing Flash, ThinkingCap and image runtimes; add the exact
# HyperQwen runtime qualified in benchmarks/hyperqwen-2026-10-21/.
FROM ghcr.io/syv-ai/hyperqwen@sha256:c696d22e80b459db9469e6b6de81ab669db34a578eede7d3e5a017615253dfc4 AS hyperqwen
FROM 127.0.0.1:5000/home-llm-qwen-image@sha256:448250f25f2393280f254ee027976ba59a22550a74003a24b39fe56fc3edbc2f
COPY --from=hyperqwen /app /app
RUN env PYTHONPATH= /app/venv/bin/python -c "import vllm; assert vllm.__version__ == '0.31.0'"
LABEL org.opencontainers.image.description="Home LLM runtimes with HyperQwen 2cf50a0, DFlash2, KVarN and vision"

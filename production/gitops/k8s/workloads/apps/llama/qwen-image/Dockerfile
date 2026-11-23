# Production llama image plus llama-swap v265 and the qwen-image API.
# Build: see README.md. Context: this directory.
FROM ghcr.io/astral-sh/uv:0.12.24 AS uv

FROM busybox:1.37 AS llama-swap
ADD --checksum=sha256:cb1f9a94be05afa129d3c17868f1dd2ca710b919333c7b868239dc31575dcfa1 \
    https://github.com/mostlygeek/llama-swap/releases/download/v265/llama-swap_265_linux_amd64.tar.gz /tmp/llama-swap.tar.gz
RUN tar -xzf /tmp/llama-swap.tar.gz -C /tmp llama-swap

FROM 127.0.0.1:5000/home-llm-gyro@sha256:61d509794b6e39de19d1b33bb49e4ecd544bf0f4417b64caeb5158c2b5cd9110
COPY --from=llama-swap /tmp/llama-swap /usr/local/bin/llama-swap
COPY --from=uv /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock /opt/qwen-image/
RUN UV_PROJECT_ENVIRONMENT=/opt/qwen-image/venv uv sync --frozen --no-dev --no-cache \
      --python /usr/bin/python3 --project /opt/qwen-image
COPY server.py pe_core.py /opt/qwen-image/
LABEL org.opencontainers.image.description="home-llm-gyro + llama-swap v265 + qwen-image API (Qwen-Image-2.1 / Turbo with PE)"

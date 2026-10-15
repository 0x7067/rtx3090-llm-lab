ARG VLLM_BASE=vllm/vllm-openai@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3
FROM ${VLLM_BASE}
RUN apt-get update && apt-get install -y --no-install-recommends g++ ninja-build curl ca-certificates && rm -rf /var/lib/apt/lists/*
RUN curl -fL 'https://github.com/turboderp-org/exllamav3/releases/download/v1.5.4/exllamav3-1.5.4%2Bcu132.torch2.13.0-cp312-cp312-linux_x86_64.whl' -o '/tmp/exllamav3-1.5.4+cu132.torch2.13.0-cp312-cp312-linux_x86_64.whl' \
 && echo 'e95ed89809435d6463c3c51f86b3af4266f6bb35bd4ca65c0c0113d4a1f6c0be  /tmp/exllamav3-1.5.4+cu132.torch2.13.0-cp312-cp312-linux_x86_64.whl' | sha256sum -c - \
 && uv pip install --system --no-deps '/tmp/exllamav3-1.5.4+cu132.torch2.13.0-cp312-cp312-linux_x86_64.whl' \
 && rm '/tmp/exllamav3-1.5.4+cu132.torch2.13.0-cp312-cp312-linux_x86_64.whl'
RUN python3 - <<'PY'
import sys, torch, subprocess
assert sys.version_info[:2] == (3,12)
assert torch.__version__.split('+')[0] == '2.13.0'
assert torch.version.cuda.startswith('13.')
subprocess.check_call(['uv','pip','install','--system','--no-deps','nvidia-cuda-nvcc=='+torch.version.cuda+'.*'])
PY
COPY upstream/pyproject.toml upstream/README.md /opt/orcasaq2/
COPY upstream/orcasaq2 /opt/orcasaq2/orcasaq2
COPY upstream/presets /opt/orcasaq2/presets
RUN uv pip install --system --no-deps /opt/orcasaq2 && chmod +x /opt/orcasaq2/presets/serve.sh
ENV VLLM_USE_FLASHINFER_SAMPLER=0
ENTRYPOINT ["/opt/orcasaq2/presets/serve.sh"]

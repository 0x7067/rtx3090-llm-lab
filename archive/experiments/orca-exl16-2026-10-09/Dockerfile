FROM local/orcasaq-latest:2026-10-07-v1
COPY exllamav3-1.6.0+cu132.torch2.13.0-cp312-cp312-linux_x86_64.whl /tmp/
RUN echo '21e517df8f8e19f07f8fa1075411a55f00689246d7844485c7c36b42b5666034  /tmp/exllamav3-1.6.0+cu132.torch2.13.0-cp312-cp312-linux_x86_64.whl' | sha256sum -c - \
 && python3 -m pip install --no-deps --no-cache-dir /tmp/exllamav3-1.6.0+cu132.torch2.13.0-cp312-cp312-linux_x86_64.whl \
 && rm /tmp/exllamav3-1.6.0+cu132.torch2.13.0-cp312-cp312-linux_x86_64.whl

FROM 127.0.0.1:5000/home-llm-hyperqwen@sha256:72bddee038683e24c71700afff186da2c77160b0233a90eacdfefff66f7eec95
COPY source/ /opt/strata-next/
COPY engine/ /opt/strata-next/bin/
RUN /opt/strata/.venv/bin/python -m py_compile /opt/strata-next/serve/server.py
LABEL org.opencontainers.image.description="Qualified Qwen runtimes: HyperQwen 2cf50a0 and Strata 011c9e6 / engine 0.1.47"

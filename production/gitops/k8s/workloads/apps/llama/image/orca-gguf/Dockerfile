FROM 127.0.0.1:5000/home-llm-strata:sparsekv-v1@sha256:ecb915cc18b92f1c81dcc97d22576bac3eacba436afc00fb7afeb6d1ba604679
COPY --link .payload/ /opt/orca/
COPY sources.json install_overlay.py /opt/orca/
COPY --chmod=755 serve-long.sh /opt/orca/serve-long.sh
COPY --chmod=755 serve-short.sh /opt/orca/serve-short.sh
RUN /opt/sparsekv/venv/bin/python -I -S /opt/orca/install_overlay.py

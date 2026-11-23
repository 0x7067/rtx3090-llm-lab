ARG SWAP_BASE=127.0.0.1:5000/home-llm-swap@sha256:b7c5c82d248d7dcc93ba4b93d92120511db93f71ec2024fe4d45fc17ce425b8c
FROM ${SWAP_BASE}
RUN python3 -m pip install --no-cache-dir --no-deps bitsandbytes==0.50.2
COPY server.py joint_schema_model.py LICENSE /opt/clef/

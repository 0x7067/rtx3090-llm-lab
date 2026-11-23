FROM 127.0.0.1:5000/home-llm-strata:0.2.8-v1@sha256:b2b21d86b3926f564499b7415d4f0afae9b2dddd7348ec7c5082c5077cbd8c42
COPY apply_patch.py /opt/normal120-dflash-short/apply_patch.py
RUN /app/venv/bin/python -I -S /opt/normal120-dflash-short/apply_patch.py /app/single-user/start_qwen.sh \
    && bash -n /app/single-user/start_qwen.sh

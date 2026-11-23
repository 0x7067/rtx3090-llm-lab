# Strata 0.2.8 (engine 0.1.54, CUDA 13) on top of the production llama image.
# The base has 131 layers, past overlay2's limit, so Docker cannot unpack it. Every
# step here is a COPY --link, which BuildKit applies without unpacking the base.
# Strata goes to its own path; the base's /opt/strata-next stays untouched.
FROM 127.0.0.1:5000/home-llm-raon:photo-v20-f4bcf045d331@sha256:aba9ee3dab0027e5b90fa372694694ae3c5edec7d274bf80bf1317600ad45454
COPY --link --chown=1000:1000 source/ /opt/strata-0.2.8/
COPY --link --chown=1000:1000 engine/ /opt/strata-0.2.8/bin/
# truststore + certifi from requirements.txt, hash-checked by build.sh; on PYTHONPATH.
COPY --link --chown=1000:1000 site/ /opt/strata-0.2.8/site/

FROM 127.0.0.1:5000/home-llm-strata:normal120-dflash-short-v1@sha256:83077f60377d9e303d446a2f045381daab28a123e8e314c5bbe1ba53314884a2
# build.sh compiles and verifies this layer in the same digest-pinned base.
# Linked COPY reuses registry base layers without unpacking the full image again.
COPY --link .runtime/ /opt/sparsekv/

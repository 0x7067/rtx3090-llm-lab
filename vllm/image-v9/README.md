# v9 overlay: the currently deployed image

`qwen38-27b-3090:v9` is what `deploy/llama` runs today. It is **not** the v4
image described in [`../../docs/vllm-companion.md`](../../docs/vllm-companion.md)
build snippet; that snippet reproduces the original syv-ai overlay only. The
deployed lineage is:

```text
syv-ai/qwen38-27b-rtx3090 @ b356e31526886b4bd614a79cd8600e7cc9383cf9
  + ../syv-ai-b356e315-v4.patch          -> v4  (FlashInfer pageable buffers,
                                                 VISION=1, UID-1000 tree)
  + locked base rebuild                  -> v7/v8 (see the docker-services repo,
                                                 k8s/workloads/apps/llama/vllm-base/)
  + xgrammar-terminated-batch.patch      -> v9
  + vllm-pr51812-gdn-spec-gates.patch
```

`Dockerfile` here is the v8 -> v9 overlay verbatim, mirrored from
`k8s/workloads/apps/llama/vllm-patches/Dockerfile` in the docker-services repo,
which is the authoritative copy Flux and the build script use.

## The two patches

- `xgrammar-terminated-batch.patch` — vLLM 0.27.1 structured-output fix. v7
  carried it; **v8 silently dropped it** by shipping from the locked base
  without this overlay. v9 re-applies it. It applies with `fuzz 1` on the
  `__init__.py` hunk against the v8 tree (4-line edge context; the applied
  region was verified byte-identical to the hunk's intended output), so its
  build gate is a content assertion on the two added signature lines rather
  than a strict reverse dry-run.
- `vllm-pr51812-gdn-spec-gates.patch` — upstream PR #51812 backport, GDN
  speculative-decode gates. Applies and reverse-checks strictly (`-F0`).

Both are asserted at build time and byte-compiled; see the `RUN` block.

## Rebuild

```bash
docker build --build-arg BASE_IMAGE=qwen38-27b-3090:v8 -t qwen38-27b-3090:v9 .
```

Then import into k3s containerd and bump the two `image:` lines in
`k8s/workloads/apps/llama/deployment.yaml`. v8 is still in containerd as the
image-level rollback.

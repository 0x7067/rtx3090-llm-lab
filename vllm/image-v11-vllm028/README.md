# vLLM overlay v11 — vLLM 0.28 candidate (exported branch diff, never deployed)

Exported, not vendored as a submodule: the source lives on the branch
`local/k8s-deploy-v11-vllm028` in the fork checkout at
`/data/buttercup_6tb/k3s/vllm-trial/df2-repo`
(remote `fork` = [`0x7067/qwen38-27b-rtx3090`](https://github.com/0x7067/qwen38-27b-rtx3090),
remote `origin` = upstream [`syv-ai/qwen38-27b-rtx3090`](https://github.com/syv-ai/qwen38-27b-rtx3090)).
See the root README's "Where things live" section for why this fork's deploy
branches are captured here as diffs rather than a second submodule.

- **Base commit**: `453104e` — `git merge-base origin/main local/k8s-deploy-v11-vllm028`,
  the same syv-ai upstream fork point as v10.
- **Diff**: [`k8s-deploy-v11-vllm028.patch`](k8s-deploy-v11-vllm028.patch),
  generated with `git diff origin/main..local/k8s-deploy-v11-vllm028`, applies
  against a checkout of `453104e`.
- **Status**: a vLLM 0.28.0 candidate, per its branch-tip commit message
  ("vLLM 0.28.0 candidate: prune the patch set to what applies, SPEC=mtp
  only"). **Never deployed.** llama.cpp v14 was promoted straight from v10 on
  2026-09-02 (see
  [`../../benchmarks/engine-trial-2026-09-02/`](../../benchmarks/engine-trial-2026-09-02/)),
  so this branch has no corresponding k8s manifest — unlike v10, there is no
  `deployment-k8s-last.yaml` to pair with it here.

## Files changed (32 files, +355/−4820)

```
 bench/mq3d_capacity_property.py               | 135 ----
 bench/mq3d_layer2_oracle.py                   | 511 -------------
 bench/mq3d_layer2_verdicts-3090.jsonl         |  11 -
 bench/mq3d_layer2_verdicts.jsonl              |  11 -
 bench/needle_test.py                          |  68 --
 bench/quality_battery.py                      |   9 -
 clients/README.md                             |  45 ++
 clients/sync-agent-models.py                  | 140 ++++
 docker-compose.override.yml                   |   5 +
 docker/patch_flashinfer_pageable_buffers.sh   |  25 +
 docker/requirements.txt                       |   2 +-
 docs/gotchas.md                               |  52 +-
 docs/optimizations.md                         |  11 -
 docs/spec-decode-scratch-token-units.md       | 221 ------
 patches/_check_applied.py                     |  54 +-
 patches/dflash2-backport.patch                | 995 --------------------------
 patches/dflash2-lookup-drafting.patch         | 947 ------------------------
 patches/dflash2-ngram-chains.patch            | 431 -----------
 patches/dflash2-prewarm.patch                 | 130 ----
 patches/offload-wsl2-devptr.patch             |  95 ---
 patches/spec-decode-int8-kv.patch             | 227 ------
 patches/spec-decode-scratch-token-units.patch | 420 -----------
 patches/triton-prefill-attn-int8.patch        | 341 ---------
 patches/vision-tower-cpu-offload.patch        |  10 +-
 patches/xgrammar-spec-terminated.patch        |  80 ---
 patches/xgrammar-terminated-batch.patch       |  77 ++
 prepare/build_draft_vocab.py                  |   8 +-
 single-user/alternative.sh                    |  14 +-
 single-user/start_qwen.sh                     |  37 +-
```

Net effect versus upstream `main`: same `xgrammar-terminated-batch` swap and
bench-script drops as v10, plus — per its "SPEC=mtp only" branch note — it
drops the whole DFlash2 patch family (`dflash2-backport`,
`dflash2-lookup-drafting`, `dflash2-ngram-chains`, `dflash2-prewarm`), the
`triton-prefill-attn-int8` and `spec-decode-int8-kv` patches, and the WSL2
devptr offload patch, pruning the patch set down to what actually applies
against vLLM 0.28.0 with MTP as the only speculative-decoding path. Keeps
`vision-tower-cpu-offload` (adjusted) from v10.

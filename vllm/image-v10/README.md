# vLLM overlay v10 — exported branch diff

Exported, not vendored as a submodule: the source lives on the branch
`local/k8s-deploy-v10` in the fork checkout at
`/data/buttercup_6tb/k3s/vllm-trial/df2-repo`
(remote `fork` = [`0x7067/qwen38-27b-rtx3090`](https://github.com/0x7067/qwen38-27b-rtx3090),
remote `origin` = upstream [`syv-ai/qwen38-27b-rtx3090`](https://github.com/syv-ai/qwen38-27b-rtx3090)).
That fork is kept only as the vehicle for upstreaming PRs to syv-ai — see the
root README's "Where things live" section — so its deploy branches are
captured here as diffs instead of this repo carrying a second submodule.

- **Base commit**: `453104e` — `git merge-base origin/main local/k8s-deploy-v10`,
  which is syv-ai upstream `main` at the point the branch forked.
- **Diff**: [`k8s-deploy-v10.patch`](k8s-deploy-v10.patch), generated with
  `git diff origin/main..local/k8s-deploy-v10`, applies against a checkout of
  `453104e`.
- **Deployed as**: this was the vLLM source rebased onto syv-ai `main 453104e`
  on 2026-09-01, running in production from then until the 2026-09-02
  llama.cpp v14 promotion (see
  [`../../benchmarks/engine-trial-2026-09-02/`](../../benchmarks/engine-trial-2026-09-02/)).

## Files changed (23 files, +345/−1645)

```
 Dockerfile                                    |   5 +
 README.md                                     |  37 +-
 batch/start_qwen.sh                           |  14 +-
 bench/mq3d_capacity_property.py               | 135 -------
 bench/mq3d_layer2_oracle.py                   | 511 --------------------------
 bench/mq3d_layer2_verdicts-3090.jsonl         |  11 -
 bench/mq3d_layer2_verdicts.jsonl              |  11 -
 bench/needle_test.py                          |  68 ----
 bench/quality_battery.py                      |   9 -
 clients/README.md                             |  45 +++
 clients/sync-agent-models.py                  | 140 +++++++
 docker-compose.override.yml                   |   5 +
 docker/patch_flashinfer_pageable_buffers.sh   |  25 ++
 docs/gotchas.md                               |  52 +--
 docs/optimizations.md                         |  11 -
 docs/spec-decode-scratch-token-units.md       | 221 -----------
 patches/_check_applied.py                     |  54 +--
 patches/spec-decode-scratch-token-units.patch | 420 ---------------------
 patches/xgrammar-spec-terminated.patch        |  80 ----
 patches/xgrammar-terminated-batch.patch       |  77 ++++
 prepare/build_draft_vocab.py                  |   8 +-
 single-user/alternative.sh                    |  14 +-
 single-user/start_qwen.sh                     |  37 +-
```

Net effect versus upstream `main`: drops the `spec-decode-scratch-token-units`
and `xgrammar-spec-terminated` patches (replaced by `xgrammar-terminated-batch`,
carried forward from the v9 overlay — see
[`../image-v9/`](../image-v9/)), drops the mq3d/needle/quality bench scripts,
and adds a `clients/sync-agent-models.py` helper plus the FlashInfer pageable
buffers patch script.

## `deployment-k8s-last.yaml`

The k3s Deployment manifest actually running this profile, captured with
`git -C /data/docker-services show 30f5140^:k8s/workloads/apps/llama/deployment.yaml`
— i.e. the manifest as it stood immediately before the docker-services commit
that promoted llama.cpp v14 (`30f5140`) replaced it. This is the last vLLM
deployment manifest, kept here as a standalone record since the GitOps repo's
working tree no longer has it (the current manifest is llama.cpp's — see
`k8s/workloads/apps/llama/deployment.yaml` and `k8s/MIGRATION_LOG.md` there for
the live rollback path).

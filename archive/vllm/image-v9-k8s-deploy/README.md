# k8s-deploy overlay (pre-v10), exported from the fork 0x7067/qwen38-27b-rtx3090

Branch `local/k8s-deploy`, 4 commits on top of syv-ai base `b356e31526886b4bd614a79cd8600e7cc9383cf9`, exported 2026-09-02 before the fork was deleted. Superseded by image-v10; kept for history.

```
48ca3e0 2026-08-27 Add client-config sync with dynamic parameter discovery
2e49eaf 2026-08-27 Take upstream's sm80 Marlin repack staging patch
7c6106e 2026-08-27 Local: VISION knob and a drafter that can leave FlashInfer
c7656bc 2026-08-27 Local: FlashInfer pageable planning buffers and a k8s-runnable image

 Dockerfile                                  |   5 +
 clients/README.md                           |  45 +++++++++
 clients/sync-agent-models.py                | 140 ++++++++++++++++++++++++++++
 docker-compose.override.yml                 |   5 +
 docker/patch_flashinfer_pageable_buffers.sh |  25 +++++
 patches/marlin-repack-staged-sm80.patch     | 129 +++++++++++++++++++++++++
 prepare/build_draft_vocab.py                |   8 +-
 single-user/start_qwen.sh                   |   9 +-
 8 files changed, 361 insertions(+), 5 deletions(-)
```

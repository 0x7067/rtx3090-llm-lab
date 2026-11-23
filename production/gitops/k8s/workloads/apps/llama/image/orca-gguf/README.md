# Orca GGUF shared-image candidate

This recipe stages the native K3 adapter and draft-page eviction policy
selected in the short and 32k Orca Q8 trials. The combined 160k GPU run
remains in qualification. The pinned base image supplies the existing image
and OCR runtimes. The Orca process loads its own Python package view with
`PYTHONPATH=/opt/orca/python:/opt/orca/plugin:/opt/orca/gguf-source`.

`sources.json` pins the adapter, llama.cpp reader revision, base image,
runtime patches, host libraries, model metadata and exact chat template.
`prepare.py` verifies each local artifact and exports the reader from its
pinned commit. It stages only small configuration and tokenizer files.
Weights and the projector remain read-only SSD mounts. The generated
`.payload/` directory stays out of Git.

```sh
uv run --no-project python k8s/workloads/apps/llama/image/orca-gguf/prepare.py \
  --adapter /srv/llama-nvme/cyber-gguf-benchmark-20261118/gguf-adapter-native-kquant-01 \
  --llama-source /data/development/cyber-gguf-benchmark-20261118/llama-source \
  --overlay /srv/llama-nvme/cyber-gguf-benchmark-20261118/cachefix-draft-page-priority-01 \
  --model-config /srv/llama-nvme/cyber-gguf-benchmark-20261118/gguf-adapter-build-01/model \
  --chat-template "$CAMPAIGN_WORKTREE/experiments/gguf-benchmark-20261118/orca_chat_template.jinja" \
  --short-runtime /data/development/cyber-gguf-benchmark-20261118/llama-b12546-fallback/extracted/llama-b12546 \
  --short-draft /data/development/cyber-gguf-benchmark-20261118/models/drafter/Qwen3.8-27B-DFlash2-Q8_0.gguf
```

Use `/opt/orca/model` for the tokenizer and HF config and
`/opt/orca/chat_template.jinja` for the explicit serving template.

The long-profile launcher takes the llama-swap backend port and binds to
loopback. The runtime uses these read-only mounts.
Their host paths are relative to
`/srv/llama-nvme/cyber-gguf-benchmark-20261118`.

| Host path | Container path |
| --- | --- |
| `orcasaq2/OrcaSAQ-2-27B-Uncensored.gguf` | `/opt/orca/weights/model.gguf` |
| `orcasaq2/mmproj-Qwen3.8-27B-Uncensored-f16.gguf` | `/opt/orca/weights/mmproj.gguf` |
| `drafts/Qwen3.8-27B-DFlash2-W4A16` | `/opt/orca/draft` |
| `native-iq4-device-03` | `/native-device` |
| `run/runtime/llama-b12110` | `/native` |
| `run/runtime/cudart-llama-b12110-bin-ubuntu-cuda-12.8-x64` | `/native-cuda` |

The short-profile launcher, `serve-short.sh PORT`, uses the same model and
projector with the measured llama.cpp b12546 K5 command, four 8,192-token
slots and Q8 target and draft caches. The model IDs are `orcasaq2-cyber`
for long contexts and `orcasaq2-cyber-fast` for short contexts. Both
profiles use the same target weights. llama-swap loads one model at a
time, and switching profiles includes the new backend's load time.
The short benchmark measures generation after load.

The short profile's two additional read-only mounts
come from the development SSD.

| Host path below `/data/development/cyber-gguf-benchmark-20261118` | Container path |
| --- | --- |
| `llama-b12546-fallback/extracted/llama-b12546` | `/opt/orca/llama-fast` |
| `models/drafter/Qwen3.8-27B-DFlash2-Q8_0.gguf` | `/opt/orca/fast-draft.gguf` |

The long launcher sets the library path and keeps its compile and tokenizer caches
under `/cache/orca`. The host mounts provide the tested CUDA libraries.
The long route enables the Qwen XML tool-call parser with parameter
strictness for clients that send function definitions through CLIProxyAPI.

`install_overlay.py` checks every original and replacement, then stages private
`vllm` and `sparsekv` package views for Orca. Unchanged files link to the pinned
base packages. HyperQwen and the other models continue to import the stock
packages through their existing environments. This deployment mounts the
verified runtime bundle at `/opt/orca` from
`/srv/llama-nvme/production-models/orca-runtime/3c5f9f37-tools`. The
base image stays pinned to the digest used to assemble and test the
bundle. Stage each future bundle in a fresh SSD directory, verify its
payload and nested mounts with that exact base, then update the hostPath.
The pending live checks cover both Orca routes, image generation and
Finance OCR. Campaign evidence is in the private
`0x7067/qwen-cyber-campaign` repository on `codex/gguf-cyber-benchmark`.

For optional saved-prefix warming, use a new Orca-specific
`SPARSEKV_WARM_PREFIX_FILE`, set `MODEL` to the GGUF path, and set
`SPARSEKV_WARM_TOKENIZER_FILE` to the separate tokenizer JSON. The warmer retains
model-path and tokenizer-hash binding. `SPARSEKV_WARM_MODEL_ID` selects the
backend's served name (default `qwen3.8-27b`). Existing directory-based models
keep their original defaults. No saved prefix bundle is created by this change.

The long launcher is mounted from the generated `llama-orca-launcher` ConfigMap
at `/orca-launcher/serve-long.sh`, so serving changes roll through GitOps.
Adaptive DFlash verifies five draft tokens for a single request and three for
two to four requests. Target, draft and host KV remain Q8; context stays 163,840.
The matched short screen measured 76.8 versus 59.3 tok/s with capped medium
reasoning and 122.1 versus 111.1 tok/s combined over four nonthinking requests.
All 16 vision fixtures passed. Full-length retention and task-quality checks
for adaptive drafting remain pending. Campaign receipt `context-adaptive-speed-20261119.json`
is committed in the private campaign repository at `c134691`.

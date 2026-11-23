# Qwen 3D Coder and Raon

The optional `qwen3.5-4b-3d-coder` and `raon-1-mini-preview` models use the
latest publisher revisions checked on 2026-10-25. Both share llama-swap's
existing authenticated endpoint and GPU eviction behavior. Models load on demand;
no specific model is preloaded at stack startup.

Qwen uses the publisher's Q8_0 GGUF (4.48 GB). Raon uses the publisher's
Q4_K_M GGUF (19.69 GB), sized for the RTX 3090. Both use one slot, their native
262,144-token context window and a default 32,768-token output budget.
The context window covers input and output together; the output budget is not
additional context. Both advertise text input only and keep all model layers
on the GPU. Qwen uses Q8 K/V cache; Raon retains the engine's F16 cache default.
No speculative draft is enabled. Both backends set the default output budget
with `--n-predict`; explicit request budgets can override this default. Client
discovery advertises 32,768 output tokens.

On 2026-10-25, both 256K candidates passed a three-record retrieval check on the
RTX 3090 with all model layers on the GPU. Each record was inserted at a
different position in a long reference document.

| Model | Actual prompt tokens | Retrieved records | Peak GPU memory | Prompt prefill |
| --- | --- | --- | --- | --- |
| Qwen 3D Coder | 245,095 | 3/3 | 10,408 MiB | 102.0 s |
| Raon | 245,075 | 3/3 | 21,102 MiB | 396.8 s |

The test used temperature zero and disabled thinking for deterministic lookup.
It checks allocation, ingestion and retrieval near the serving limit; it does
not establish general reasoning quality at 256K. Long prompts cost additional
prefill time. Short requests do not fill the entire allocated context window.

Balanced coding defaults follow the publishers' sampling recommendations:

| Model | Temperature | Top P | Top K | Repeat penalty | Presence penalty |
| --- | --- | --- | --- | --- | --- |
| Qwen 3D Coder | 0.7 | 0.95 | 20 | 1.0 | 0.0 |
| Raon | 1.0 | 0.95 | 20 | 1.0 | 0.0 |

Qwen uses medium reasoning effort; Raon keeps thinking enabled. Both use
`min_p=0` and a zero frequency penalty. Explicit request sampling parameters
can override these server defaults. Raon's publisher recommends presence
penalty 0 for coding and math, and 1.5 for other tasks. See the
[Raon sampling guide](https://huggingface.co/KRAFTON/Raon-1-mini-preview-gguf#sampling-parameters)
and [Qwen model card](https://huggingface.co/ccore/Qwen3.5-4B-3D-Coder).

The serving stack stays running through the existing single-replica Deployment,
with startup, readiness and liveness probes; it has no idle scale-down or startup
model preload. Models load and swap on demand on the shared GPU. `ttl: 0`
disables idle backend unloading, but does not prevent eviction when another model is selected.
The two models are not a persistent group and have no CPU expert offload.

Raon requires [Krafton's llama.cpp fork](https://github.com/krafton-ai/llama.cpp/tree/raon-1-mini-preview).
The latest branch commit is `fb895662d25868756a19253ca5e0688f99b18994`
(2026-10-20). The user explicitly requested the latest versions after the
older-source plan was explained, authorizing the under-seven-day source
exception for this deployment. The base image is the exact deployed OCR v12
image; the additive build retains its existing engines and application code.
Both new models use the added engine under `/opt/raon`; other model commands
keep their existing runtimes. No new packages are installed during the build.
The user requested upstream build defaults, including its normal UI asset
downloads. The existing llama-swap interface remains the entry point.
This build resolved the prebuilt UI from upstream's `latest` bucket, verified
its archive checksum and embedded 70 assets.

`artifacts.json` records the immutable source revisions and publisher SHA-256
values. Download model files into their recorded library directories with a
`.part` suffix, verify size and SHA-256, then rename to the original filename.
The directories are mounted read-only in the serving pod.

Build with the verified source archive in a temporary directory on Buttercup:

```sh
curl -fL https://github.com/krafton-ai/llama.cpp/archive/fb895662d25868756a19253ca5e0688f99b18994.tar.gz \
  -o "$build_dir/llama.cpp-fb895662.tar.gz"
docker build -t home-llm-raon:20261025-fb895662-upstream \
  -f k8s/workloads/apps/llama/image/raon/Dockerfile "$build_dir"
scripts/publish-image-to-k3s.sh home-llm-raon:20261025-fb895662-upstream
```

Use the publish script's digest reference in `deployment.yaml`, run the
supply-chain checks and render both Kustomizations, then deploy through Flux.
To roll back, revert the image pin, two model entries and their read-only
mounts. Keep verified weights and runtime artifacts available on Buttercup.

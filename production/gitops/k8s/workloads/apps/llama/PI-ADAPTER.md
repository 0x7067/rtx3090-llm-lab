# Qwen3.8 Pi adapter

`qwen3.8-pi` serves the original harness-hub rank-32 LoRA over the existing
AutoRound Qwen3.8-27B base, through the authenticated llama-swap endpoint.
Pi's shared dotfiles register it as `home-llama/qwen3.8-pi`, using Chat
Completions and the existing `HOME_VLLM_API_KEY` environment credential.

## Artifact identity

- [Original adapter](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted),
  revision `3cad999e2b9ec72fea8d802cc286e4502adfa082`.
- `adapter_model.safetensors`: 637610624 bytes,
  SHA256 `869b2a089882c0dac4683993f11081bd70edc30b32c64a988c688d6ac26b792a`.
  Both the library download and NVMe serving copy matched the published LFS hash.
- `adapter_config.json` specifies rank 32, alpha 64, and q/k/v/o/gate/up/down
  projection targets on `Qwen/Qwen3.8-27B`.
- The library retains the pinned revision under
  `/data/models/qwen3.8/harness-hub/Qwen3.8-27B-pi-adapted/`, with `source.json`
  provenance and a catalog entry. `prepare-ssd-models.sh` stages it onto NVMe;
  `deployment.yaml` mounts that revision read-only at `/app/pi-adapter`.
- The existing base and runtime are reused. Their configuration and image pin
  remain in the manifests; no new runtime image is introduced.

## Serving choices

The model card evaluates BF16 vLLM with thinking off, temperature 0.8 and top-p
1.0. The local configuration retains those generation choices, the recommended
`qwen3_xml` parser, aligned Mamba prefix caching and text-only input. The 3090
uses the existing quantized base instead of BF16 weights.

The selected speed profile uses four static MTP drafts, per-head int8 KV,
four admission slots and a 172,032-token window including output. The installed
PTH attention kernel fits about 192K cached tokens at a 95% GPU budget. Long
requests share that pool. `swap-configmap.yaml` owns the exact settings.

Prefix matching uses 16-token boundaries independently of the physical cache
page policy, with shared-prefix checkpoints enabled. This reduces replay of
unchanged input between Pi turns. The controlled draft-count and cache comparison,
qualification receipts and reproduction probe are in
[`benchmarks/qwen-pi-cache-2026-11-20/REPORT.md`](../../../../benchmarks/qwen-pi-cache-2026-11-20/REPORT.md).

The profile restores the model's FP32 recurrent state: the shared launcher
otherwise forces FP16. The installed runtime confirms `GDN decode kernel: cuda`
with FP32. Triton's compilation cache is persistent on `/cache/triton`.

LoRA is restricted to the seven trained projection types. Without this restriction,
vLLM attempts to wrap the existing packed embedding, whose implementation has no
`.weight`, and initialization fails. The original adapter has no embedding targets.

The Pi worker is preloaded at startup and has no idle expiry, keeping weights
and prefix state ready for coding. Selecting another model unloads the active
worker on the shared GPU. HyperQwen and the other configured models remain
available on demand.

The shared Pi provider uses the authenticated Caddy Tailnet route documented in
the service README. The remote Pro and Ravn Macs reached that route successfully;
the public hostname returned Cloudflare 403s on those networks. Credentials
matched across the machines and the edge restrictions remain in force.

## Qualification, 2026-11-20

The user approved an exclusive GPU maintenance window. Temporary pods used the
same digest-pinned production image and restored replicas and Flux afterward.

| Profile | Context | Coding output tokens/s, including prefill | Allocated GPU memory |
| --- | ---: | ---: | ---: |
| GGUF IQ4_XS + F16 adapter | 131072 | 34.27 | 18698 MiB |
| GGUF Q4_K_XL + F16 adapter, NVMe | 131072 | 27.12 | 21692 MiB |
| vLLM original adapter, BF16 KV / FP32 state | 98304 | 33.97 | 22718 MiB |
| vLLM original adapter, MTP3 / per-head int8 KV / FP32 state | 131072 | 111.43 | 22734 MiB |

These are single-run local checks on the existing stock/300 W GPU profile,
not statistically significant speed rankings or a reproduction of the card's
SWE-bench/Terminal-Bench scores. vLLM was selected for the original adapter,
standard recurrent-state precision, verified prefix reuse and Pi integration.

The MTP profile passed exact output, JSON arithmetic, five independent Fibonacci
cases, a complete tool-call/result exchange, streamed tool-call reconstruction,
and an actual Pi parser-editing session with five independent checks. It measured
111.43 output tokens/s including prefill on the coding probe and 278.60 aggregate
tokens/s across four simultaneous 128-token responses. The deployed one-slot
baseline measured 33.47 aggregate tokens/s on that four-caller probe.

It retrieved an identifier from a 120,042-token prompt in 101.89 seconds. The
repeat reused 117,216 prefix tokens and took 4.14 seconds. The Pi editing task
took 18.28 seconds. These timings include the corresponding prompt processing.

Initial compilation exceeded the trial's ten-minute startup probe immediately
before HTTP became ready; production already permits fifteen minutes. The rerun
used the saved compilation and completed every acceptance gate. This initial
build cost is separate from the warm inference numbers above.

The earlier FP16-state profile also passed the long-context and Pi checks.
The initial HDD-backed Q4 load timed out; the completed NVMe retry is the one
reported in the table. All reported GGUF base downloads matched upstream hashes.

The reused base is a local derivative: its first five weight shards matched
original Hub download metadata; the remaining embedding/head/MTP artifacts were
intentionally repacked to int8. The preserved preparation receipt and serving
hashes are in `pi-base-preparation.log` and `pi-base-verification.json`. Original
Hub checksums are ancestry records for those transformed files, not their current
checksums.

Raw responses and timings are retained in `pi-qualification.json`,
`pi-q4-nvme-qualification.json`, `pi-vllm-qualification.json`,
`pi-vllm-targeted-qualification.json`, `pi-vllm-96k-qualification.json` and
`pi-vllm-96k-fp32-qualification.json` and `pi-vllm-mtp-pth-qualification.json`.
The deployed MTP profile passed authenticated API requests and independently
checked Pi edits from Studio, Pro, Ravn and the Linux server. All four used the
shared Tailnet provider with 131072 context tokens. Their concurrent Pi tasks
completed in 15.82–32.64 seconds; each edited a real temporary file and passed
five independent behavior checks. Fixtures and Pi sessions were removed.

The production endpoint measured 98 ms to first output token and 95.08 output
tokens/s for a 256-token stream, including prefill. Four simultaneous 128-token
responses measured 333.28 aggregate tokens/s, compared with 33.47 in the deployed
one-slot baseline. These are single-run checks; prompt content and load affect
speculative acceptance and throughput. The startup-preloaded worker answered the
warm-up request after 7.53 seconds of waiting; this is not full pod startup time.

`pi-machine-verification.json`, `pi-deployed-performance.json` and
`pi-deployed-warmup-fast.json` retain the actual production receipts. The
relocated Orca serving files were checked against 74 existing production hashes
in `pi-orca-stage-verification.json`; their archive remains intact. Both existing
Orca routes passed real arithmetic requests after relocation, and Pi was restored
to the resident worker afterward (`pi-existing-model-verification.json`).
Reloading Pi after the Orca checks took 135.58 seconds, including worker startup;
keeping Pi resident avoids that swap cost.
`pi-production-status.json` records the final deployment/image pins, reconciled
and unsuspended Flux state, GPU/disk usage and Prometheus receipt.


## Context expansion, 2026-11-22

The four-slot, 95% GPU profile retrieved the expected identifier from 171,766
prompt tokens in 172.18 seconds. A follow-up reused 169,488 tokens and completed
in 4.56 seconds. Eight callers forked two warm documents and recovered the
correct document and branch identifiers, with 31,360 reused tokens per caller.
The streamed tool-call and tool-result exchange passed.

On the same fixtures, the 168K candidate and the 128K control delivered 152.6
and 142.6 aggregate output tokens/s for eight callers sharing a 49.6K prefix,
and 87.4 and 96.5 tokens/s on the single short stream. These are single-run
observations. Both served the mixed cold-prefill workload in about 36 seconds.

Eight admission slots improved the short eight-caller burst but increased
shared-prefix replay under cache pressure. The 97% GPU candidate ran out of
VRAM during concurrent prefill. The selected four-slot profile had 977 MiB
free after startup in the temporary worker.

Run the probe from the repository root on `denguinho-server`. Confirm the home
node and Pi worker first. If the worker is idle or another model is loaded,
make a regular authenticated Qwen Pi request through llama-swap and wait for it
to complete.

```sh
test "$(hostname -s)" = denguinho-server &&
  ~/.local/bin/kubectl get node denguinho-server &&
  ~/.local/bin/kubectl -n apps exec deploy/llama -c llama-swap -- \
    curl --fail --silent --show-error --max-time 10 http://127.0.0.1:5894/health &&
  ~/.local/bin/kubectl -n apps exec -i deploy/llama -c llama-swap -- \
    /app/venv/bin/python - --context 172032 --callers 8 \
    < benchmarks/qwen-pi-cache-2026-11-20/acceptance_probe.py
```

The API advertises the model window in `/v1/models`. The static
`cliproxyapi/local/qwen3.8-pi` Pi entry reads `contextWindow` from the shared
dotfiles. Update that entry after the new server limit is live.

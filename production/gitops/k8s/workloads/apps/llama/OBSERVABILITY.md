# SparseKV normal-use observations

The resident model runs through `serve_observer.py` and the pinned SparseKV
launcher. The GPU profile is defined in
[resources and alerts](../../../../docs/resources-and-alerts.md#gpu-rtx-3090).
The observer samples every ten seconds and leaves model recovery to the
operator. Normal llama-swap termination reaches the model's process group.
Observation and storage failures leave inference running.

## Evidence

The existing cache PVC holds `/cache/sparsekv-observations/current/status.json`
and a timestamped directory for each model launch. Samples include running
and waiting requests, cumulative prompt/output tokens, cache counters, GPU
utilization, memory, clocks and power, cgroup memory events/pressure, and host
available memory/swap. The recorder stores metadata and counters; request
prompts, generated text and environment dumps stay outside its evidence.

Four rolling JSONL files retain roughly 1 MiB of recent samples. Each capture
copies those files and adds the current metrics response, process names/PIDs,
RSS, state and wait channels, and memory state. Copies remain independent of
later rotation. Up to eight captures are retained per launch. The 64 MiB
budget includes prior launches; earlier captures are preserved when recording
reaches the budget. Budget checks occur every minute and before captures, so
one capture or rotation can overshoot the sampled limit. Status updates
continue, exposing that recording needs attention.

Automatic capture triggers when:

- Active or waiting requests show unchanged token counters for 240 seconds.
- A previously available metrics endpoint stays unavailable for 60 seconds.
- Startup reaches 900 seconds without valid progress metrics.

Each transition triggers a capture; recovery allows a later event. These are
suspected stalls: a long prefill can also leave counters unchanged. Repetitive
generation can keep counters advancing, so the history helps distinguish it
from a stopped engine when the user reports a loop. Full CUDA traces are an
optional focused diagnostic described in [CUDA-TRACING.md](CUDA-TRACING.md).

For an operator-requested capture inside `llama-swap`:

```sh
touch /cache/sparsekv-observations/current/capture
cat /cache/sparsekv-observations/current/status.json
```

The next sample records `last_capture` with its path and reason. Preserve that
capture and corresponding pod logs before a restart. A GPU telemetry command
has a three-second deadline plus one second to reap it; an unresponsive probe
stays the only outstanding GPU probe. Diagnostic collection performs no GPU
reset, debugger attachment or automatic model restart.

## Prometheus

The existing `vllm-metrics` sidecar reads status through a read-only cache
mount and appends these metrics to its existing `/metrics` response:

| Metric | Meaning |
|---|---|
| `sparsekv_observer_up` | Fresh observer status (at most 30 seconds old) with a live target. |
| `sparsekv_observer_sample_age_seconds` | Time since the last status update. |
| `sparsekv_observer_suspected_stall` | A fresh status identifies a suspected stall. |
| `sparsekv_observer_stalled_seconds` | Time since the last observed token change or idle sample. |
| `sparsekv_observer_captures_total` | Captures in the current model launch. |
| `sparsekv_observer_storage_budget_reached` | The observation directory reached its retention budget. |

The existing PodMonitor scrapes these metrics. Reading them keeps model
selection unchanged, including while OCR or image generation uses the GPU.

## Rollback

The previous DFlash image and resident model block remain in Git history at
`e52faa3454da19a3e7f860dfd55827c127444e21`. A retry rollback restores that image
and model block while retaining the stock-clock, 300 W GPU profile. Apply it
through GitOps and verify authenticated inference. The cache PVC retains the
evidence across pod replacement.

## Saved prefix warming

The observer starts `warm_saved_prefixes.py` alongside the backend. If the
private cache file `/cache/sparsekv-warm-prefixes.json` exists, it checks the
model path and tokenizer SHA-256, waits up to 15 minutes for backend health,
and warms up to two exact token prefixes with `sparse_pin_prefix=1` and one
output token. Each prefix ends on an 864-token boundary. The backend log
confirms the retained host KV blocks and GDN checkpoint; the warmer logs token
counts, cache usage and elapsed time. Optional warming failures leave inference
and observation running. Model termination also reaps a pending warmer.

Capture two independent current client payloads and use
`optimization/prefix-mtp/warm_prefix.py` against the loopback backend to find
their exact shared prefix. Its tokenizer call carries the chat request's
reasoning settings. For captures taken before a gateway, use
`--reasoning-effort` to match its actual backend effort. The current CLIProxy
model definition advertises low/medium/high and maps Pi's off request to low;
its saved profiles are low and high. The live Pi route is part of qualification.
Keep payloads, token IDs and the saved bundle private;
tokens can reconstruct prompt text. Refresh the bundle after changing the
client's tools, prompt structure or reasoning profiles. The bundle has keys
`model`, `tokenizer_sha256` and `prefixes` (one or two arrays of token IDs).
It is generated machine-local data, outside Git. Each new backend process
rewarms it; the engine retains at most two prefixes.

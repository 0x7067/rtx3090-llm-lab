# Optional SparseKV launch-time CUDA tracing

Normal use runs the lightweight [observer](OBSERVABILITY.md). For a focused
CUDA investigation, add `cuda_trace.py` to the chat-template ConfigMap files
and change the resident command to `/usr/bin/env PORT=5890 /app/venv/bin/python
/chat-template/cuda_trace.py` through GitOps. The image contains Nsight Systems
2026.4.1.
The 16 GB host cache and normal asynchronous/CUDA graph execution remain enabled.

The recorder injects CUDA tracing when the model process starts and follows
its spawned workers. It records individual CUDA graph nodes, kernel launches,
copies and CUDA API calls. CPU sampling, CPU context switches, CUDA event
tracing, and tracing between fork and exec are disabled. Reports discard
environment variables. The target writes directly to the pod log stream, so
Nsight does not accumulate another copy of serving logs.

## Storage and lifetime

`/cache/sparsekv-cuda-traces/current/status.json` identifies the active session,
current report, latest counters, and recorder state. `/cache` is the existing
`llama-runtime-cache` PVC; reports and Nsight's raw temporary files survive a
pod replacement. Each launch gets its own private timestamped directory.

Collection rotates every 30 seconds. Up to 12 completed windows are retained
(roughly six minutes). Stopping/finalizing and restarting collection creates a
short gap between windows. CUDA buffers flush to persistent temporary storage
every second. An unfinished kernel may lack a completed activity record; API
launches and earlier completed work still help localize the stall.

The 2 GiB total storage budget includes earlier launches and raw data. Healthy
windows from the current run are pruned oldest-first, keeping at least two.
Earlier runs and raw data are preserved. If retained evidence fills the budget,
new recording stops and status becomes `frozen`. This is a sampled budget, not
a filesystem quota: one in-flight window/finalization can exceed it. The model
continues serving. Archive reviewed evidence and remove its run directory to
make room before the next model launch. The existing 36 GiB container cap also
covers the profiler; its memory and CPU use are checked during qualification.

Recording freezes when active requests make no token-counter progress for
240 seconds, when an operator creates `current/freeze`, or when profiler
control fails. A long prefill may also trigger the progress condition; the
marker describes a suspected stall rather than asserting a GPU fault.
A control command gets 20 seconds. A timed-out command leaves raw evidence
intact and triggers no further automatic profiler calls. After a successful
stop the profiler detaches with `--kill=none`; the model keeps running.
Normal model shutdown finalizes and preserves that launch's evidence too.

## Capturing and inspecting

From the running `llama-swap` container:

```sh
cat /cache/sparsekv-cuda-traces/current/status.json
touch /cache/sparsekv-cuda-traces/current/freeze
```

Inspect the status again after finalization. Preserve the entire timestamped
directory, including `raw/`, `status.json`, and any `.nsys-rep` reports, before
restarting a hung model. `nsys stats --report cuda_gpu_kern_sum REPORT.nsys-rep`
shows captured kernels. Nsight's timeline links CUDA API calls, streams and
graph nodes. Raw `.qdstrm` files may be recoverable with `nsys import` if report
finalization was interrupted. Never attach CUDA-GDB to this shared host's
stalled engine; the previous attachment coincided with a host-wide freeze.

Launch-time instrumentation can change timing and has overhead. Successful
qualification demonstrates capture on this configuration; it cannot guarantee
recovery from every GPU/driver failure. The recorder performs no GPU reset or
automatic model restart.

## Disable

Restore the resident model's command in `swap-configmap.yaml` to
`/usr/bin/env PORT=5890 /app/venv/bin/python -I -S /chat-template/serve_observer.py`
through GitOps.
Reconcile and verify real inference. Existing evidence stays on the cache PVC.

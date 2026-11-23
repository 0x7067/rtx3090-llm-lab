# SparseKV rollout memory diagnosis — 2026-11-11

The production trial passed text, streaming, tools, JSON, image input, four
31,537-token clients, and cancellation recovery. A global OOM occurred during
the subsequent near-189K request. PR #218 restored DFlash.

Historical Prometheus process-exporter samples distinguish model memory from
the competing load. Times below are America/Sao_Paulo (UTC−03:00):

| Time | Engine RSS | vLLM API RSS | workerd group RSS | Host available |
|---|---:|---:|---:|---:|
| 11:18:30 | 20.07 GiB | 2.07 GiB | 0 GiB | 13.60 GiB |
| 11:20:00 | 20.04 GiB | 2.16 GiB | 7.80 GiB | 5.39 GiB |
| 11:21:00 | 19.91 GiB | 2.13 GiB | 8.97 GiB | 1.93 GiB |
| 11:24:00 | 19.82 GiB | 2.11 GiB | 10.41 GiB | 1.97 GiB |

The kernel's 11:22:48 OOM report had only 176 KiB swap free out of 16 GiB.
Its process table reported the SparseKV engine at 5,195,438 resident pages
(~19.82 GiB), matching Prometheus. The available pod/container memory metrics
returned no historical series, so attribution uses process RSS and the kernel
snapshot. RSS totals can include shared pages; these are not per-process PSS.
The workerd growth is a substantial competing load, and the evidence does not
establish that it explains every byte of the host decline. It does establish
that engine RSS stayed near its post-startup level during this interval.

The old monitor sampled available memory every ten seconds. It recorded the
decline below 5 GiB at 11:19:43 and below 3 GiB at 11:20:33; the first global
OOM was at 11:22:47.

The adoption retry retains the 12 decimal GB host KV pool, 200K context, four
admissions, original pod limits, and upstream allocation check. It uses the
upstream launcher directly. Live qualification must cover the near-window
request and follow-up while recording process and cgroup memory. Competing
development workloads should be accounted for when assessing host headroom.

Evidence: `process-memory.json`, `host-memory.json`, the saved rollout's
`live-production12gb/live.jsonl`, and the kernel journal for 11:18–11:25.
The service rollout requires the separately approved maintenance window.

A later host process inspection found one workerd (~0.5 GiB RSS) launched from
the pi-nimbus development worktree under rex.service. This identifies the
current instance. The historical process-exporter samples aggregate by process
name and do not identify the project responsible for each earlier worker.

The retry keeps the serving path on the pinned upstream launcher, with its
existing allocation check and the original pod resource limits.

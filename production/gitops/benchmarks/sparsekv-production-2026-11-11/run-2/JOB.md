Target: compare production HyperQwen DFlash7 with SparseKV MTP7 on the same RTX3090, then restore production.
Command: harness/start.sh --execute /data/development/worktrees/.treehouse/docker-services-3c01ca/5/docker-services/benchmarks/sparsekv-production-2026-11-11/run-2
Owner: system unit sparsekv-production-bench.service, invocation b9d4697f393d433bb33a9bd457780e1e.
Start: 2026-11-11 09:33:09 America/Sao_Paulo.
Completion: both arms terminate and original production deployment returns with verified inference; results may include failures.
Poll interval: 30–45 seconds, using ../status.sh, phase files and bounded server-log tails.
Deadline: 10:33:09 measurement owner; 10:43:09 includes restoration timeout.
Expected report: paired speed/latency, quality and long-context results, raw samples, resource telemetry, any partial phases, and production restoration receipt.
Recovery: owner remains responsible; ExecStopPost invokes scoped cleanup and restoration automatically. No candidate promotion.

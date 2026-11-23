Target: measure SparseKV against the completed run-2 baseline and restore production.
Command: SPARSEKV_RUNTIME_SECONDS=1800 harness/start.sh --execute <run-3> <run-2>
Owner: sparsekv-production-bench.service, invocation dfdd844b515c49b4bce18e71a3b2f842.
Start: 2026-11-11 09:57:25 America/Sao_Paulo (systemd journal is authoritative).
Poll: status.sh, phase files and bounded logs every 30–45 seconds.
Deadline: 30 minutes of candidate work plus 10 minutes of restoration, within the original 10:43 approved limit.
Completion: candidate phases terminate, original production inference verified, temporary image pins restored.
Baseline: real copy of run-2/baseline; trial ID preserved to make long-context request hashes identical.
Expected report: matched saved samples, failures or partial phases, resource telemetry and restoration receipt.

# Spark coding settings pilot

This directory records a settings-first experiment on the served Spark X2.5
4B Q8_0 model. Read `REPORT.md` for results and `EXPERIMENTS.md` for the
selection history. `LORA.md` and `lora-pilot.yaml` prepare a later training
experiment; no weights were trained here.

## Workload and interpretation

The paired harness supplies synthetic Python behavioral contracts and frozen
tool-call transcripts. A deterministic split, made before inference, assigns
14 coding cases and six two-step tool episodes to development. Six coding
cases and two two-step tool episodes are reserved for a final comparison.
The split unit is a coding task or an entire tool episode, not an individual
message. Reference implementations passed their grading assertions.

Each screening request has seed 42 and an 8,192-token total output budget.
This budget is deliberately smaller than the service's 32,768-token output
allowance. Truncations count as failures. The separate 32k diagnostic changes
only that allowance for one development task; it is not a full 32k benchmark.
One sample per task cannot establish general coding accuracy or statistical
equivalence. These fixtures are not HumanEval, SWE-bench, or real repository
issues. Latencies describe serial requests to a warm shared endpoint.

`agent-suite.json` adds three synthetic file-editing episodes with live tool
feedback. The model reads in-memory source and visible tests, writes the source,
and invokes the sandboxed test runner. Separate immutable final assertions grade
the resulting source. Code cannot access host files or the network. Success of
the final source and completion of the read/write/test workflow are recorded
separately. API-time sums exclude Docker/tool overhead.

## Reproduce

Run from this directory, with Docker and the paired harness's pinned Python
image available. The recorded endpoint is internal to this cluster. For a new
endpoint, update an arms file and freeze a new plan; do not edit a hashed plan.

```sh
python3 ../paired-harness/harness.py run \
  --plan artifacts/plans/plan-dev-publisher.json \
  --arm spark --out /tmp/spark-coding-repeat-publisher --timeout 180

python3 ../paired-harness/harness.py summarize \
  --plan artifacts/plans/plan-dev-publisher.json \
  --run /tmp/spark-coding-repeat-publisher \
  --out /tmp/spark-coding-repeat-publisher-summary.json

python3 agent_smoke.py --suite agent-suite.json \
  --arms artifacts/arms.json --budget -1 \
  --out /tmp/spark-agent-repeat-publisher

python3 agent_smoke.py --suite agent-suite.json \
  --arms artifacts/arms.json --budget -1 --no-thinking \
  --out /tmp/spark-agent-repeat-no-thinking

python3 -m unittest -v test_agent_smoke.py
python3 summarize_pilot.py > /tmp/spark-coding-analysis-repeat.json
```

Keep GPU inference runs serial. Output directories must be new. The existing
harness's `compare` command requires identical frozen plans and is inappropriate
for these deliberate request-policy changes. Summarize each plan separately,
then pair observations by case ID while verifying that only the stated setting
or system instruction changed. Include every attempted case, including failed
requests and incomplete output. Do not assign a full-suite score to an
interrupted diagnostic.

## Evidence

`artifacts/plans/` preserves frozen requests, graders and source hashes.
`artifacts/runs/` preserves manifests and gzip-compressed raw JSONL responses,
including reasoning, SSE events, final code and failures. Decompress a copy
before using the original harness to summarize an archived run. The agent
manifests also pin the runner hash and complete fixtures. `split.json` records
the original development/holdout partition. Recorded data is synthetic;
authentication headers are never included.

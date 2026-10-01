# Speed and Strata quality follow-up

Research and experiment selection are in [RESEARCH.md](RESEARCH.md). The first
quality screen keeps the tested Strata artifact, runtime, context, KV type, GPU
profile, output allowance, and graders fixed. It changes one request policy at
a time.

`build-followup-plans.py` creates five frozen 36-observation plans:

- corrected control;
- 512, 1,024, and 2,048-token hard reasoning budgets for coding cases;
- nonthinking sampling for extraction and relevance cases.

The subset contains the six original misses and six passing guard cases,
including both steps of one tool replay. Three fixed seeds expose sampling
variance. Two source
contract defects are corrected in every plan: `code-redact` now says that
`sensitive` is a list, matching its executable grader, and `extract-quoted`
names the literal JSON key `value`. These plans form a new policy campaign and
must not be used to rewrite the frozen 48.67/54 baseline.

Generate the plans from the campaign directory:

```bash
UV_CACHE_DIR=/tmp/uv-cache UV_PYTHON_INSTALL_DIR=/tmp/uv-python \
  uv run --python /usr/bin/python3 --no-project python build-followup-plans.py
```

With the pinned Strata server listening on `127.0.0.1:18182`, run each plan into
a new directory:

```bash
for policy in control budget-512 budget-1024 budget-2048 nonthinking-simple; do
  UV_CACHE_DIR=/tmp/uv-cache UV_PYTHON_INSTALL_DIR=/tmp/uv-python \
    uv run --python /usr/bin/python3 --no-project python \
    ../paired-harness/harness.py run \
    --plan "followup-plans/plan-${policy}.json" \
    --arm strata_flash_next \
    --out "followup-runs/${policy}" \
    --timeout 1800
done
```

Retain a policy only when it fixes its target failures across seeds and does not
regress the guard cases. Then apply that policy to a corrected full suite and
rerun all compatible arms. Speed experiments begin only after reproducing the
existing Strata performance probes three times. Calibration, a suite-derived
expert profile, and the reduced draft vocabulary are separate one-variable
arms with fresh servers and unchanged requests.

The current managed execution sandbox denies Docker, k3s, user systemd, and
local network sockets. No follow-up inference result has been measured in this
environment.

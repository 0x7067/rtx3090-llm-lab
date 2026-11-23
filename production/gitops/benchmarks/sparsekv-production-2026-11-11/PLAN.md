# SparseKV versus production — prepared comparison

Status: completed. Both arms passed all phases; production restoration and cleanup are recorded in final-verification.json. See REPORT.md for the result.

Compare the installed production HyperQwen DFlash2-7 stack against
https://github.com/bradj00/sparsekv-hyperqwen using the same existing
Qwen3.8-27B-W4A16-AutoRound weights (INT8 lm_head), chat template, fixtures,
and sampling. The sparse upstream uses MTP-7 and a different HyperQwen/PTH
implementation. This is a whole-stack comparison; any observed difference
cannot be attributed solely to sparse attention.

Production: 200000 configured context, four admissions, INT8 PTH attention,
7.25 GiB KV pool and 16 GiB CPU prefix tier. Live deployment and configuration
were captured in `production/`. No production credentials are captured.
Both arms retain the installed 290 W GPU profile and use 12 CPUs/36 GiB RAM.

## Initial screen

The `harness/` runner reuses repository validation fixtures for three 4K and
three 32K speed samples, smoke/tool continuation and a small executable
code/tool/JSON quality screen. Long-context fixtures exercise approximately
124K tokens: A cold, A follow-up, B cold, A return and two concurrent warm
sessions. Both arms receive the same trial identifier and request hashes.
Native server timings are reported only when actually available. Client
first-token latency, end-to-end time, output count, prefix reuse, errors,
truncation and correctness remain separate fields.

The target is a bounded 60-minute measurement window including model
starts, plus up to 10 minutes reserved for production restoration. Long cold prefill and failed cache reuse may exhaust the bound;
partial samples are retained and clearly labeled, not treated as a win.
This screen does not establish four-session 250K capacity, statistical
significance, full production readiness or quality equivalence.

## Execution boundary

Before GPU execution: candidate CPU checks pass, production recovery is
complete, no other GPU owner is active, and the user has approved the
interruption. Their AGENTS.md requires approval before disrupting shared
services. The single GPU cannot host both arms alongside the resident model.

The run will pause only the llama deployment, measure the arms sequentially
with fresh private writable caches and read-only model mounts, then restore
its original replica count and reconciliation state. A bounded service and
restoration handler must be armed before stopping production. Verify the
original image/config, ready replica, and an actual authenticated inference
after restoration. No candidate promotion is part of this benchmark.

## Pre-existing availability issue

At initial inspection the production deployment had one desired and zero
available replicas. Its pod was evicted during image pulling due to disk
pressure after another task's GPU lease. That task's restoration units are
active; this benchmark has not changed them. Building a small overlay reuses
cached image layers and avoids large image downloads. Do not start GPU work
or claim a production baseline until recovery and ownership are verified.

Recovery update: the existing owner completed restoration at 08:47:16 local
time; its authenticated request returned `42` with `finish_reason=stop`.
The replacement pod has 2/2 containers ready, zero restarts and node
DiskPressure=False. Candidate CPU build and both launch-argument checks
passed; GPU compatibility and benchmark results remain unmeasured.

## Authorized launch attempt

The user approved the service interruption. The first detached launch exited
in preflight because its systemd environment omitted KUBECONFIG and kubectl
fell back to root-only `/etc/rancher/k3s/k3s.yaml`. No ownership claim or
production scaling occurred. `start.sh` now passes the existing user kubeconfig
explicitly. The captured journal is `run-1/launch-failure.log`.

A retry then refused before launch because available host memory had fallen
below the existing 12 GiB pre-handoff minimum. The threshold remains intact;
production remains at one available replica. No benchmark measurements exist
yet. The resume point is the read-only preflight, then the already approved
bounded runner when host memory satisfies its requirements.

## Corrected replacement-memory check and active run

The original 12 GiB before / 36 GiB after handoff gates mistakenly treated a
container maximum as a required reservation. The replacement check measures
production pod resident anonymous and shared memory (excluding ordinary file
cache already counted in MemAvailable), projects at least 30 GiB after stop,
and then verifies that amount after stop. The 30 GiB budget includes the
candidate 17.8 decimal-GB pool, its unchanged 8 decimal-GB allocator reserve
and about 6 GiB for runtime overhead. The candidate's native allocation guard
and both 36 GiB container caps remain unchanged. A one-second monitor aborts
the run below 3 GiB host MemAvailable and invokes the existing restoration.

Run-2 passed projected and actual handoff checks and started the baseline.
The prepared /dev/shm ceiling was corrected from 16 to production's 24 GiB
in both launch modes. The already-starting disposable baseline mount was
remounted to 24 GiB before inference; /proc/863143/mountinfo confirmed
size=25165824k in its distinct mount namespace. Historical CPU receipts show
the original 16 GiB preparation; this runtime correction supersedes that size.

## Image GC recovery and candidate-only retry

Run-2 completed every baseline phase. Kubelet image GC crossed its configured
80% image-filesystem usage threshold and removed the production images while
the Deployment was scaled down (the standalone ctr task was not a CRI pod).
Candidate startup then failed before inference with image-not-found.
The restore check exceeded its 9-minute verification deadline during image
unpacking. Kubernetes kept the original replica count and completed startup;
a separate bounded verifier obtained the expected answer 42 before run-3.

Run-3 preserves run-2 baseline results with cp -a and reuses its trial ID.
All three deployment image digests are temporarily retained with
io.cri-containerd.pinned=pinned, with CRI confirmation before stopping
production. Prior labels are recorded and restored after verified inference.
The retention mechanism was tested live and rolled back before run-3.
See https://github.com/containerd/containerd/discussions/12156 for the
maintainer's description of this kubelet GC flag. No global image-GC policy
or host resource cap was changed.

## Harness review corrections (2026-11-18)

The maintained restoration command retries API reads and patches within its
existing nine-minute deadline. Every retry reads the Deployment again and
checks its UID and owner before acting; an ambiguous patch timeout can therefore
complete through verification without repeating an already-applied patch.
Ownership conflicts stop restoration before task cleanup or Deployment changes.

New arm runs record `trial.json` beside their results. Reused baselines carry
that metadata forward. For legacy runs, `baseline_trial.py` follows the recorded
baseline source chain and verifies the trial against the saved cold-request
hash before production is claimed. Archived absolute paths can resolve to
sibling run directories; the request hash still has to match.

The regression checks inject initial API read failures, patch failures before
and after application, UID/owner conflicts, and repeated baseline reuse using
independent copies of the archived data. They run without cluster access.
Historical measurements and `run-3/runtime-source` remain unchanged snapshots
of the original run; these corrections affect future harness use.

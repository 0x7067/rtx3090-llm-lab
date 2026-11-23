# Experimental short DFlash launcher gate

Built and CPU-qualified here; deployed as the llama-swap image (see `../../deployment.yaml`). See `cpu-qualification/` for the exact digest and successful receipt; those receipts name the pre-move path `optimization/normal-120/dflash-short` and are kept unchanged. Base image is the original deployed `home-llm-strata:0.2.8-v1@sha256:b2b21d86b3926f564499b7415d4f0afae9b2dddd7348ec7c5082c5077cbd8c42`, **not** the rejected custom FP16-kernel image. No model weights, runtime kernels, sampling parameters or service defaults change.

`upstream/` is a read-only capture of the installed launcher's six source/license files; `provenance.json` records their SHA256 hashes and inspected runtime version. The production launcher SHA256 is `e8f4d78344b94410c9ea0a9a16636c6a1e38cd26e82c7ce47840d186b541dcb5`. The patched script is `2f76802a7624cfc1ccadba0ef61f5c362627c34fa341c700c0eab27b521727cd`.

`apply_patch.py` refuses any different source hash or ambiguous anchor. Only the existing PTH DFlash draft-length guard changes: `NORMAL120_PTH_SHORT_DFLASH=1` permits exactly `DFLASH_TOKENS=3` or `5`. Without that flag the original behavior remains; k=7 remains the default. Other lengths still refuse on PTH. The existing lookup-on, DFlash-method, dynamic-K, context, backend-override and KV-connector guards remain. A conspicuous experimental notice accompanies an admitted short block. The flag has no effect on MTP or non-PTH arms.

The Dockerfile extends the pinned original image and patches its existing script in place with stdlib-only Python (`-I -S`), then checks Bash syntax. It installs no packages and executes no GPU/runtime imports. Build context excludes captured source/test/provenance files; only the patch program is copied. No build is authorized by the presence of this recipe.

## Why this remains experimental

Installed `/app/single-user/start_qwen.sh:317–333` refuses non-seven draft counts because they are unmeasured with PTH. The underlying `/app/pthfa/python/pth_flash_attn/vllm_backend.py:112–123` sizes verify workspace as `1+k`, and vLLM's `model_executor/models/qwen3_dflash2.py:250–263` sizes grouped convolution as `min(1+k, trained_block_size)`. `v1/worker/gpu/spec_decode/dflash/speculator.py:62–91` supports shorter query blocks. These observations justify a bounded experiment, not a correctness claim.

The drafter is noncausal; using fewer mask positions can change even its early proposals and acceptance. Target verification is intended to preserve the target distribution, but recurrent rollback, query geometry, CUDA graph replay and simultaneous request completion must be qualified. Shorter blocks automatically change graph capture size and recurrent/KV geometry through existing launcher/runtime logic. This is one configured draft-length change, not a promise that only one derived allocation changes.

## CPU validation

Run from the repository root:

```sh
uv run --no-project python k8s/workloads/apps/llama/image/strata-dflash-short/check_cpu.py
```

The test uses disposable source copies, an explicit clean environment, `VLLM_OFFLOAD_KEEP_SHM=1`, stub GPU commands, a Python stub that accepts only the package-presence probe, and a vLLM argv/environment stub. It never imports Torch/vLLM or starts a server, uses no real credentials, and removes its temporary directory. It verifies source hashes, drift refusal, unchanged default stdout/stderr, opt-in k3/k5 argv, 150K/four slots/6GiB/int8 KV/vision/UVA, unchanged other-arm behavior and retained negative guards. Passing establishes launcher behavior only.

## Five prepared hidden profiles (not applied)

The ConfigMap now contains five clones of the existing hidden HQ DFlash150K entry, with defaults and preload unchanged:

| Hidden name | Port | Changes relative to HQ DFlash150K |
|---|---|---|
| qwen3.8-27b-hyperqwen-trial-dflash150k-k5 | 5902 | `DFLASH_TOKENS=5`, `NORMAL120_PTH_SHORT_DFLASH=1` |
| qwen3.8-27b-hyperqwen-trial-dflash150k-k3 | 5903 | `DFLASH_TOKENS=3`, `NORMAL120_PTH_SHORT_DFLASH=1` |
| qwen3.8-27b-uncensored-trial-dflash150k | 5901 | Keep k7; explicit MODEL pointing at existing uncensored draft40k directory; no opt-in needed |
| qwen3.8-27b-uncensored-trial-dflash150k-k5 | 5904 | Same uncensored MODEL, `DFLASH_TOKENS=5`, `NORMAL120_PTH_SHORT_DFLASH=1` |
| qwen3.8-27b-uncensored-trial-dflash150k-k3 | 5905 | Same uncensored MODEL, `DFLASH_TOKENS=3`, `NORMAL120_PTH_SHORT_DFLASH=1` |

Ports5901–5905 were verified unused in the serving pod network namespace on2026-11-06; recheck before invocation. Preserve TTL600, unlisted/disableAuto, probabilistic drafting, LOOKUP1, CUSTOM PTH attention, INT8 KV, MAX_SEQS4, 150K limit, KV_MEM as set in `../../swap-configmap.yaml`, EMBED_UVA1 and vision offload. The uncensored proposal and additional full-head memory risk are detailed in `../../optimization/normal-120/UNCENSORED-DFLASH.md`.

Any future image publication needs a unique tag and captured digest plus full supply-chain checks. Image deployment, profile additions and GPU trials remain separate coordinated steps. Start with k5 before k3, compare against matched k7 card-profile baseline, and retain thermal/Xid guards and bounded default restoration. Startup/150K allocation proof, vision/tools smoke, held-out quality, prefix-cache reuse and four-client completion/rollback checks are required before promotion. Neither this patch nor a faster speed screen satisfies those gates.

## Build and CPU image qualification

`run_build_cpu.py --execute` uses the existing `strata-local-registry` builder with network disabled for build steps, 2 CPU/8 GiB limits, a 30-minute build deadline and unique local tag `normal120-dflash-short-v1`. It rejects an existing local receipt or registry tag. A memory/PSI preflight leaves host headroom. The same owner then runs a nonroot read-only CPU pod in a disposable namespace with deny-all networking, no NVIDIA devices, no host mounts or service-account credential, tmpfs and a read-only ConfigMap containing only the reviewed check sources. The 240-second pod deadline, 300-second host polling bound and namespace cleanup are retained even on check failure.

The CPU receipt verifies exact installed patched-launcher bytes, Bash syntax, original installed vLLM wrapper hash, absent custom FP16 GDN plugin, default-off GDN gate and no Torch/vLLM imports. Isolated argv checks use the already reviewed stub approach (default k7 and explicitly gated k3/k5, SHM cleanup disabled). This tests actual installed source identity and its isolated launcher behavior; it never starts a model. The host hashes the registry manifest and checks the actual container image identity before accepting the receipt. Logs, source hashes, registry metadata and failure state remain under `jobs/`; the namespace is removed before complete status.

Only after the root coordinator explicitly releases the current screen, launch from the task worktree:

```sh
systemd-run --user --unit=normal120-dflash-short-build-v1 --property=RemainAfterExit=yes --property=RuntimeMaxSec=3000 --property=TimeoutStopSec=150 --setenv=PATH --setenv=KUBECONFIG --working-directory="$PWD" uv run --no-project python k8s/workloads/apps/llama/image/strata-dflash-short/run_build_cpu.py --execute
```

Monitor `sh k8s/workloads/apps/llama/image/strata-dflash-short/status.sh` every 45 seconds. Completion requires owner MainPID0/ExecMainStatus0, status complete/cleanup_complete, accepted `cpu-receipt.json` and absent temporary namespace. Publication is not serving deployment; no GPU execution is included. On build failure retain logs, diagnose, and use a new tag for a subsequent reviewed attempt rather than overwriting a published artifact.

SIGTERM/SIGINT use the normal cleanup path; a forced SIGKILL or host failure cannot execute Python cleanup. The coordinator must inspect and delete only `normal120-dflash-short-cpu` after verifying ownership if the host job dies and that namespace remains. This prepared runner does not mutate or restore serving.

## Profile validation and invocation gate

The23-model ConfigMap passes the installed llama-swap -validate command using a
disposable temporary config and placeholder API token. All18 original model
entries and every top-level setting (including the sole HQ preload) compare
semantically equal to the pre-change configuration. The source-only launcher
CPU dry run also passes with VLLM_OFFLOAD_KEEP_SHM=1, stub GPU commands and no
Torch/vLLM import. Temporary validation files are removed; no server or model
was started and no target weights/sampling were changed.

Root owns the separately prepared image pin and full policy/render checks.
No merge, deployment or invocation occurs until the active32K screen is terminal
and the default model is restored. Run HQ k5 before HQ k3 with matched k7
controls. The uncensored k5/k3 profiles may be invoked only after HQ short-block
integration passes; preparing their hidden entries is not permission to skip
that gate. All five additions remain pending runtime/quality qualification,
including context, vision, tools, cache and simultaneous completion/rollback.

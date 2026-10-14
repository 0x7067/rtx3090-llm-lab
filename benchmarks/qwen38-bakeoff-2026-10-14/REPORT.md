# Qwen3.8 bake-off — 2026-10-14

## Result

On base Qwen3.8-27B, **sglang-exl3 (EXL3 3.00 bpw) is the best all-round arm**: 54/54, near-vLLM single-stream speed, the fastest prefill, and vision. **HyperQwen vLLM** is the throughput choice (269 tok/s aggregate at 4 callers, ~3x the others) but is text-only and missed one case. Both llama.cpp arms are ~60-70% of their speed. Flash-Next on Strata (rerun) scored 50/54 with the fastest prefill of any arm (~1,750 tok/s at 64K-120K) and 57-71 tok/s decode, but needs ~31 GiB of host RAM and drove `MemAvailable` to 8.4 GiB.

Every cell is one run of one frozen synthetic suite. Single-case differences (53 vs 54) are not evidence of a quality gap.

| Arm | Status | Score | Short prose tok/s | Short code tok/s | 64K decode | 64K prefill | 120K decode | 120K prefill | 4-caller aggregate | Peak VRAM MiB | Min host avail GiB | Container peak GiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llamacpp_q4ks | ok | 53/54 | 57.83 | 67.16 | 51.54 | 838.6 | 40.71 | 663.8 | 48.32 | 22937 | 29.9 | 16.8 |
| llamacpp_gsq | ok | 54/54 | 58.17 | 66.75 | 48.57 | 827.8 | 38.74 | 656.3 | 50.12 | 22191 | 29.3 | 12.2 |
| vllm_hyperqwen | ok | 53/54 | 83.07 | 96.85 | **83.59** | 744.5 | **68.13** | 601.9 | **269.14** | 23704 | 33.2 | 13.1 |
| sglang_exl3 | ok | 54/54 | 80.45 | 96.72 | 73.36 | **1076.6** | 64.52 | **804.2** | 81.72 | 22564 | 30.0 | 23.7 |
| strata_flashnext_iq3xxs (rerun) | ok | 50/54 | 57.11 | 70.55 | 63.24 | **1750.3** | 66.39 | **1724.4** | 61.2 | 23869 | 8.4 | 31.1 |

Regenerate with `python3 -I aggregate.py` (saved as `aggregate.md`).

Misses: `strata_flashnext_iq3xxs` scored 0 on `code-env-lines`, `code-diff-keys`, `code-redact` and `extract-en-correction`; `llamacpp_q4ks` scored 0 on `episode-unicode-first`; `vllm_hyperqwen` scored 0 on `code-rolling-window`. All completed normally; the graded answers were wrong.

## Conditions

- Arms, pinned images and weight hashes: `arms.json`, `images.lock`, `weights.sha256`, `weights-hyperqwen.sha256` (one root-owned HyperQwen prepare output is missing from the latter). Setup notes: `README.md`.
- Frozen plan `plan.json`: the 2026-09-30 suite (54 scored cases + 4 perf probes), temperature 1.0, top-p 0.95, top-k 20, min-p 0, 16,384-token output cap. Only the grader image was re-pinned (the old one was gone).
- Speed probes from `speed-probe.py`; 64K/120K prompts sized with the Qwen3.8 tokenizer.
- GPU: RTX 3090, driver 595.91.07, the matched "cool" profile from 2026-09-30 (`gpu-cool-profile.sh`: 210-1350 MHz graphics lock, +100 graphics VF offset, stock memory offset, 370 W), thermal guard and fan curve active. No thermal trip. The normal profile was restored after both invocations.
- Run sequence: the main run (`run-bakeoff-main.log`, 12:40-13:59Z) skipped `llamacpp_q4ks` because an aborted normal-profile attempt had left a partial results dir (moved to `aborted-normal-profile/`); Q4_K_S was then run alone under the same profile (`run-bakeoff-q4ks.log`, 13:59Z). `run-bakeoff-failed-profile-exec.log` is a launch that failed before any arm started.
- Production llama was scaled to 0 for the duration and restored afterwards.
- Not verified: whether each engine honours `top_k`, `min_p` and `reasoning_effort` identically.

## Strata: first run failed, rerun passed

First run (moved to `aborted-normal-profile/strata_flashnext_iq3xxs-run1/`): Strata v0.1.44.1 served three requests, then its HTTP front end reset connections, so 52/54 cases and every speed probe failed; later the 12 GiB `MemAvailable` floor stopped the arm (11.8 GiB, PSI avg10 0.26). That was the first start after setup hashed and copied the IQ3_XXS files, on the same SSD as etcd.

Replaying the failing requests by hand against a fresh container, including the harness's `seed` and `stream_options`, did not reproduce the resets. The rerun (`run-bakeoff-strata2.log`, 14:30Z) used the same image and config with the floor lowered to 4 GiB (`MONITOR_FLOOR_KIB=4194304`, bake-off-local `monitor.sh`; PSI trigger unchanged) and completed every case and probe. Its low point of 8.4 GiB available would have tripped the old floor. The cause of the first-run resets is unconfirmed; first-start I/O is the leading suspect.

## Relation to earlier local results

The 2026-09-30 llama.cpp b11277 runs scored ThinkingCap 54/54 and GSQ 53/54 without speculation at ~37-42 tok/s short-context. Here GSQ on b11580 with MTP n=2 scored 54/54 at ~58-67 tok/s. These are different builds and settings, so treat the old numbers as context, not a baseline.

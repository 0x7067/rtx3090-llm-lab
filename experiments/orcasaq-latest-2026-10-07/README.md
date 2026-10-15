# Latest stable Orca engine qualification

2026-10-07. User requested latest engines plus Orca recommendations after selecting OrcaSAQ2 for two long coding-agent sessions.

Candidate uses vLLM0.31.0 + exllamav3 1.5.4 and current Orca source commit `7ddffb12c00c24b8bc6b1164723c04ac490a91bc`. Exact base digest/wheel hash/image ID are in `source-provenance.json`; actual Python/PyTorch/CUDA/package versions are in `runtime-versions.json`.

`PRESET=16gb-mtp` supplies MTP2, four sequences, batch2048, FP8 E4M3 KV, block32, prefix caching, Qwen reasoning/XML tool parsers. The only model-behavior configuration override is maxlen150000 for the previously requested long sessions. Server loopback port18185 and read-only model mount are test-specific.

Build:

```bash
docker build -t local/orcasaq-latest:2026-10-07-v1 experiments/orcasaq-latest-2026-10-07
```

Run from the lab root:

```bash
python3 -u experiments/orcasaq-latest-2026-10-07/run-qualification.py
```

The GPU runner temporarily suspends Flux apps and scales llama to zero. It saves state, uses host-memory/PSI/thermal guards, then removes the candidate and restores replica count/Flux/template in cleanup. It refuses to overwrite a completed window. Review `qualification/state-restored.json` before treating a run as complete.

The frozen 54-case quality suite, three-seed speed requests, 24-case two-caller coding panel and corrected 83K concurrent retrievals reuse prior request/grader bytes. Arm provenance differs. Latest engines and publisher settings both change, so this is a deployable package comparison, not causal attribution to a version. Startup and first-use JIT effects are separately recorded.

No shared-prefix checkpoint, custom fused kernel, precision change or GPU clock adjustment is enabled in this qualification. Those remain subsequent optimization experiments. This image is a direct Orca test server; it does not replace the production llama-swap/Humanlike/Clef wrapper.

Qualification completed: serial quality53/54 (SSE generator/list ambiguity), paired coding24/24, long-context retrieval8/8. Two ~83K cold histories took246.17s median; cached follow-ups8.28s. Previous tuned settings measured232.49s and8.15s. The latest/vendor configuration has251,886 reported shared KV tokens versus240,825 previously. Engine and preset changes are confounded. See `comparison.json` for exact observations and matched request-hash audits; rerun its collector with `python3 experiments/orcasaq-latest-2026-10-07/collect.py` from the lab root.

Saved deployment/Flux restoration succeeded with no failures. Public endpoint verification passed and is separately recorded in `qualification/restored-public-verification.json`. The full interpretation is in the parent services repository at `docs/research/orca-saq2-latest-engine-qualification-2026-10-07.md`.

## Subsequent serving tuning

The completed tuning window selected MTP2/batch2048 with `GPU_FRAC=0.92` and `--mamba-cache-mode align --prefix-match-unit 64`. The extra shared-prefix checkpoint flag remains off. See `memory-qualified/recommended.env` and `memory-qualified/results.json` for the final settings and acceptance evidence.

At the same 0.92 fraction, matched two-agent growing-session median follow-ups improved from11.24s to2.64s; the cached512-token-per-agent generation pair improved from26.42s to15.61s. Both profiles passed42/42 history checks; the selected profile passed24/24 coding and16/16 tool checks. Timings include residual prefill. A preceding 0.95 run crashed with a CUDA workspace allocation failure; the lower fraction passed that sequence and reported256,132 shared KV tokens.

`run-tuning.py`, `run-prefix.py` and their saved directories preserve the screens, checkpoint test, failed baseline and memory-safe qualification. `collect-memory.py` recomputes the final acceptance decision from saved observations. The detailed report is `docs/research/orca-saq2-serving-tuning-2026-10-07.md` in the parent repository. The original public service was restored; this direct test image still needs integration into the production wrapper before rollout.

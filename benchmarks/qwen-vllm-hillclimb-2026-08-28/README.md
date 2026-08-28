# Qwen vLLM hillclimb, 2026-08-28

This campaign investigated an apparent Qwen3.8-27B slowdown on the production
RTX 3090 profile. The frozen long-context control measured 81.49 tok/s. None of
the experimental runtime changes cleared the promotion gates.

The accepted fix was operational. Two orphaned benchmark shells had consumed
about 29% CPU each for 39 hours. Removing them restored the short suite from
74.25 to 104.36 tok/s. Long-context decode remained GPU-bound at 80.89 tok/s.

[`decision.tsv`](decision.tsv) is the index for all 16 attempts. The JSON files
are the raw outputs named by each decision. The implementation prototypes live
under [`../../experiments/`](../../experiments/).

These files record a completed experiment. They do not change the production
configuration.

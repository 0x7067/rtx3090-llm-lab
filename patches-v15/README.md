# patches-v15 (2026-09-03)

Base: llama.cpp master `0f3a71be1` (unchanged from v14). 0002-0008 are the
v14 kernel patches, byte-identical to `patches-v14/`. New in v15:

- `0009-k2-horizon-arch.patch`: the K2 Horizon architecture (IFM, MBZUAI).
  Squashed from the five commits on `MBZUAI-IFM/llama.cpp` branch
  `model/K2Horizon` (490c41950..35999d101, 2026-08-29..09-01), cherry-picked
  onto 0f3a71be1 with no conflicts. Adds `LLM_ARCH_K2_HORIZON`, the
  `k2-horizon` BPE pre-tokenizer (pre type 57), grouped RMS norm, MoVA
  hparams (unused by the dense 7B), the HF->GGUF converter and the
  `models/templates/k2-horizon.jinja` chat template. Upstream PR was still
  "in progress" per the IFM GGUF model card on 2026-09-03; drop this patch
  once master has it.

Apply order is lexical (`git apply patches/*.patch`); 0009 touches no file
the kernel patches touch. Verified 2026-09-03: all seven apply cleanly in
either order.

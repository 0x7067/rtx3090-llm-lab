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

- `0010-effort-specific-reasoning-tags.patch` (v16/v17, 2026-09-03/04): the
  differential auto-parser learns one reasoning tag from the template's
  default kwargs. K2 Horizon opens `<ifm|think>` / `<ifm|think_fast>` /
  `<ifm|think_faster>` depending on `reasoning_effort`, so medium and low
  requests returned the reasoning inline with a stray closing tag in
  `content`. The patch inspects the request's rendered generation prompt and,
  if it ends with a sibling tag (same stem, extra suffix, no `/`), adopts it
  and its `</...>` closer for that request. Generic; no-op for every template
  whose generation prompt ends with the analyzer's own tag.
  v17 revision: the model closes `<ifm|think_fast>` with `</ifm|think>` (all
  six tags are distinct added tokens, so this is trained behaviour, not a
  tokenizer artefact), which made v16 swallow the answer into reasoning at
  medium effort. The analyzer struct gained `end_alternatives`; the PEG
  builder uses `until_one_of` + a choice of closers, and `thinking_end_tags`
  carries both. Target: high/medium/low/thinking-off all
  return the answer in `content` and the thinking in `reasoning_content`.

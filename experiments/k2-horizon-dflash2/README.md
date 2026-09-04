# k2-horizon-dflash2 (experiment, not production)

Preparation kit for training a DFlash2 speculative drafter for
IFM/K2-Horizon-7B with SpecForge and serving it through llama.cpp on the
RTX 3090. Nothing here has run on a GPU yet; see RUNBOOK.md for the
sequence, costs and open risks.

- `sglang-k2-capture.patch`: adds SpecForge's aux hidden-state capture hooks to SGLang's K2 Horizon (xLLM) model class.
- `configs/k2-horizon-7b-dflash2.json`: DFlash2 draft config (5 layers, hidden 4096, GQA 32/8, target layers [1,9,17,26,34] of 36, mask token `reserved_special_token_573` = 250623).
- `specforge_k2_template.py`: K2 Horizon chat template registration for SpecForge.
- `train-k2-7b-dflash2-online.yaml`: online disaggregated training config.

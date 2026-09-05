# k2-horizon-dflash2 (experiment, not production)

Preparation kit for training a DFlash2 speculative drafter for
IFM/K2-Horizon-7B with SpecForge and serving it through llama.cpp on the
RTX 3090. Local EAGLE-3 capture and an initial training attempt ran on
September 4; that training data was discarded. See RUNBOOK.md for the
current regeneration recovery, sequence, costs and open risks.

- `sglang-k2-capture.patch`: adds SpecForge's aux hidden-state capture hooks to SGLang's K2 Horizon (xLLM) model class.
- `configs/k2-horizon-7b-dflash2.json`: DFlash2 draft config (5 layers, hidden 4096, GQA 32/8, target layers [1,9,17,26,34] of 36, mask token `reserved_special_token_573` = 250623).
- `specforge_k2_template.py`: K2 Horizon chat template registration for SpecForge.
- `train-k2-7b-dflash2-online.yaml`: online disaggregated training config.
- `serve-k2-sglang.sh`, `build_regen_prompts.py`, `regenerate_sessions.py`: the regeneration pass that produces target-generated training data.
- `train-eagle3.sh`, `train-k2-7b-eagle3-offline.yaml`: offline EAGLE-3 training on the captured features.
- `inspect_features.py`: feature shape/dtype and bytes-per-token check.
- `sglang-k2-nonstream-reasoning.patch`, `test_resume.py`: fix and regressions for swallowed medium/low-effort final answers in non-streaming regeneration.
- `resume-regeneration.sh`: resume into a separate dataset, with exclusive GPU use and restoration of the local API when the job exits. Requires an authorized API maintenance window.
- `notify_completion.py`: watch one systemd invocation and queue its completion/failure event to the originating Codex chat; state prevents re-sending after a successful queue receipt.

# A later LoRA pilot

No weights were trained in this settings-first experiment. The recipe in
`lora-pilot.yaml` is preparation, not a verified training environment or a
claim that the model will improve.

The completed prompt pilot improved development coding from 4/14 to 9/14 and
reserved coding from 4/6 to 6/6 within an 8k output budget. Its remaining
development failures are four truncations and one incorrect topological-sort
implementation. Establish the improved prompt baseline first; a later adapter
must beat that baseline under matching settings, not receive credit for the
prompt change. See `REPORT.md` for the sample size and reproducibility limits.

## Supported starting point

Use the publisher's [LLaMA-Factory fork and Spark guide](https://github.com/XHToken/LlamaFactory/blob/f56888d47dc5681e15b83ddbea01adddf2050adf/examples/torch_spark2_5/Spark2_5_Finetune.md),
pinned to `f56888d47dc5681e15b83ddbea01adddf2050adf`, with the original
[post-trained HF weights](https://huggingface.co/XHToken/Spark-X2.5-4B/tree/5e10fcc0286756aebf7c41dc52c1e42d95c70281).
The serving GGUF is not the training input. Rank 8, alpha 16, dropout 0.05,
AdamW, learning rate 5e-5 and one epoch follow the publisher's 4B LoRA recipe.
The fork exposes `disable_gradient_checkpointing`; this recipe explicitly
sets it false. Do not enable Muon for LoRA.

The guide has stale details: it describes FP32 weights and `model_type=spark3`,
whereas this weight revision is BF16 with `model_type=spark2_5`. Its suggested
version-check bypass is not justified by the stated compatible range. Install
the pinned fork in a new environment, resolve its declared dependencies, and
record the resulting lockfile rather than reusing the unrelated SpecForge
environment or disabling compatibility checks.

## Data and evaluation

First separate inference-budget failures from incorrect finished code. Training
on output-truncation failures would target a symptom that serving settings can
address. For residual bugs, collect licensed, independently verified examples
of edge-case handling and short read/edit/test/retry workflows. Start with a
small reviewable set (for example 300–1,000 examples), not unfiltered model
outputs. Store provenance and test results with every example. Do not assume
this small set guarantees gains.

Split by repository/task family before constructing examples, remove exact and
near duplicates across splits, and reserve fresh repositories for final
evaluation. Neither this pilot's fixtures nor its hidden assertions belong in
training or training validation. Once these benchmark results have influenced
data selection, the present holdout is no longer a fresh final test for LoRA.

Use the fork's OpenAI-format conversation dataset registration, with separate
`spark_coding_train` and `spark_coding_validation` entries in
`spark-coding-data/dataset_info.json`. Verify tokenized tool names, arguments,
observations and assistant loss masks against the original HF chat template.
The fork's `spark` template replaces the Jinja template: matching its name is
insufficient evidence of tool-format parity. Use complete, short episodes that
fit the 2,048-token training cutoff; do not silently truncate away their fixes.

## Before launching

1. Review the pinned model's remote code, build the isolated environment and
   parse the YAML using that environment's real argument parser.
2. Render/tokenize representative ordinary, thinking and tool conversations;
   confirm template parity and masking. Run one forward/backward batch and a
   short train/save/reload smoke test, measuring peak GPU memory. BF16 frozen
   4B weights alone are about 7.7 GiB; activations and CUDA overhead still need
   measurement. A 2k checkpointed LoRA pilot is plausible on 24 GiB, not yet
   qualified. The 384k inference context says nothing about training capacity.
3. Schedule GPU use: the current 384k inference profile must release its memory
   for training. Do not run both and interpret memory contention as a model bug.
4. Train one candidate only after data review, retaining the untouched baseline.
   Compare baseline and adapter with identical inference settings on fresh
   repository tasks, tool workflows, general instruction following and long
   context retrieval. Keep failures and repeated seeds in the report.
5. Verify the exact adapter tensors can export to the Spark GGUF implementation.
   If direct adapter export is unsupported, test a merged BF16 export followed
   by the same Q8_0 quantization. Benchmark the actual served artifact, including
   384k-context regressions, before considering replacement.

The next weight-training decision should use the finished-code failures and
the user's actual repository workload; a lower training loss is not a coding
quality gate.

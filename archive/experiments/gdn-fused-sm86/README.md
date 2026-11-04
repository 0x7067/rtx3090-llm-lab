# Isolated GDN MTP decode experiment for SM86

This directory contains a private PyTorch CUDA extension based on vLLM PR
[#51674](https://github.com/vllm-project/vllm/pull/51674). It fuses the
post-convolution Qwen3.5/Qwen3.8 Gated DeltaNet MTP decode step into one launch.
It never changes vLLM's `_C` library or the production configuration.

The upstream kernel assumes `HV/H == 8` and accepts only FP32 or BF16 recurrent
state. Qwen3.8-27B uses 16 QK heads and 48 value heads, so its ratio is 3. The
copy in this experiment computes `value_head / (HV / H)` at runtime and adds
FP16 state load, store, and dispatch. The rest of the arithmetic follows the
upstream kernel, including BF16 post-convolution inputs, QK L2 normalization,
gating, recurrence, and gated RMS normalization.

## Build

The build needs a CUDA toolkit with `nvcc`, a matching CUDA-enabled PyTorch,
and a free SM86 device for runtime tests. From this directory:

```bash
PYTHON=/path/to/python ./build_sm86.sh
```

`build_sm86.sh` defaults to `TORCH_CUDA_ARCH_LIST=8.6`. To load the resulting
library in a test or an isolated vLLM worker:

```bash
export VLLM_ENABLE_FUSED_GDN_SM86=1
export GDN_FUSED_SM86_LIBRARY=$PWD/gdn_fused_sm86*.so
```

The Python wrapper's `run_if_enabled(...)` returns `False` unless the gate is
set, the device is exactly SM86, and the private dispatcher op is loaded. The
gate has its own name because vLLM's existing `VLLM_GDN_DECODE_KERNEL`
parser accepts only `cuda` and `triton`. The caller should keep its Triton path
when this wrapper returns `False`.

At vLLM's existing `_forward_core_decode_spec_post_conv_fused_norm` call site,
pass the same tensors to `run_if_enabled(...)` before the Triton fallback. A
`True` result means the private op wrote `core_attn_out` and updated the state.
The wrapper is deliberately not imported by vLLM automatically.

## Checks

```bash
python -m pytest -q tests/test_gdn_fused_sm86.py
```

The CUDA cases cover Qwen3.8 ratios `(16, 48)`, `(8, 24)`, and `(4, 12)` with
both FP16 and BF16 recurrent state. They compare output and in-place state
updates against vLLM's `fused_sigmoid_gating_delta_rule_update` Triton path for
ragged MTP requests and accepted-token counts 1, 2, and the full request
width. They skip when no SM86 GPU or built library is available.

## Artifact compatibility

The official [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B)
describes 48 value heads, 16 QK heads, and 128-dimensional GDN heads. The
prepared local `dbirks/Qwen3.8-27B-W4A16-AutoRound` artifact and the local
`Qwen3.8-27B-DFlash2-W4A16` artifact keep BF16 activations while quantizing
weights. That distinction matters here: this operator consumes the runtime
post-convolution tensors and recurrent cache, so the weight distribution does
not change its head-ratio or state-dtype checks. The surrounding vLLM model
code still has to preserve the MTP tensors and the `[slots, HV, 128, 128]`
state layout.

## Result

The extension built inside `qwen38-27b-3090:v9` and all eight CUDA checks
passed on the RTX 3090. The private end-to-end derivative was nevertheless
rejected: the frozen short suite fell from 97.57 to 36.23 tok/s and MTP draft
agreement collapsed to roughly one token per step. Per-layer output errors in
the isolated tests were tiny (normally below 0.01% relative L2), but they
compound through the target's GDN layers enough to change autoregressive
logits. Do not promote this kernel without a target-logit parity test and
end-to-end acceptance gate.

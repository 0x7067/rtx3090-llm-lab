# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVarN: Variance-Normalized KV-cache quantization for vLLM.

Hadamard rotation along the channel dimension, then iterative log-domain
variance-normalization (Sinkhorn-like) over both axes of each 128-token
tile, then asymmetric round-to-nearest. Keys are quantized per-channel
(KIVI K-axis); values are quantized per-token (KIVI V-axis).
"""

# port(0.27.1): also re-export KVARN_PRESETS (the plan refers to it from this
# package). The MLA prototypes (mla_probe.py / mla_quant.py) are deliberately
# NOT ported and must not be imported here.
from vllm.model_executor.layers.quantization.kvarn.config import (
    KVARN_PRESETS,
    KVarNConfig,
)

__all__ = ["KVarNConfig", "KVARN_PRESETS"]

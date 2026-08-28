// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Standalone dispatcher registration for the isolated sm86 experiment.  The
// operation is deliberately in its own namespace so loading this extension
// cannot replace vLLM's production _C operator.

#include <torch/extension.h>

void fused_gdn_decode_post_conv_mtp(
    torch::Tensor const& mixed_qkv, torch::Tensor const& a,
    torch::Tensor const& b, torch::Tensor const& a_log,
    torch::Tensor const& dt_bias, torch::Tensor const& state_indices,
    torch::Tensor const& cu_seqlens,
    torch::Tensor const& num_accepted_tokens, torch::Tensor& state,
    torch::Tensor const& output_gate, torch::Tensor const& norm_weight,
    torch::Tensor& out, double scale, double norm_eps);

TORCH_LIBRARY(gdn_fused_sm86, m) {
  m.def(
      "decode_post_conv_mtp(Tensor mixed_qkv, Tensor a, Tensor b, "
      "Tensor A_log, Tensor dt_bias, Tensor state_indices, "
      "Tensor cu_seqlens, Tensor num_accepted_tokens, Tensor! state, "
      "Tensor output_gate, Tensor norm_weight, Tensor! out, float scale, "
      "float norm_eps=1e-5) -> ()");
}

TORCH_LIBRARY_IMPL(gdn_fused_sm86, CUDA, m) {
  m.impl("decode_post_conv_mtp", &fused_gdn_decode_post_conv_mtp);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("registered", [] { return true; });
}

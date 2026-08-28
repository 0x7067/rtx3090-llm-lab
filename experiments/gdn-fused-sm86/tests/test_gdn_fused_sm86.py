"""Numerical checks for the isolated Qwen3.8 GDN MTP extension.

The test compares the extension with vLLM's existing Triton recurrent path.
It needs a free SM86 GPU and a built extension, so environments without those
pieces skip the CUDA cases instead of pretending they passed.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

try:
    import torch as _torch
except ModuleNotFoundError:
    _torch = None

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))


def _cuda_available() -> bool:
    return _torch is not None and _torch.cuda.is_available()


def test_source_has_runtime_ratio_and_fp16_state_dispatch() -> None:
    source = (EXPERIMENT_DIR / "fused_gdn_decode_sm86.cu").read_text()
    assert "value_head / (HV / H)" in source
    assert "num_value_heads % num_key_heads == 0" in source
    assert "load_state4<__half>" in source
    assert "store_state4<__half>" in source
    assert "state_scalar_type == at::kHalf" in source
    assert "num_value_heads == 8 * num_key_heads" not in source


def test_gate_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    from gdn_fused_sm86 import enabled  # noqa: PLC0415

    monkeypatch.delenv("VLLM_ENABLE_FUSED_GDN_SM86", raising=False)
    assert not enabled()
    monkeypatch.setenv("VLLM_ENABLE_FUSED_GDN_SM86", "sm86")
    assert enabled()
    monkeypatch.setenv("VLLM_ENABLE_FUSED_GDN_SM86", "0")
    assert not enabled()


@pytest.mark.parametrize("H,HV", [(16, 48), (8, 24), (4, 12)])
@pytest.mark.parametrize("state_dtype_name", ["float16", "bfloat16"])
@pytest.mark.skipif(not _cuda_available(), reason="requires a CUDA device")
def test_matches_triton_for_qwen38_ratios(
    H: int, HV: int, state_dtype_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    from gdn_fused_sm86 import available, load_library  # noqa: PLC0415

    state_dtype = getattr(torch, state_dtype_name)
    monkeypatch.setenv("VLLM_ENABLE_FUSED_GDN_SM86", "1")
    if tuple(torch.cuda.get_device_capability()) != (8, 6):
        pytest.skip("the extension is built for SM86")
    library = os.environ.get("GDN_FUSED_SM86_LIBRARY")
    if library is None:
        candidates = list(EXPERIMENT_DIR.glob("gdn_fused_sm86*.so"))
        if len(candidates) != 1:
            pytest.skip("build the extension or set GDN_FUSED_SM86_LIBRARY")
        library = str(candidates[0])
    load_library(library)
    if not available():
        pytest.skip("gdn_fused_sm86 dispatcher op is unavailable")

    from vllm.third_party.flash_linear_attention.ops import (  # noqa: PLC0415
        fused_sigmoid_gating_delta_rule_update,
    )

    torch.manual_seed(0)
    device = "cuda"
    K = V = 128
    query_lengths = (4, 2)
    num_requests = len(query_lengths)
    state_width = max(query_lengths)
    num_tokens = sum(query_lengths)
    num_slots = num_requests * state_width + 1
    scale = K**-0.5
    eps = 1e-6

    mixed_qkv = torch.randn(
        num_tokens, 2 * H * K + HV * V, dtype=torch.bfloat16, device=device
    )
    query, key, value = torch.split(mixed_qkv, [H * K, H * K, HV * V], dim=-1)
    query = query.view(1, num_tokens, H, K)
    key = key.view(1, num_tokens, H, K)
    value = value.view(1, num_tokens, HV, V)
    ba = torch.randn(num_tokens, 2 * HV, dtype=torch.bfloat16, device=device)
    b, a = ba.chunk(2, dim=-1)
    A_log = 0.5 * torch.randn(HV, dtype=torch.float32, device=device)
    dt_bias = 0.1 * torch.randn(HV, dtype=torch.float32, device=device)
    output_gate = torch.randn(num_tokens, HV, V, dtype=torch.bfloat16, device=device)
    norm_weight = torch.randn(V, dtype=torch.float32, device=device)
    state_ref = (
        0.01
        * torch.randn(num_slots, HV, V, K, dtype=torch.float32, device=device)
    ).to(state_dtype)
    state_actual = state_ref.clone()
    state_indices = torch.arange(1, num_slots, dtype=torch.int32, device=device).view(
        num_requests, state_width
    )
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(query_lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    num_accepted_tokens = torch.ones(num_requests, dtype=torch.int32, device=device)

    for accepted_tokens in dict.fromkeys((1, min(2, state_width), state_width)):
        num_accepted_tokens.fill_(accepted_tokens)
        raw_ref, _ = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            a=a,
            b=b,
            dt_bias=dt_bias,
            q=query,
            k=key,
            v=value,
            initial_state=state_ref,
            inplace_final_state=True,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=state_indices,
            num_accepted_tokens=num_accepted_tokens,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
        )
        from vllm.third_party.flash_linear_attention.ops.layernorm_guard import (  # noqa: PLC0415
            rmsnorm_fn,
        )

        expected = rmsnorm_fn(
            raw_ref.squeeze(0),
            norm_weight,
            None,
            z=output_gate,
            eps=eps,
            norm_before_gate=True,
            activation="silu",
        )
        actual = torch.empty_like(output_gate)
        getattr(torch.ops.gdn_fused_sm86, "decode_post_conv_mtp")(
            mixed_qkv,
            a,
            b,
            A_log,
            dt_bias,
            state_indices,
            cu_seqlens,
            num_accepted_tokens,
            state_actual,
            output_gate,
            norm_weight,
            actual,
            scale,
            eps,
        )
        torch.cuda.synchronize()
        output_relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-20)
        print(
            f"H={H} HV={HV} state={state_dtype_name} accepted={accepted_tokens} "
            f"output_relative_l2={output_relative_l2.item():.8f} "
            f"output_max_abs={(actual.float() - expected.float()).abs().max().item():.8f}"
        )
        assert output_relative_l2 < 5e-3
        torch.testing.assert_close(state_actual, state_ref, atol=3e-2, rtol=3e-2)

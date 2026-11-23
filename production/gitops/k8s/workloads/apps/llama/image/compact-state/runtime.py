"""Compact speculative states for the pinned Qwen PIECEWISE/align runtime."""

import torch
from vllm import envs
from vllm.logger import init_logger
from vllm.third_party.flash_linear_attention.ops import (
    fused_sigmoid_gating_delta_rule_update,
)

from .kernels import restore_kernel, verify_kernel

logger = init_logger("vllm.compact_gdn")
_real_forward = False


def begin(dummy_run):
    global _real_forward
    _real_forward = not dummy_run


def initialize(layer, config):
    layer._compact = None
    layer._compact_pending = None
    logger.info_once(
        "Compact GDN initializer: enabled=%s heads=%s/%s dims=%s/%s",
        envs.VLLM_COMPACT_GDN,
        layer.num_k_heads,
        layer.num_v_heads,
        layer.head_k_dim,
        layer.head_v_dim,
    )
    if not envs.VLLM_COMPACT_GDN:
        return
    if (
        not config.use_v2_model_runner
        or config.cache_config.mamba_cache_mode != "align"
        or str(config.compilation_config.cudagraph_mode).split(".")[-1] != "PIECEWISE"
        or config.parallel_config.tensor_parallel_size != 1
        or config.parallel_config.pipeline_parallel_size != 1
    ):
        raise ValueError(
            "Compact GDN requires V2, align cache, PIECEWISE graphs and TP=PP=1"
        )
    if (layer.num_k_heads, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim) != (
        16,
        48,
        128,
        128,
    ):
        return
    n = config.scheduler_config.max_num_seqs
    device = layer.A_log.device
    layer._compact = {
        "delta": torch.empty((n, 8, 48, 128), device=device, dtype=torch.float32),
        "keys": torch.empty((n, 8, 16, 128), device=device, dtype=torch.float32),
        "decay": torch.empty((n, 8, 48), device=device, dtype=torch.float32),
        "source": torch.empty(n, device=device, dtype=torch.int32),
    }
    logger.info_once(
        "Compact GDN initialized: accepted and prefix-boundary reconstruction"
    )


def update(layer, metadata, **kw):
    scratch = layer._compact
    q, k, v = kw["q"], kw["k"], kw["v"]
    state = kw["initial_state"]
    n = metadata.num_spec_decodes
    indices = kw["ssm_state_indices"]
    lengths = kw["cu_seqlens"]
    # Pure speculative batches preserve request-row identity through sampling.
    # Mixed prefill batches retain the stock implementation and cache behavior.
    eligible = (
        _real_forward
        and scratch is not None
        and metadata.num_prefills == 0
        and metadata.num_decodes == 0
        and n > 0
        and n <= scratch["source"].numel()
        and indices.shape[1] <= 8
        and state.dtype == torch.float16
        and q.dtype == torch.bfloat16
        and state.shape[1:] == (48, 128, 128)
        and metadata.uniform_spec_sequence_length in range(2, 9)
        and not torch.cuda.is_current_stream_capturing()
    )
    if not eligible:
        return fused_sigmoid_gating_delta_rule_update(**kw)
    assert layer._compact_pending is None, (
        "Unfinalized compact state from prior forward"
    )
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    a, b = kw["a"].contiguous(), kw["b"].contiguous()
    out = torch.empty_like(v)
    verify_kernel[(4, n * 48)](
        q,
        k,
        v,
        a,
        kw["dt_bias"],
        kw["A_log"],
        b,
        state,
        out,
        scratch["delta"],
        scratch["keys"],
        scratch["decay"],
        1.0,
        20.0,
        128**-0.5,
        lengths,
        indices,
        kw["num_accepted_tokens"],
        scratch["source"],
        SS=state.stride(0),
        IS=indices.stride(0),
        T=8,
        H=16,
        HV=48,
        D=128,
        BV=32,
        num_warps=4,
        num_stages=3,
    )
    layer._compact_pending = (state, indices, lengths, n)
    logger.info_once("Compact GDN verification engaged")
    return out, state


def finalize(runner, idx_mapping, num_sampled):
    global _real_forward
    _real_forward = False
    num_reqs = idx_mapping.shape[0]
    layers = runner.compilation_config.static_forward_context.values()
    pending = [
        layer
        for layer in layers
        if getattr(layer, "_compact_pending", None) is not None
    ]
    if not pending:
        return
    ctx = runner.model_state._mamba_ctx
    assert ctx is not None
    # Every GDN layer sees the same pure-spec batch and request ordering.
    for layer in pending:
        state, indices, lengths, n = layer._compact_pending
        assert n == num_reqs
        s = layer._compact
        restore_kernel[(4, n * 48)](
            state,
            s["delta"],
            s["keys"],
            s["decay"],
            num_sampled,
            s["source"],
            indices,
            lengths,
            runner.req_states.num_computed_tokens.gpu,
            idx_mapping,
            BlockSize=ctx.block_size,
            SS=state.stride(0),
            IS=indices.stride(0),
            T=8,
            H=16,
            HV=48,
            D=128,
            BV=32,
            num_warps=4,
            num_stages=3,
        )
        layer._compact_pending = None
    logger.info_once(
        "Compact GDN accepted and boundary states finalized before cache copies"
    )

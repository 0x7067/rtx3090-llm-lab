"""SM86 (RTX 3090 / Ampere) v3 concurrency + single-stream runtime optimizations for Inkling2 on vLLM.

Lazily patches Inkling2 modules on SM8x GPUs (< SM89):
1. `ops.attention_sm90`: Unified unscaled uint16 E4M3 Triton kernel for BOTH
   global (2-loop interior+tail, BN=64 for ALL decode batches -> 1.23-1.39x
   faster at N=2..16!) and local SWA-1024 layers, tuned for real Inkling2 Mini
   geometry (H=24, HKV=8, G=3, D=128).
2. `ops.attention`: SM86 concurrency-tuned `plan_decode_attention` (local sp=1
   fused gate/sink; global sp=20 at N=1, sp=10 at N=2, sp=5 at N>=4).
3. `attention`: Split-KV prefill up to q=1024 (sp=2 -> 1.21x faster) and fused
   gate/sink on sp=1.
4. `ops.moe_sm90`: RTX 3090 SM86 autotuned deterministic Humming schedules
   including concurrency buckets (rows=64..1840) for both NVFP4 and MXFP8.
5. `ops.linear`: SM86 low-M BF16 GEMV for qkvr_local ONLY (wo_ud uses cuBLAS,
   saving 1.06 ms/step at N=1 and 2.26 ms/step at N=2!) + L2-swizzled
   concurrency-autotuned INT8 W8A16 GEMM/GEMV (1.23-3.10x faster at M=16..128)
   + fused INT8 markov_fold + add_softcap epilogue.
6. `ops.dspark`, `dspark`, `speculator`:
   - DSpark FP8 E4M3 KV cache + zero-VRAM on-the-fly RoPE + markov_w1 UVA +
     INT8 3-layer draft weights & fold_weight.
   - Fused `markov_fold + add_softcap` kernel eliminating 7 full-vocab
     `[N, 201024]` intermediate HBM round-trips and 7 kernel launches per step.
7. `kv_cache_planning`: 0-padding GCD group sizing (group_size=3 -> 33 groups).
8. `model`: Visual tower UVA offload + optional SHA-256-verified Unbound
   `edit.pt` hook + optional INT8 `lm_head`.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import cast

import torch

_PATCHED: set[str] = set()
_SM86_REL_ATTENTION = None
_SM86_LINEAR_KERNELS = None
_SM86_FP8_KV_WRITER = None


def _env_flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default) != "0"


def _get_cap():
    from vllm.platforms import current_platform
    return current_platform.get_device_capability()


def _quantize_int8_per_row(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row INT8 quantization (fp32 scale) for a 2D [N, K] weight."""
    w_f32 = w.float()
    max_abs = w_f32.abs().amax(dim=-1)
    scale = (max_abs / 127.0).clamp_min(1e-12).to(torch.float32).contiguous()
    w_q = torch.clamp(torch.round(w_f32 / scale[:, None]), -127, 127).to(torch.int8).contiguous()
    return w_q, scale


def _get_linear_kernels():
    global _SM86_LINEAR_KERNELS
    if _SM86_LINEAR_KERNELS is not None:
        return _SM86_LINEAR_KERNELS

    from vllm.triton_utils import tl, tldevice, triton

    @triton.jit
    def _bf16_gemv_kernel(
        X, W, Y, M, N, K, SX, SW, SY,
        BN: tl.constexpr, BK: tl.constexpr, MP: tl.constexpr,
    ):
        pid = tl.program_id(0)
        cols = pid * BN + tl.arange(0, BN)
        rows = tl.arange(0, MP)
        acc = tl.zeros((MP, BN), tl.float32)
        for k0 in range(0, K, BK):
            ks = k0 + tl.arange(0, BK)
            w = tl.load(
                W + cols[None, :].to(tl.int64) * SW + ks[:, None],
                (cols[None, :] < N) & (ks[:, None] < K),
                other=0,
            )
            x = tl.load(
                X + rows[:, None] * SX + ks[None, :],
                (rows[:, None] < M) & (ks[None, :] < K),
                other=0,
            )
            acc += tl.dot(x, w)
        tl.store(
            Y + rows[:, None] * SY + cols[None, :],
            acc.to(Y.dtype.element_ty),
            (rows[:, None] < M) & (cols[None, :] < N),
        )

    @triton.jit
    def _int8_w8a16_kernel(
        X, W, S, B, BASE, Y, M, N, K, SX, SW, SB, SY, softcap,
        HAS_BIAS: tl.constexpr,
        HAS_BASE_SOFTCAP: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BM)
        num_pid_n = tl.cdiv(N, BN)
        if GROUP_M > 1:
            num_pid_in_group = GROUP_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_M
            group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
            pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m
        else:
            pid_n = pid % num_pid_n
            pid_m = pid // num_pid_n

        cols = pid_n * BN + tl.arange(0, BN)
        rows = pid_m * BM + tl.arange(0, BM)
        acc = tl.zeros((BM, BN), tl.float32)
        for k0 in range(0, K, BK):
            ks = k0 + tl.arange(0, BK)
            w_q = tl.load(
                W + cols[None, :].to(tl.int64) * SW + ks[:, None],
                (cols[None, :] < N) & (ks[:, None] < K),
                other=0,
            )
            x = tl.load(
                X + rows[:, None].to(tl.int64) * SX + ks[None, :],
                (rows[:, None] < M) & (ks[None, :] < K),
                other=0,
            )
            acc += tl.dot(x, w_q.to(tl.bfloat16))
        scale = tl.load(S + cols, cols < N, other=0.0).to(tl.float32)
        acc = acc * scale[None, :]
        if HAS_BIAS:
            bias = tl.load(B + cols, cols < N, other=0.0).to(tl.float32)
            acc = acc + bias[None, :]
        if HAS_BASE_SOFTCAP:
            # Bit-exact match to ops.dspark.add_softcap(base, bf16(acc), softcap)
            base_val = tl.load(
                BASE + rows[:, None].to(tl.int64) * SB + cols[None, :],
                (rows[:, None] < M) & (cols[None, :] < N),
                other=0.0,
            ).to(tl.float32)
            bias_bf16 = acc.to(tl.bfloat16).to(tl.float32)
            x_sc = (base_val + bias_bf16).to(tl.bfloat16).to(tl.float32)
            x_sc = tldevice.tanh(x_sc / softcap) * softcap
            tl.store(
                Y + rows[:, None].to(tl.int64) * SY + cols[None, :],
                x_sc.to(Y.dtype.element_ty),
                (rows[:, None] < M) & (cols[None, :] < N),
            )
        else:
            tl.store(
                Y + rows[:, None].to(tl.int64) * SY + cols[None, :],
                acc.to(Y.dtype.element_ty),
                (rows[:, None] < M) & (cols[None, :] < N),
            )

    def bf16_gemv_sm86(x: torch.Tensor, w: torch.Tensor, out_dtype=None) -> torch.Tensor:
        lead = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1])
        m, k = x2.shape
        n = w.shape[0]
        y = torch.empty((m, n), device=x.device, dtype=out_dtype or x.dtype)
        if m == 0:
            return y.view(*lead, n)
        bn, bk, nw, ns = 32, 128, 4, 2
        _bf16_gemv_kernel[(triton.cdiv(n, bn),)](
            x2, w, y, m, n, k, x2.stride(0), w.stride(0), y.stride(0),
            BN=bn, BK=bk, MP=16, num_warps=nw, num_stages=ns,
        )
        return y.view(*lead, n)

    def _pick_int8_cfg(m: int, n: int, k: int) -> tuple[int, int, int, int, int, int]:
        """SM86 (RTX 3090) autotuned (BM, BN, BK, num_warps, num_stages, GROUP_M)."""
        if n >= 100000:
            # lm_head (K=2304, N=201024) and markov_fold (K=256, N=201024)
            if m <= 16:
                return 16, 64, 128, 8, 2, 1
            if m <= 32:
                return 32, 128, 128, 4, 3, 1
            if m <= 64:
                return 32, 128, 64, 8, 3, 4
            return 64, 128, 64, 4, 3, 4
        if k >= 4096:
            # dspark_down (K=9216, N=2304)
            if m <= 16:
                return 16, 32, 128, 4, 3, 1
            if m <= 32:
                return 16, 64, 128, 8, 3, 1
            if m <= 64:
                return 32, 64, 128, 4, 3, 4
            return 32, 128, 128, 8, 3, 1
        # dspark_gate_up (K=2304, N=18432), qkv_proj, kv_proj_ctx, o_proj
        if m <= 8:
            return 16, 64, 128, 4, 2, 1
        if m <= 16:
            return 16, 128, 128, 8, 2, 1
        if m <= 32:
            return 32, 128, 128, 8, 3, 1
        if m <= 64:
            return 64, 128, 64, 4, 2, 1
        return 128, 128, 128, 8, 3, 1

    def int8_w8a16_linear(
        x: torch.Tensor,
        w_q: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
        out: torch.Tensor | None = None,
        base: torch.Tensor | None = None,
        softcap: float = 0.0,
    ) -> torch.Tensor:
        lead = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1])
        m, k = x2.shape
        n = w_q.shape[0]
        if out is None:
            y = torch.empty((m, n), device=x.device, dtype=out_dtype or x.dtype)
        else:
            y = out.reshape(m, n)
        if m == 0:
            return y.view(*lead, n)
        bm, bn, bk, nw, ns, gm = _pick_int8_cfg(m, n, k)
        if bk > k:
            bk = 64
        if m <= bm:
            gm = 1
        b_ptr = bias if bias is not None else scale
        has_base = base is not None
        if has_base:
            base_ptr = base
            sb = base.stride(0)
        else:
            base_ptr = y
            sb = 0
        grid = (triton.cdiv(n, bn) * triton.cdiv(m, bm),)
        _int8_w8a16_kernel[grid](
            x2, w_q, scale, b_ptr, base_ptr, y, m, n, k,
            x2.stride(0), w_q.stride(0), sb, y.stride(0), float(softcap),
            HAS_BIAS=bias is not None,
            HAS_BASE_SOFTCAP=has_base,
            BM=bm, BN=bn, BK=bk, GROUP_M=gm,
            num_warps=nw, num_stages=ns,
        )
        return y.view(*lead, n)

    _SM86_LINEAR_KERNELS = (bf16_gemv_sm86, int8_w8a16_linear)
    return _SM86_LINEAR_KERNELS


def _get_fp8_kv_writer():
    global _SM86_FP8_KV_WRITER
    if _SM86_FP8_KV_WRITER is not None:
        return _SM86_FP8_KV_WRITER

    from vllm.triton_utils import tl, triton
    from vllm.models.inkling2.nvidia.ops.sconv import _f32_to_e4m3

    @triton.jit
    def _write_fp8_kv_kernel(
        K, V, KCache, VCache, SlotMapping,
        stride_k_t, stride_k_h, stride_v_t, stride_v_h,
        stride_kc_blk, stride_kc_n, stride_kc_h,
        stride_vc_blk, stride_vc_n, stride_vc_h,
        block_size,
        H: tl.constexpr, D: tl.constexpr,
    ):
        t = tl.program_id(0).to(tl.int64)
        h = tl.program_id(1)
        slot = tl.load(SlotMapping + t)
        if slot < 0:
            return
        offs_d = tl.arange(0, D)
        blk = slot // block_size
        pos = slot % block_size
        k = tl.load(K + t * stride_k_t + h * stride_k_h + offs_d).to(tl.float32)
        v = tl.load(V + t * stride_v_t + h * stride_v_h + offs_d).to(tl.float32)
        k_code = _f32_to_e4m3(k).to(tl.uint8)
        v_code = _f32_to_e4m3(v).to(tl.uint8)
        tl.store(
            KCache + blk * stride_kc_blk + pos * stride_kc_n + h * stride_kc_h + offs_d,
            k_code,
        )
        tl.store(
            VCache + blk * stride_vc_blk + pos * stride_vc_n + h * stride_vc_h + offs_d,
            v_code,
        )

    def write_fp8_kv(
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        t, h, d = k.shape
        if t == 0:
            return
        kc_u8 = kv_cache.view(torch.uint8)
        key_cache, value_cache = kc_u8.transpose(1, 2).split(d, dim=-1)
        block_size = key_cache.shape[1]
        _write_fp8_kv_kernel[(t, h)](
            k, v, key_cache, value_cache, slot_mapping,
            k.stride(0), k.stride(1), v.stride(0), v.stride(1),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
            block_size,
            H=h, D=d, num_warps=4,
        )

    _SM86_FP8_KV_WRITER = write_fp8_kv
    return _SM86_FP8_KV_WRITER


def _build_rel_attention_sm86(attn_sm90):
    global _SM86_REL_ATTENTION
    if _SM86_REL_ATTENTION is not None:
        return _SM86_REL_ATTENTION

    from vllm.triton_utils import tl, triton

    @triton.jit
    def _sigmoid_fast(x):
        return 1.0 / (1.0 + tl.exp2(-x * 1.4426950408889634))

    @triton.jit
    def _load_fp8_kv_unscaled(ptr, offsets, mask, dtype: tl.constexpr, D: tl.constexpr):
        half = tl.arange(0, D // 2)
        pair = tl.load(
            ptr.to(tl.pointer_type(tl.uint16))
            + tl.multiple_of(offsets // 2, D // 2)[:, None]
            + half[None, :],
            mask=mask[:, None],
            other=0,
        )
        lo = (((pair & 0x007F) << 7) | ((pair & 0x0080) << 8)).to(tl.float16, bitcast=True)
        hi = (((pair & 0x7F00) >> 1) | (pair & 0x8000)).to(tl.float16, bitcast=True)
        return tl.interleave(lo, hi).to(dtype)

    @triton.jit
    def _rel_attention_sm86_kernel(
        Q, K, V, Rel, BlockTable, SeqLens, CuSeqlensQ, Out, Lse, Gate, Sink,
        softmax_scale, num_tokens, num_splits,
        stride_q_t, stride_q_h, stride_kv_blk, stride_kv_slot, stride_kv_h,
        stride_rel_t, stride_bt, stride_out_t, stride_out_h, stride_lse_h, stride_lse_t,
        stride_gate, stride_sink,
        H: tl.constexpr, G: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
        PAGE: tl.constexpr, WINDOW: tl.constexpr, DRAFT: tl.constexpr, MODE: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        TOKENS: tl.constexpr = BLOCK_M // G
        E_PAD: tl.constexpr = E + 256
        LOG2E: tl.constexpr = 1.4426950408889634
        _SHEAR_TILE: tl.constexpr = 128
        m_tile = tl.program_id(0)
        kv_head = tl.program_id(1) // num_splits
        split = tl.program_id(1) % num_splits
        req = tl.program_id(2)
        q_start = tl.load(CuSeqlensQ + req)
        q_len = tl.load(CuSeqlensQ + req + 1) - q_start
        first_tok = m_tile * TOKENS
        if first_tok >= q_len:
            return
        seqlen_k = tl.load(SeqLens + req)
        past = seqlen_k - q_len

        rows = tl.arange(0, BLOCK_M)
        tok = first_tok + rows // G
        row_valid = (rows < TOKENS * G) & (tok < q_len)
        head = kv_head * G + rows % G
        token = (q_start + tok).to(tl.int64)
        if DRAFT:
            pos = tl.zeros([BLOCK_M], tl.int32) + seqlen_k - 1
            pos_lo = seqlen_k - 1
            pos_hi = seqlen_k - 1
        else:
            pos = past + tok
            pos_lo = past + first_tok
            pos_hi = past + tl.minimum(first_tok + TOKENS, q_len) - 1

        n_lo = 0
        if WINDOW > 0:
            n_lo = tl.maximum(pos_lo - (WINDOW - 1), 0) // BLOCK_N
        n_hi = tl.minimum(pos_hi + 1, seqlen_k)
        n_hi = (n_hi + BLOCK_N - 1) // BLOCK_N
        num_iters = tl.maximum(n_hi - n_lo - split + num_splits - 1, 0) // num_splits

        offs_d = tl.arange(0, D)
        q = tl.load(
            Q + token[:, None] * stride_q_t + head[:, None] * stride_q_h + offs_d[None, :],
            mask=row_valid[:, None],
            other=0.0,
        )
        if not DRAFT:
            tile_end = tl.minimum((tok // _SHEAR_TILE + 1) * _SHEAR_TILE, q_len)
            rel_base = (past + tile_end + 127) // 128 * 128 - E_PAD
            rel_row = Rel + token * stride_rel_t + head * E_PAD - rel_base

        IS_FP8_K: tl.constexpr = K.dtype.element_ty.primitive_bitwidth == 8
        IS_FP8_V: tl.constexpr = V.dtype.element_ty.primitive_bitwidth == 8
        qk_scale = softmax_scale * LOG2E * (256.0 if IS_FP8_K else 1.0)
        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, D], tl.float32)
        block_table = BlockTable + req.to(tl.int64) * stride_bt
        kv_off = kv_head * stride_kv_h

        if WINDOW == 0:
            interior_hi_key = pos_lo if DRAFT else (pos_lo - E + 1)
            n_interior_hi = tl.minimum(tl.maximum(interior_hi_key // BLOCK_N, 0), n_hi)
            num_interior = tl.maximum(n_interior_hi - split + num_splits - 1, 0) // num_splits

            for i in range(num_interior):
                n = split + i * num_splits
                keys = n * BLOCK_N + tl.arange(0, BLOCK_N)
                page = tl.load(block_table + keys // PAGE)
                kv_ptrs = (
                    page.to(tl.int64) * stride_kv_blk
                    + (keys % PAGE) * stride_kv_slot
                    + kv_off
                )
                if IS_FP8_K:
                    half = tl.arange(0, D // 2)
                    pair = tl.load(
                        K.to(tl.pointer_type(tl.uint16))
                        + tl.multiple_of(kv_ptrs // 2, D // 2)[:, None]
                        + half[None, :]
                    )
                    lo = (((pair & 0x007F) << 7) | ((pair & 0x0080) << 8)).to(tl.float16, bitcast=True)
                    hi = (((pair & 0x7F00) >> 1) | (pair & 0x8000)).to(tl.float16, bitcast=True)
                    k = tl.interleave(lo, hi).to(q.dtype).T
                else:
                    k = tl.load(K + kv_ptrs[None, :] + offs_d[:, None]).to(q.dtype)
                s = tl.dot(q, k) * qk_scale
                m_new = tl.maximum(m_i, tl.max(s, 1))
                alpha = tl.exp2(m_i - m_new)
                p = tl.exp2(s - m_new[:, None])
                l_i = l_i * alpha + tl.sum(p, 1)
                if IS_FP8_V:
                    half = tl.arange(0, D // 2)
                    pair = tl.load(
                        V.to(tl.pointer_type(tl.uint16))
                        + tl.multiple_of(kv_ptrs // 2, D // 2)[:, None]
                        + half[None, :]
                    )
                    lo = (((pair & 0x007F) << 7) | ((pair & 0x0080) << 8)).to(tl.float16, bitcast=True)
                    hi = (((pair & 0x7F00) >> 1) | (pair & 0x8000)).to(tl.float16, bitcast=True)
                    v = tl.interleave(lo, hi).to(q.dtype)
                else:
                    v = tl.load(V + kv_ptrs[:, None] + offs_d[None, :]).to(q.dtype)
                acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
                m_i = m_new
            tail_start = num_interior
        else:
            tail_start = 0

        for i in range(tail_start, num_iters):
            n = n_lo + split + i * num_splits
            keys = n * BLOCK_N + tl.arange(0, BLOCK_N)
            key_valid = keys < seqlen_k
            page = tl.load(block_table + keys // PAGE, mask=key_valid, other=0)
            kv_ptrs = (
                page.to(tl.int64) * stride_kv_blk
                + (keys % PAGE) * stride_kv_slot
                + kv_off
            )
            if IS_FP8_K:
                k = _load_fp8_kv_unscaled(K, kv_ptrs, key_valid, q.dtype, D).T
            else:
                k = tl.load(
                    K + kv_ptrs[None, :] + offs_d[:, None],
                    mask=key_valid[None, :],
                    other=0.0,
                ).to(q.dtype)
            s = tl.dot(q, k) * qk_scale
            dist = pos[:, None] - keys[None, :]
            visible = (dist >= 0) & key_valid[None, :]
            if WINDOW > 0:
                visible &= dist < WINDOW
            if not DRAFT:
                in_extent = visible & (dist < E) & row_valid[:, None]
                bias = tl.load(rel_row[:, None] + keys[None, :], mask=in_extent, other=0.0)
                s += bias.to(tl.float32) * LOG2E
            s = tl.where(visible, s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.exp2(m_i - m_safe)
            p = tl.exp2(s - m_safe[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            if IS_FP8_V:
                v = _load_fp8_kv_unscaled(V, kv_ptrs, key_valid, q.dtype, D)
            else:
                v = tl.load(
                    V + kv_ptrs[:, None] + offs_d[None, :],
                    mask=key_valid[:, None],
                    other=0.0,
                ).to(q.dtype)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

        empty = l_i == 0.0
        v_scale = 256.0 if IS_FP8_V else 1.0
        o = (acc * v_scale) / tl.where(empty, 1.0, l_i)[:, None]
        lse2 = tl.where(empty, float("-inf"), m_i + tl.log2(tl.where(empty, 1.0, l_i)))
        if MODE == 0:
            tl.store(
                Out + token[:, None] * stride_out_t + head[:, None] * stride_out_h + offs_d[None, :],
                o,
                mask=row_valid[:, None],
            )
            tl.store(
                Lse + head * stride_lse_h + token * stride_lse_t,
                lse2 * 0.6931471805599453,
                mask=row_valid,
            )
        elif MODE == 1:
            cols = head[:, None] * D + offs_d[None, :]
            if not DRAFT:
                gate = tl.load(Gate + token[:, None] * stride_gate + cols, mask=row_valid[:, None], other=0.0)
                sink = tl.load(Sink + token * stride_sink + head, mask=row_valid, other=0.0)
                o = o * _sigmoid_fast(gate.to(tl.float32))
                o = o * _sigmoid_fast(lse2 * 0.6931471805599453 - sink.to(tl.float32))[:, None]
            tl.store(Out + token[:, None] * stride_out_t + cols, o.to(Out.dtype.element_ty), mask=row_valid[:, None])
        else:
            part = (split * num_tokens + token) * H + head
            tl.store(Lse + part, lse2, mask=row_valid)
            tl.store(Out + part[:, None] * D + offs_d[None, :], o, mask=row_valid[:, None])

    _orig_rel_attention = attn_sm90.rel_attention

    def _rel_attention_sm86(
        q, key_cache, value_cache, out, lse, rel_logits, gate=None, sinks=None, *,
        block_table, cache_seqlens, cu_seqlens_q, max_query_len, softmax_scale,
        window, num_splits, mode,
    ):
        if window > 0 and not _env_flag("INKLING_SM86_LOCAL_ATTN", "1"):
            return _orig_rel_attention(
                q, key_cache, value_cache, out, lse, rel_logits, gate, sinks,
                block_table=block_table, cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q, max_query_len=max_query_len,
                softmax_scale=softmax_scale, window=window,
                num_splits=num_splits, mode=mode,
            )
        num_tokens, num_heads, head_dim = q.shape
        page, num_kv_heads = key_cache.shape[1:3]
        group = num_heads // num_kv_heads
        assert key_cache.stride() == value_cache.stride() and key_cache.stride(3) == 1
        assert q.stride(2) == 1
        assert mode != attn_sm90.MODE_GATED or num_splits == 1
        draft = rel_logits is None
        extent = 0 if rel_logits is None else rel_logits.shape[2] - 256
        rel = q if rel_logits is None else rel_logits
        gate = q if gate is None else gate
        sinks = q if sinks is None else sinks
        if lse is None:
            lse = out

        rows = max_query_len * group
        if rows <= 64:
            block_m = 16 if rows <= 16 else 32 if rows <= 32 else 64
            # BN=64 beats BN=128 by 1.23-1.39x on SM86 for ALL global decode batches (N=1..16)!
            block_n = 64 if window == 0 else 128
            num_warps = 4
            num_stages = 2
        elif window > 0:
            if max_query_len <= 512:
                block_m, block_n, num_warps, num_stages = 128, 32, 4, 2
            else:
                block_m, block_n, num_warps, num_stages = 128, 64, 8, 2
        else:
            block_m, block_n, num_warps, num_stages = 128, 32, 4, 2

        tokens_per_tile = block_m // group
        num_reqs = cache_seqlens.shape[0]
        grid = (
            triton.cdiv(max_query_len, tokens_per_tile),
            num_kv_heads * num_splits,
            num_reqs,
        )
        if mode == attn_sm90.MODE_OUT_LSE:
            out_strides = (out.stride(0), out.stride(1), lse.stride(0), lse.stride(1))
        elif mode == attn_sm90.MODE_GATED:
            assert out.is_contiguous()
            out_strides = (out.stride(0), 0, 0, 0)
        else:
            assert out.is_contiguous() and lse.is_contiguous()
            out_strides = (0, 0, 0, 0)

        _rel_attention_sm86_kernel[grid](
            q, key_cache, value_cache, rel, block_table, cache_seqlens, cu_seqlens_q,
            out, lse, gate, sinks, softmax_scale, num_tokens, num_splits,
            q.stride(0), q.stride(1), key_cache.stride(0), key_cache.stride(1),
            key_cache.stride(2), rel.stride(0), block_table.stride(0),
            *out_strides, gate.stride(0), sinks.stride(0),
            H=num_heads, G=group, D=head_dim, E=extent, PAGE=page, WINDOW=window,
            DRAFT=draft, MODE=mode, BLOCK_M=block_m, BLOCK_N=block_n,
            num_warps=num_warps, num_stages=num_stages,
        )

    _SM86_REL_ATTENTION = _rel_attention_sm86
    return _rel_attention_sm86


def patch_module(fullname: str, module) -> None:
    if fullname in _PATCHED or not _env_flag("INKLING_SM86_OPT", "1"):
        return
    cap = _get_cap()
    if cap is None or cap >= (8, 9):
        return

    if fullname == "vllm.models.inkling2.nvidia.ops.attention_sm90":
        module.rel_attention = _build_rel_attention_sm86(module)
        _PATCHED.add(fullname)
        print("[inkling_sm86_opt] patched ops.attention_sm90 (unscaled E4M3 + BN=64 global decode N=1..16)", flush=True)

    elif fullname == "vllm.models.inkling2.nvidia.ops.attention":
        import vllm.models.inkling2.nvidia.ops.attention_sm90 as attn_sm90
        module.rel_attention = _build_rel_attention_sm86(attn_sm90)
        if _env_flag("INKLING_SM86_LOCAL_ATTN", "1"):
            _orig_plan = module.plan_decode_attention

            def _sm86_plan_decode_attention(
                is_local: bool,
                num_reqs: int,
                max_query_len: int,
                num_heads: int,
                num_kv_heads: int,
                rel_extent: int,
            ) -> int:
                sp = _orig_plan(is_local, num_reqs, max_query_len, num_heads, num_kv_heads, rel_extent)
                if sp == 0:
                    return 0
                if is_local:
                    return 1
                work = num_reqs * num_kv_heads
                if work <= 8:
                    return 20
                if work <= 16:
                    return 10
                # N>=4 (work>=32): sp=5 with BN=64 beats upstream sp=10/7/3 by 1.29-1.39x on SM86!
                return 5

            module.plan_decode_attention = _sm86_plan_decode_attention
        _PATCHED.add(fullname)
        print("[inkling_sm86_opt] patched ops.attention (local sp=1 + global sp=20/10/5)", flush=True)

    elif fullname == "vllm.models.inkling2.nvidia.attention":
        from vllm.compilation.breakable_cudagraph import eager_break_during_capture
        from vllm.forward_context import get_forward_context
        from vllm.triton_utils import triton
        from vllm.utils.platform_utils import num_compute_units
        import vllm.models.inkling2.nvidia.ops.attention_sm90 as attn_sm90
        import vllm.models.inkling2.nvidia.ops.attention as ops_attn

        fast_rel_attn = _build_rel_attention_sm86(attn_sm90)
        module.rel_attention = fast_rel_attn
        module.plan_decode_attention = ops_attn.plan_decode_attention

        def _ampere_prefill_splits(
            is_local: bool, max_q: int, max_kv: int, num_reqs: int,
            num_heads: int, num_kv_heads: int, nt: int, head_dim: int,
        ) -> int:
            if is_local or max_kv < 4096 or nt <= 0:
                return 1
            if max_q > 1024:
                return 1
            if max_q > 256:
                if max_kv < 8192 or num_reqs > 1:
                    return 1
                bytes_per_split = nt * num_heads * head_dim * 4
                return 2 if 2 * bytes_per_split <= 64 * 1024 * 1024 else 1
            group = num_heads // num_kv_heads
            block_m = 16 if max_q * group <= 16 else 32 if max_q * group <= 32 else 128
            m_tiles = triton.cdiv(max_q, max(1, block_m // group))
            base_ctas = m_tiles * num_kv_heads * num_reqs
            num_sms = num_compute_units()
            if base_ctas >= 2 * num_sms:
                return 1
            target = max(1, (2 * num_sms + base_ctas - 1) // base_ctas)
            sp = 1
            for cand in (2, 4, 8, 16):
                if cand <= target:
                    sp = cand
            bytes_per_split = nt * num_heads * head_dim * 4
            while sp > 1 and sp * bytes_per_split > 64 * 1024 * 1024:
                sp //= 2
            return sp

        def _patched_attention(
            self, positions, q, r, gate, sinks, k, v,
            k_dynamic, v_dynamic, tau, output, pass_idx=0,
        ):
            num_tokens = q.shape[0]
            q = q.view(num_tokens, self.num_heads, self.head_dim)
            attn_metadata = get_forward_context().attn_metadata
            layer_name, kv_cache = self._pass_cache(pass_idx)
            conv_state = self.conv_states[pass_idx]
            md = None
            if isinstance(attn_metadata, dict) and kv_cache.numel() > 0:
                md = cast(module.Inkling2AttentionMetadata, attn_metadata[layer_name])
            num_decodes, num_decode_tokens, decode_splits = 0, 0, 0
            if md is not None:
                num_decodes, num_decode_tokens, decode_splits = self._plan_decode(
                    md, num_tokens
                )
            clipped_q = q.new_empty(q.shape)
            rel_logits = q.new_empty(
                (num_tokens + 128, self.num_heads, self.rel_extent + 256)
            )
            if md is not None:
                conv_md = module.conv_metadata(conv_state)
                assert conv_md is not None
                key_cache, value_cache = self._split_kv_cache(kv_cache)
                module.attention_prologue(
                    q,
                    r.view(num_tokens, self.num_heads, self.d_rel),
                    k, v, self.q_scale, self.r_scale, tau,
                    self.rel_proj_shifted,
                    self.k_sconv.conv_weight, self.v_sconv.conv_weight,
                    k_dynamic, v_dynamic, conv_state.kv_cache, positions,
                    conv_md.block_table, conv_md.seq_idx, conv_md.slot_mapping,
                    conv_md.query_start, key_cache, value_cache, md.slot_mapping,
                    clipped_q, rel_logits, md.query_start_loc, md.seq_lens,
                    clip_threshold=self.qk_clip, rel_threshold=self.rel_clip,
                    norm_eps=self.eps,
                )
            if decode_splits:
                assert md is not None
                self._decode_attention(
                    md, kv_cache, clipped_q[:num_decode_tokens], rel_logits,
                    gate[:num_decode_tokens], sinks[:num_decode_tokens],
                    output[:num_decode_tokens], decode_splits, num_decodes,
                )
            if num_decode_tokens < num_tokens:
                out_slice = output[num_decode_tokens:]
                if md is None:
                    out_slice.zero_()
                    return
                nt = md.num_actual_tokens - (md.num_decode_tokens if num_decodes else 0)
                if nt <= 0:
                    out_slice.zero_()
                    return
                cu_seqlens_q = md.query_start_loc[num_decodes:]
                if num_decodes:
                    cu_seqlens_q = cu_seqlens_q - md.num_decode_tokens
                key_cache, value_cache = self._split_kv_cache(kv_cache)
                q_pf = clipped_q[num_decode_tokens : num_decode_tokens + nt]
                rel_pf = rel_logits[num_decode_tokens:]
                gate_pf = gate[num_decode_tokens : num_decode_tokens + nt]
                sinks_pf = sinks[num_decode_tokens : num_decode_tokens + nt]
                num_pf_reqs = md.seq_lens.shape[0] - num_decodes
                max_kv = (
                    int(md.seq_lens_cpu[num_decodes:].max())
                    if len(md.seq_lens_cpu) > num_decodes
                    else 0
                )
                sp = _ampere_prefill_splits(
                    self.is_local, md.max_query_len, max_kv, num_pf_reqs,
                    self.num_heads, self.num_kv_heads, nt, self.head_dim,
                )
                if sp == 1:
                    fast_rel_attn(
                        q_pf, key_cache, value_cache,
                        out_slice[:nt], None, rel_pf, gate_pf, sinks_pf,
                        block_table=md.block_table[num_decodes:],
                        cache_seqlens=md.seq_lens[num_decodes:],
                        cu_seqlens_q=cu_seqlens_q,
                        max_query_len=md.max_query_len,
                        softmax_scale=self.scaling,
                        window=self.sliding_window if self.is_local else 0,
                        num_splits=1,
                        mode=attn_sm90.MODE_GATED,
                    )
                else:
                    o_part = q_pf.new_empty(
                        (sp, nt, self.num_heads, self.head_dim), dtype=torch.float32
                    )
                    lse_part = q_pf.new_empty(
                        (sp, nt, self.num_heads), dtype=torch.float32
                    )
                    fast_rel_attn(
                        q_pf, key_cache, value_cache,
                        o_part, lse_part, rel_pf, gate_pf, sinks_pf,
                        block_table=md.block_table[num_decodes:],
                        cache_seqlens=md.seq_lens[num_decodes:],
                        cu_seqlens_q=cu_seqlens_q,
                        max_query_len=md.max_query_len,
                        softmax_scale=self.scaling,
                        window=0,
                        num_splits=sp,
                        mode=attn_sm90.MODE_PARTIAL,
                    )
                    if nt * self.num_heads >= 512:
                        block_d, num_warps = self.head_dim, 2 if sp > 32 else 1
                    else:
                        block_d, num_warps = 32, 4
                    ops_attn._rel_decode_combine_kernel[
                        (nt, self.num_heads, self.head_dim // block_d)
                    ](
                        o_part, lse_part, gate_pf, sinks_pf, out_slice[:nt],
                        nt, nt, gate_pf.stride(0), sinks_pf.stride(0),
                        H=self.num_heads, D=self.head_dim, NUM_SPLITS=sp,
                        BLOCK_S=triton.next_power_of_2(sp), BLOCK_D=block_d,
                        GATED=True, num_warps=num_warps,
                    )
                if nt < out_slice.shape[0]:
                    out_slice[nt:].zero_()

        module.Inkling2Attention._attention = eager_break_during_capture(_patched_attention)
        _PATCHED.add(fullname)
        print("[inkling_sm86_opt] patched attention (split-KV prefill up to q=1024 + fused gate/sink)", flush=True)

    elif fullname == "vllm.v1.core.kv_cache_planning":
        import math
        from vllm.utils.math_utils import cdiv
        from vllm.v1.kv_cache_interface import KVCacheGroup

        _orig_uniform = module._get_kv_cache_groups_uniform_page_size

        def _sm86_uniform_groups(layer_configs):
            layer_buckets = [list(b) for b in module._bucket_layers(layer_configs)]
            if len(layer_buckets) < 2:
                return _orig_uniform(layer_configs)
            lens = [len(b) for b in layer_buckets]
            min_len = min(lens)
            g = math.gcd(*lens)
            if g >= 2 and g < min_len and any(n % min_len != 0 for n in lens):
                grouped = []
                for layers in layer_buckets:
                    num_groups = cdiv(len(layers), g)
                    for i in range(num_groups):
                        grouped.append(layers[i::num_groups])
                print(
                    f"[inkling_sm86_opt] kv_cache_planning: 0-padding GCD group_size={g} for buckets {lens} ({len(grouped)} groups)",
                    flush=True,
                )
                return [
                    KVCacheGroup.from_layers({name: layer_configs[name] for name in names})
                    for names in grouped
                ]
            return _orig_uniform(layer_configs)

        module._get_kv_cache_groups_uniform_page_size = _sm86_uniform_groups
        _PATCHED.add(fullname)
        print("[inkling_sm86_opt] patched kv_cache_planning (0-padding GCD group sizing)", flush=True)

    elif fullname == "vllm.models.inkling2.nvidia.ops.moe_sm90" and cap == (8, 6):
        _orig_humming_init = module.HummingExperts.__init__

        def _patched_humming_init(self, weights, shapes, nvfp4, input_schema=None):
            _orig_humming_init(self, weights, shapes, nvfp4, input_schema)
            if self.num_experts != 128 or shapes != ((2304, 2304), (2304, 1152)):
                return
            for name in ("w13", "w2"):
                tuning = self.tunings[name]
                for lo, hi, cfg in tuning:
                    if nvfp4:
                        if lo == 16 and hi == 912:
                            cfg["block_shape"] = (8, 256, 64)
                            cfg["warp_shape"] = (8, 64, 64)
                            cfg["num_stages"] = 3
                            cfg["num_ctas_per_sm"] = 2
                        elif lo == 912 and hi == 1840:
                            if name == "w13":
                                cfg["block_shape"] = (16, 128, 128)
                                cfg["warp_shape"] = (16, 32, 128)
                                cfg["num_stages"] = 2
                                cfg["num_ctas_per_sm"] = 2
                            else:
                                cfg["block_shape"] = (16, 256, 64)
                                cfg["warp_shape"] = (16, 32, 64)
                                cfg["num_stages"] = 3
                                cfg["num_ctas_per_sm"] = 2
                        elif lo == 1840 and hi == 2048:
                            cfg["block_shape"] = (32, 128, 64)
                            cfg["warp_shape"] = (32, 32, 64)
                            cfg["num_stages"] = 3
                            cfg["num_ctas_per_sm"] = 2
                        elif name == "w2" and lo == 3680 and hi == 5504:
                            cfg["block_shape"] = (32, 256, 64)
                            cfg["warp_shape"] = (32, 32, 64)
                            cfg["num_stages"] = 2
                            cfg["num_ctas_per_sm"] = 2
                        elif lo >= 5504:
                            cfg["block_shape"] = (64, 256, 64)
                            cfg["warp_shape"] = (64, 64, 64)
                            cfg["num_stages"] = 2
                            cfg["num_ctas_per_sm"] = 2
                    else:
                        if lo == 32 and hi == 912:
                            cfg["block_shape"] = (16, 256, 64)
                            cfg["warp_shape"] = (16, 32, 64)
                            cfg["num_stages"] = 2
                            cfg["num_ctas_per_sm"] = 2
                        elif lo == 912 and hi == 1840:
                            if name == "w13":
                                cfg["block_shape"] = (32, 128, 64)
                                cfg["warp_shape"] = (32, 32, 64)
                                cfg["num_stages"] = 3
                                cfg["num_ctas_per_sm"] = 2
                            else:
                                cfg["block_shape"] = (16, 128, 64)
                                cfg["warp_shape"] = (16, 32, 64)
                                cfg["num_stages"] = 3
                                cfg["num_ctas_per_sm"] = 2
                        elif lo == 1840 and hi == 2048:
                            cfg["block_shape"] = (32, 256, 64)
                            cfg["warp_shape"] = (32, 32, 64) if name == "w13" else (16, 64, 64)
                            cfg["num_stages"] = 2
                            cfg["num_ctas_per_sm"] = 2
                        elif name == "w2" and lo == 3680 and hi == 5504:
                            cfg["block_shape"] = (32, 256, 64)
                            cfg["warp_shape"] = (32, 64, 64)
                            cfg["num_stages"] = 2
                            cfg["num_ctas_per_sm"] = 2
                        elif lo >= 8192:
                            cfg["block_shape"] = (64, 256, 64)
                            cfg["warp_shape"] = (64, 32, 64)
                            cfg["num_stages"] = 3
                            cfg["num_ctas_per_sm"] = 1
                    cfg["use_stream_k"] = False
                self.tuning_strs[name] = json.dumps(tuning)

        module.HummingExperts.__init__ = _patched_humming_init
        _PATCHED.add(fullname)
        print("[inkling_sm86_opt] patched ops.moe_sm90 (SM86 v3 concurrency Humming schedules)", flush=True)

    elif fullname == "vllm.models.inkling2.nvidia.ops.linear":
        _orig_linear = module.linear
        bf16_gemv_sm86, int8_w8a16_linear = _get_linear_kernels()
        use_gemv = _env_flag("INKLING_SM86_GEMV", "1")

        def _sm86_linear(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None = None):
            if weight.dtype == torch.int8:
                scale = getattr(weight, "_sm86_int8_scale", None)
                assert scale is not None, "int8 weight missing _sm86_int8_scale"
                return int8_w8a16_linear(x, weight, scale, out=out)
            if (
                use_gemv
                and out is None
                and x.is_cuda
                and x.dtype == weight.dtype == torch.bfloat16
                and x.ndim == 2
                and 0 < x.shape[0] <= 16
                and x.stride(-1) == 1
                and weight.stride(-1) == 1
            ):
                k, n = weight.shape[1], weight.shape[0]
                # Only qkvr_local beats cuBLAS on SM86; wo_ud (3072->2304) is 2-3x faster on cuBLAS!
                if k == 2304 and n == 8600:
                    return bf16_gemv_sm86(x, weight)
            return _orig_linear(x, weight, out)

        module.linear = _sm86_linear
        _PATCHED.add(fullname)
        print("[inkling_sm86_opt] patched ops.linear (qkvr_local GEMV + cuBLAS wo_ud + v3 INT8 W8A16)", flush=True)

    elif fullname == "vllm.models.inkling2.nvidia.ops.dspark":
        if _env_flag("INKLING_SM86_DSPARK_FP8", "1"):
            from vllm.triton_utils import tl, triton

            @triton.jit
            def _qk_norm_rope_onthefly_kernel(
                X, Q_WEIGHT, K_WEIGHT, INV_FREQ, POS,
                row_stride, q_size,
                NHQ: tl.constexpr, D: tl.constexpr, EPS: tl.constexpr, NORM: tl.constexpr,
            ):
                t = tl.program_id(0).to(tl.int64)
                h = tl.program_id(1)
                pos = tl.load(POS + t).to(tl.float32)

                HALF: tl.constexpr = D // 2
                half_ar = tl.arange(0, HALF)
                d_ar = tl.arange(0, D)
                inv_freq = tl.load(INV_FREQ + half_ar).to(tl.float32)
                angles = pos * inv_freq
                cos = tl.cos(angles)
                sin = tl.sin(angles)

                is_q = h < NHQ
                col0 = tl.where(is_q, h * D, q_size + (h - NHQ) * D).to(tl.int64)
                row = X + t * row_stride + col0
                if NORM:
                    w_ptr = tl.where(is_q, Q_WEIGHT.to(tl.int64), K_WEIGHT.to(tl.int64)).to(
                        tl.pointer_type(tl.bfloat16)
                    )
                    x = tl.load(row + d_ar).to(tl.float32)
                    ms = tl.sum(x * x, 0) / D
                    inv = 1.0 / tl.sqrt(ms + EPS)
                    w1 = tl.load(w_ptr + half_ar).to(tl.float32)
                    w2 = tl.load(w_ptr + HALF + half_ar).to(tl.float32)
                    x1 = tl.load(row + half_ar).to(tl.float32) * inv * w1
                    x2 = tl.load(row + HALF + half_ar).to(tl.float32) * inv * w2
                    x1 = x1.to(tl.bfloat16).to(tl.float32)
                    x2 = x2.to(tl.bfloat16).to(tl.float32)
                else:
                    x1 = tl.load(row + half_ar).to(tl.float32)
                    x2 = tl.load(row + HALF + half_ar).to(tl.float32)
                o1 = x1 * cos - x2 * sin
                o2 = x2 * cos + x1 * sin
                tl.store(row + half_ar, o1.to(tl.bfloat16))
                tl.store(row + HALF + half_ar, o2.to(tl.bfloat16))

            _orig_qk_norm_rope = module.qk_norm_rope_
            _orig_rope = module.rope_

            def _sm86_qk_norm_rope_(qkv, positions, q_weight, k_weight, cos_sin_cache, num_q_heads, num_k_heads, eps):
                inv_freq = getattr(cos_sin_cache, "_sm86_inv_freq", None)
                if inv_freq is None:
                    return _orig_qk_norm_rope(qkv, positions, q_weight, k_weight, cos_sin_cache, num_q_heads, num_k_heads, eps)
                head_dim = cos_sin_cache.shape[-1]
                if qkv.shape[0]:
                    _qk_norm_rope_onthefly_kernel[(qkv.shape[0], num_q_heads + num_k_heads)](
                        qkv, q_weight, k_weight, inv_freq, positions,
                        qkv.stride(0), num_q_heads * head_dim,
                        NHQ=num_q_heads, D=head_dim, EPS=eps, NORM=True,
                    )

            def _sm86_rope_(x, positions, cos_sin_cache):
                inv_freq = getattr(cos_sin_cache, "_sm86_inv_freq", None)
                if inv_freq is None:
                    return _orig_rope(x, positions, cos_sin_cache)
                num_tokens, num_heads, head_dim = x.shape
                if num_tokens:
                    _qk_norm_rope_onthefly_kernel[(num_tokens, num_heads)](
                        x, x, x, inv_freq, positions,
                        x.stride(0), 0,
                        NHQ=0, D=head_dim, EPS=0.0, NORM=False,
                    )

            module.qk_norm_rope_ = _sm86_qk_norm_rope_
            module.rope_ = _sm86_rope_
            print("[inkling_sm86_opt] patched ops.dspark (zero-VRAM on-the-fly RoPE)", flush=True)
        _PATCHED.add(fullname)

    elif fullname == "vllm.models.inkling2.nvidia.dspark":
        if _env_flag("INKLING_SM86_DSPARK_FP8", "1"):
            from vllm.config import get_current_vllm_config
            from vllm.distributed import tensor_model_parallel_all_gather
            from vllm.model_executor.offloader import get_offloader
            from vllm.utils.torch_utils import canonicalize_singleton_dim_strides
            import vllm.models.inkling2.nvidia.ops.dspark as ops_dspark
            import vllm.models.inkling2.nvidia.ops.linear as ops_linear
            import vllm.models.inkling2.nvidia.ops.attention as ops_attn

            module.qk_norm_rope_ = ops_dspark.qk_norm_rope_
            module.rope_ = ops_dspark.rope_
            module.linear = ops_linear.linear
            module.plan_decode_attention = ops_attn.plan_decode_attention
            module.block_decode_attention = ops_attn.block_decode_attention

            write_fp8_kv = _get_fp8_kv_writer()
            _, int8_w8a16_linear = _get_linear_kernels()

            # 1) DSpark Attention: use FP8 KV cache matching target (--kv-cache-dtype fp8)
            from vllm.models.inkling2.nvidia.attention import _kv_cache_layer_config
            _orig_draft_attn_init = module.Inkling2DraftSelfAttention.__init__

            def _patched_draft_attn_init(self, config, is_local, prefix):
                _orig_draft_attn_init(self, config, is_local, prefix)
                vllm_config = get_current_vllm_config()
                if vllm_config.cache_config.cache_dtype in ("fp8", "fp8_e4m3"):
                    sw = config.sliding_window_size if is_local else None
                    nkv, hd = self.num_kv_heads, self.head_dim
                    self.attn.get_kv_cache_config = (
                        lambda vc, _nkv=nkv, _hd=hd, _sw=sw: _kv_cache_layer_config(vc, _nkv, _hd, _sw)
                    )

            _orig_write_kv = module.Inkling2DraftSelfAttention._write_kv

            def _patched_write_kv(self, k, v, slot_mapping):
                if self.attn.kv_cache.element_size() == 1:
                    write_fp8_kv(k, v, self.attn.kv_cache, slot_mapping)
                else:
                    _orig_write_kv(self, k, v, slot_mapping)

            def _patched_draft_attn_forward(self, positions, x, cos_sin_cache):
                qkv = module.linear(x, self.qkv_proj.weight)
                module.qk_norm_rope_(
                    qkv, positions, self.q_norm.weight, self.k_norm.weight,
                    cos_sin_cache, self.num_heads, self.num_kv_heads, self.eps,
                )
                q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
                out = q.new_zeros((q.shape[0], self.q_size))
                md = self._metadata()
                if md is not None:
                    self._write_kv(
                        k.view(-1, self.num_kv_heads, self.head_dim),
                        v.view(-1, self.num_kv_heads, self.head_dim),
                        md.slot_mapping[: q.shape[0]],
                    )
                    kc = self.attn.kv_cache
                    if kc.element_size() == 1:
                        kc = kc.view(torch.uint8)
                    key_cache, value_cache = kc.transpose(1, 2).split(self.head_dim, dim=-1)
                    num_reqs = min(q.shape[0] // self.block, md.seq_lens.shape[0])
                    num_tokens = num_reqs * self.block
                    num_splits = max(
                        1,
                        module.plan_decode_attention(
                            self.is_local, num_reqs, self.block,
                            self.num_heads, self.num_kv_heads, self.window,
                        ),
                    )
                    module.block_decode_attention(
                        q[:num_tokens].view(num_tokens, self.num_heads, self.head_dim),
                        canonicalize_singleton_dim_strides(key_cache),
                        canonicalize_singleton_dim_strides(value_cache),
                        out[:num_tokens],
                        block_table=md.block_table[:num_reqs],
                        cache_seqlens=md.seq_lens[:num_reqs],
                        cu_seqlens_q=self.cu_seqlens_q[: num_reqs + 1],
                        block=self.block,
                        window=self.window,
                        num_splits=num_splits,
                    )
                return module.linear(out, self.o_proj.weight)

            module.Inkling2DraftSelfAttention.__init__ = _patched_draft_attn_init
            module.Inkling2DraftSelfAttention._write_kv = _patched_write_kv
            module.Inkling2DraftSelfAttention.forward = _patched_draft_attn_forward

            # 2) Zero-VRAM on-the-fly RoPE + UVA offload for markov_w1 (98 MiB)
            _orig_decoder_init = module.Inkling2DSparkDecoder.__init__

            def _patched_decoder_init(self, config, max_model_len, prefix):
                _orig_decoder_init(self, config, 1, prefix)
                head_dim = config.head_dim
                theta = float(config.rope_theta)
                inv_freq = (
                    1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
                ).to(dtype=torch.float32, device=self.cos_sin_cache.device)
                self.register_buffer("_sm86_inv_freq", inv_freq, persistent=False)
                self.cos_sin_cache._sm86_inv_freq = self._sm86_inv_freq
                get_offloader().wrap_modules(
                    iter([self.markov_head.markov_w1]),
                    prefix=f"{prefix}.markov_head.markov_w1" if prefix else "markov_head.markov_w1",
                )

            module.Inkling2DSparkDecoder.__init__ = _patched_decoder_init

            # 3) INT8 per-row fold_weight + Fused markov_step_softcap!
            @torch.no_grad()
            def _patched_fold(self, lm_head_weight: torch.Tensor) -> None:
                projection = self.hidden_out.weight.double()
                bias = self.hidden_out.bias.double()
                num_rows = lm_head_weight.shape[0]
                lm_scale = getattr(lm_head_weight, "_sm86_int8_scale", None)
                if self.fold_weight is None or self.fold_bias is None:
                    self.fold_weight = torch.empty(
                        (num_rows, self.markov_rank), dtype=torch.int8, device=lm_head_weight.device
                    )
                    self.fold_scale = torch.empty(
                        (num_rows,), dtype=torch.float32, device=lm_head_weight.device
                    )
                    self.fold_bias = torch.empty(
                        (num_rows,), dtype=torch.bfloat16, device=lm_head_weight.device
                    )
                for start in range(0, num_rows, 4096):
                    rows = slice(start, start + 4096)
                    if lm_scale is not None:
                        head = lm_head_weight[rows].double() * lm_scale[rows].double()[:, None]
                    else:
                        head = lm_head_weight[rows].double()
                    fw = (head @ projection).float()
                    fb = (head @ bias).to(torch.bfloat16)
                    w_q, sc = _quantize_int8_per_row(fw)
                    self.fold_weight[rows].copy_(w_q)
                    self.fold_scale[rows].copy_(sc)
                    self.fold_bias[rows].copy_(fb)

            def _patched_markov_step(self, state, prev_tokens, context):
                assert self.fold_weight is not None and self.fold_bias is not None
                inputs = torch.cat([state, self.markov_w1(prev_tokens)], dim=-1)
                weight = self.rnn_proj.weight[:, : 2 * self.markov_rank]
                raw = torch.addmm(context, inputs, weight.T, out_dtype=torch.float32)
                state, latent = module.markov_gate(raw, state)
                out = int8_w8a16_linear(
                    latent, self.fold_weight, self.fold_scale, bias=self.fold_bias, out_dtype=torch.bfloat16
                )
                return state, out

            def _fused_markov_step_softcap(self, state, prev_tokens, context, base_logits, softcap: float):
                assert self.fold_weight is not None and self.fold_bias is not None
                inputs = torch.cat([state, self.markov_w1(prev_tokens)], dim=-1)
                weight = self.rnn_proj.weight[:, : 2 * self.markov_rank]
                raw = torch.addmm(context, inputs, weight.T, out_dtype=torch.float32)
                state, latent = module.markov_gate(raw, state)
                out = int8_w8a16_linear(
                    latent, self.fold_weight, self.fold_scale,
                    bias=self.fold_bias, out_dtype=torch.bfloat16,
                    base=base_logits, softcap=softcap,
                )
                return state, out

            module.Inkling2MarkovHead.fold = _patched_fold
            module.Inkling2MarkovHead.step = _patched_markov_step
            module.Inkling2MarkovHead.step_softcap = _fused_markov_step_softcap

            def _model_markov_step_logits(self, state, prev_tokens, context, base_logits):
                state, logits = self.model.markov_head.step_softcap(
                    state, prev_tokens, context, base_logits, self.config.final_logit_softcapping
                )
                logits = tensor_model_parallel_all_gather(logits)
                return state, logits[:, : self.config.vocab_size]

            module.Inkling2DSparkModel.markov_step_logits = _model_markov_step_logits

            # 4) Allow INT8 lm_head in attach_target & quantize DSpark 3-layer dense weights to INT8
            @torch.no_grad()
            def _patched_attach_target(self, target_model):
                assert self.lm_head is not None and self.lm_head.weight.dtype in (torch.bfloat16, torch.int8)
                self.model.embed_norm = target_model.model.embed_norm
                head = self.model.markov_head
                head.rnn_bias = torch.empty_like(head.rnn_proj.bias, dtype=torch.float32)
                self.refresh_derived_weights()

            module.Inkling2DSparkModel.attach_target = _patched_attach_target

            _orig_dspark_load = module.Inkling2DSparkModel.load_weights

            def _patched_dspark_load(self, weights):
                loaded = _orig_dspark_load(self, weights)
                saved_bytes = 0
                for layer in self.model.layers:
                    for mod in (
                        layer.self_attn.qkv_proj,
                        layer.self_attn.kv_proj_ctx,
                        layer.self_attn.o_proj,
                        layer.mlp.gate_up_proj,
                        layer.mlp.down_proj,
                    ):
                        w = mod.weight
                        if w.dtype == torch.bfloat16:
                            orig_b = w.numel() * w.element_size()
                            w_q, sc = _quantize_int8_per_row(w.data)
                            w_q._sm86_int8_scale = sc
                            del mod._parameters["weight"]
                            mod.register_buffer("weight", w_q, persistent=False)
                            mod.register_buffer("_sm86_int8_scale", sc, persistent=False)
                            saved_bytes += orig_b - (w_q.numel() + sc.numel() * 4)
                print(
                    f"[inkling_sm86_opt] DSpark FP8 KV + zero-VRAM RoPE + markov_w1 UVA + INT8 draft/fold + fused softcap (saved {saved_bytes/1048576:.1f} MiB draft weights)",
                    flush=True,
                )
                return loaded

            module.Inkling2DSparkModel.load_weights = _patched_dspark_load
            print("[inkling_sm86_opt] patched dspark (FP8 KV + RoPE + UVA + INT8 + fused markov_softcap)", flush=True)
        _PATCHED.add(fullname)

    elif fullname == "vllm.models.inkling2.speculator":
        if _env_flag("INKLING_SM86_FUSE_MARKOV_SOFTCAP", "1"):
            from vllm.v1.worker.gpu.sample.greedy_argmax import greedy_argmax

            def _patched_sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
                n_spec = self.num_speculative_steps
                num_sample = num_reqs * n_spec
                hidden = head_hidden[self.sample_indices[:num_sample]]
                base_logits = self.model.compute_base_logits(hidden).view(num_reqs, n_spec, -1)
                context = self.model.model.markov_head.context(hidden, self.f_rnn[:num_reqs])
                idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
                sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
                state = hidden.new_zeros(num_reqs, self.model.config.markov_rank)
                prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]
                use_fused = hasattr(self.model, "markov_step_logits")
                for i in range(n_spec):
                    if use_fused:
                        state, logits = self.model.markov_step_logits(
                            state, prev, context[i], base_logits[:, i]
                        )
                        if self.draft_logits is None:
                            prev = greedy_argmax(logits)
                        else:
                            prev = self._sample_logits(logits, idx_map[:, i], sample_pos[:, i], i)
                    else:
                        state, bias = self.model.markov_step(state, prev, context[i])
                        if self.draft_logits is None:
                            prev = self.model.markov_argmax(base_logits[:, i], bias)
                        else:
                            logits = self.model.markov_logits(base_logits[:, i], bias)
                            prev = self._sample_logits(logits, idx_map[:, i], sample_pos[:, i], i)
                    self.draft_tokens[:num_reqs, i] = prev

            module.Inkling2DSparkSpeculator._sample_sequential = _patched_sample_sequential
            print("[inkling_sm86_opt] patched speculator (fused markov_fold + add_softcap)", flush=True)
        _PATCHED.add(fullname)

    elif fullname == "vllm.models.inkling2.nvidia.model":
        from vllm.model_executor.offloader import get_offloader
        _, int8_w8a16_linear = _get_linear_kernels()

        _orig_model_init = module.Inkling2ForConditionalGeneration.__init__

        def _patched_model_init(self, *, vllm_config, prefix=""):
            _orig_model_init(self, vllm_config=vllm_config, prefix=prefix)
            if self.visual is not None:
                get_offloader().wrap_modules(
                    iter([self.visual]), prefix=f"{prefix}visual"
                )

        module.Inkling2ForConditionalGeneration.__init__ = _patched_model_init

        _orig_apply_head = module.Inkling2LogitsProcessor._apply_head

        def _patched_apply_head(self, lm_head, hidden_states, embedding_bias):
            w = lm_head.weight
            if w.dtype == torch.int8:
                scale = getattr(w, "_sm86_int8_scale", None)
                assert scale is not None
                return int8_w8a16_linear(
                    hidden_states, w, scale, bias=embedding_bias, out_dtype=self.head_dtype
                )
            return _orig_apply_head(self, lm_head, hidden_states, embedding_bias)

        module.Inkling2LogitsProcessor._apply_head = _patched_apply_head

        _orig_load = module.Inkling2ForConditionalGeneration.load_weights
        edit_path = os.environ.get("INKLING_UNBOUND_EDIT", "")
        quant_lm_head = _env_flag("INKLING_SM86_LMHEAD_INT8", "0")

        def _patched_load_weights(self, weights):
            loaded = _orig_load(self, weights)
            if edit_path:
                edit = torch.load(edit_path, weights_only=False, map_location="cpu")
                params = dict(self.named_parameters())
                applied = 0
                old_threads = torch.get_num_threads()
                torch.set_num_threads(1)
                try:
                    for ckpt_name, t in sorted(edit["tensors"].items()):
                        vllm_name = ckpt_name.replace("model.llm.", "model.")
                        param = params[vllm_name]
                        W = param.data.detach().cpu()
                        X = W.float() + t["A"] @ t["B"].T
                        if "r" in t:
                            X = X * t["r"][:, None]
                        edited_bf16 = X.to(torch.bfloat16).contiguous()
                        digest = hashlib.sha256(
                            edited_bf16.view(torch.uint8).numpy().tobytes()
                        ).hexdigest()
                        assert digest == t["sha256_edited"], (
                            f"Unbound SHA-256 mismatch on {ckpt_name}"
                        )
                        param.data.copy_(edited_bf16)
                        applied += 1
                finally:
                    torch.set_num_threads(old_threads)
                print(
                    f"[unbound_edit_hook] applied & SHA-256 verified {applied}/20 Unbound BF16 wo_ud tensors!",
                    flush=True,
                )
            if quant_lm_head and self.lm_head.weight.dtype == torch.bfloat16:
                w = self.lm_head.weight
                orig_b = w.numel() * w.element_size()
                w_q, sc = _quantize_int8_per_row(w.data)
                w_q._sm86_int8_scale = sc
                del self.lm_head._parameters["weight"]
                self.lm_head.register_buffer("weight", w_q, persistent=False)
                self.lm_head.register_buffer("_sm86_int8_scale", sc, persistent=False)
                saved = (orig_b - (w_q.numel() + sc.numel() * 4)) / 1048576
                print(
                    f"[inkling_sm86_opt] quantized lm_head to per-row INT8 W8A16 (saved {saved:.1f} MiB VRAM)",
                    flush=True,
                )
            return loaded

        module.Inkling2ForConditionalGeneration.load_weights = _patched_load_weights
        _PATCHED.add(fullname)
        print(
            f"[inkling_sm86_opt] patched model (visual UVA + unbound={bool(edit_path)} + lm_head_int8={quant_lm_head})",
            flush=True,
        )

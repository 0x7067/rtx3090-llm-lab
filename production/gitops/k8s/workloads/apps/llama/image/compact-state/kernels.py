"""Indexed GDN verification and accepted/prefix-boundary reconstruction.

Recurrence follows vLLM's Apache-2.0 fused_sigmoid_gating kernel, derived from
flash-linear-attention (MIT, Copyright 2023-2025 Songlin Yang, Yu Zhang).
The serving hooks retain the existing cache format and reconstruct required
snapshots before scheduler copies or prefix offload.
"""

from vllm.triton_utils import tl, triton


@triton.jit
def verify_kernel(
    Q,
    Ks,
    Vs,
    A,
    Bias,
    Alog,
    B,
    State,
    Out,
    Delta,
    Keys,
    Decay,
    SoftplusBeta,
    Threshold,
    Scale,
    Lengths,
    Indices,
    Previous,
    Source,
    SS: tl.constexpr,
    IS: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    D: tl.constexpr,
    BV: tl.constexpr,
):
    iv, nh = tl.program_id(0), tl.program_id(1)
    n, hv = nh // HV, nh % HV
    h = hv // (HV // H)
    kk = tl.arange(0, D)
    vv = iv * BV + tl.arange(0, BV)
    start = tl.load(Lengths + n)
    end = tl.load(Lengths + n + 1)
    previous = tl.load(Previous + n)
    source = tl.load(
        Indices + n * IS + previous - 1, mask=(previous > 0) & (previous <= IS), other=0
    )
    if iv == 0 and hv == 0:
        tl.store(Source + n, source)
    sp = State + source * SS + (hv * D + vv[:, None]) * D + kk[None, :]
    state = tl.load(sp, mask=source > 0, other=0).to(tl.float32)
    alog = tl.load(Alog + hv).to(tl.float32)
    bias = tl.load(Bias + hv).to(tl.float32)
    for t in range(end - start):
        q = tl.load(Q + ((start + t) * H + h) * D + kk).to(tl.float32)
        k = tl.load(Ks + ((start + t) * H + h) * D + kk).to(tl.float32)
        v = tl.load(Vs + ((start + t) * HV + hv) * D + vv).to(tl.float32)
        x = tl.load(A + (start + t) * HV + hv).to(tl.float32) + bias
        g = -tl.exp(alog) * tl.where(
            SoftplusBeta * x <= Threshold,
            (1 / SoftplusBeta) * tl.log(1 + tl.exp(SoftplusBeta * x)),
            x,
        )
        decay = tl.exp(g)
        beta = tl.sigmoid(tl.load(B + (start + t) * HV + hv).to(tl.float32))
        q = q * tl.rsqrt(tl.sum(q * q) + 1e-6)
        k = k * tl.rsqrt(tl.sum(k * k) + 1e-6)
        q = q * Scale
        state *= decay
        v -= tl.sum(state * k[None, :], 1)
        v *= beta
        state += v[:, None] * k[None, :]
        output = tl.sum(state * q[None, :], 1)
        tl.store(
            Out + ((start + t) * HV + hv) * D + vv, tl.where(source > 0, output, 0)
        )
        tl.store(Delta + ((n * T + t) * HV + hv) * D + vv, v)
        if iv == 0:
            tl.store(Decay + (n * T + t) * HV + hv, decay)
            if hv % (HV // H) == 0:
                tl.store(Keys + ((n * T + t) * H + h) * D + kk, k)


@triton.jit
def restore_kernel(
    State,
    Delta,
    Keys,
    Decay,
    Accepted,
    Source,
    Indices,
    Lengths,
    NewComputed,
    Mapping,
    BlockSize: tl.constexpr,
    SS: tl.constexpr,
    IS: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    D: tl.constexpr,
    BV: tl.constexpr,
):
    iv, nh = tl.program_id(0), tl.program_id(1)
    n, hv = nh // HV, nh % HV
    h = hv // (HV // H)
    count = tl.maximum(tl.load(Accepted + n), 1)
    req_idx = tl.load(Mapping + n)
    length = tl.load(Lengths + n + 1) - tl.load(Lengths + n)
    source = tl.load(Source + n)
    new_computed = tl.load(NewComputed + req_idx, mask=req_idx >= 0, other=0)
    running = new_computed - count + 1
    aligned = ((running + count - 1) // BlockSize) * BlockSize
    boundary = aligned - running + 1
    if (source > 0) & (req_idx >= 0) & (count <= length) & (count <= T):
        kk = tl.arange(0, D)
        vv = iv * BV + tl.arange(0, BV)
        offset = (hv * D + vv[:, None]) * D + kk[None, :]
        state = tl.load(State + source * SS + offset).to(tl.float32)
        for t in range(count):
            k = tl.load(Keys + ((n * T + t) * H + h) * D + kk)
            v = tl.load(Delta + ((n * T + t) * HV + hv) * D + vv)
            decay = tl.load(Decay + (n * T + t) * HV + hv)
            state = tl.extra.cuda.libdevice.mul_rn(state, decay)
            state = tl.fma(v[:, None], k[None, :], state)
            if (t + 1 == count) | (t + 1 == boundary):
                dest = tl.load(Indices + n * IS + t)
                tl.store(State + dest * SS + offset, state, mask=dest > 0)

"""Minimal GPTQ (Frantar et al.) for symmetric group-wise int quantization, plus
Hessian accumulation from calibration activations. Output format matches the
compressed-tensors pack-quantized layout used elsewhere in this repo
(int8 codes in [-2^(b-1), 2^(b-1)-1], fp16 per-group scales, group along K).
"""
import torch


def accumulate_hessian(H, X, n_seen):
    """H: [K,K] fp32 running (2/n) * X^T X ; X: [n,K]. Returns (H, n_seen) updated (GPTQ's scaling)."""
    X = X.float()
    n = X.shape[0]
    H *= n_seen / (n_seen + n)
    n_seen += n
    X = (2.0 / n_seen) ** 0.5 * X
    H += X.t() @ X
    return H, n_seen


@torch.no_grad()
def gptq_quantize(W, H, bits=4, group=128, blocksize=128, percdamp=0.01, mse_clip=False):
    """W: [N,K] (any float dtype), H: [K,K] fp32 Hessian of the layer input.
    Returns q int8 [N,K], scale fp32 [N, K//group]. Column order is preserved (no act-order),
    so no g_idx is needed by Marlin."""
    dev = W.device
    W = W.float().clone()
    N, K = W.shape
    assert K % group == 0
    qmax = 2 ** (bits - 1) - 1
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    damp = percdamp * torch.mean(torch.diag(H))
    H += torch.eye(K, device=dev) * damp
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    Q = torch.zeros_like(W)
    scales = torch.zeros(N, K // group, device=dev)
    scale = None
    assert blocksize == group, "one quantization group per GPTQ block keeps the scale logic simple"
    for i1 in range(0, K, blocksize):
        i2 = i1 + blocksize
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        # group scale from the current (error-compensated) weights of this group
        amax = W1.abs().amax(dim=1)
        scale = torch.clamp(amax / qmax, min=1e-8)
        if mse_clip:
            best = scale.clone(); best_err = torch.full_like(scale, float("inf"))
            for f in torch.linspace(0.75, 1.0, 11, device=dev):
                s_ = torch.clamp(amax * f / qmax, min=1e-8)
                q_ = torch.clamp(torch.round(W1 / s_[:, None]), -qmax - 1, qmax) * s_[:, None]
                e = ((q_ - W1) ** 2).sum(1)
                better = e < best_err
                best[better] = s_[better]; best_err[better] = e[better]
            scale = best
        scales[:, i1 // group] = scale
        for i in range(blocksize):
            w = W1[:, i]
            d = Hinv1[i, i]
            q = torch.clamp(torch.round(w / scale), -qmax - 1, qmax) * scale
            Q1[:, i] = q
            err = (w - q) / d
            W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    q_int = torch.round(Q / scales.repeat_interleave(group, dim=1)).clamp(-qmax - 1, qmax).to(torch.int8)
    return q_int, scales


@torch.no_grad()
def rtn_quantize(W, bits=4, group=128):
    W = W.float()
    N, K = W.shape
    qmax = 2 ** (bits - 1) - 1
    g = W.reshape(N, K // group, group)
    scale = torch.clamp(g.abs().amax(dim=-1) / qmax, min=1e-8)
    q = torch.clamp(torch.round(g / scale[..., None]), -qmax - 1, qmax).to(torch.int8).reshape(N, K)
    return q, scale


def dequant(q, scale, group=128):
    N, K = q.shape
    return (q.float().reshape(N, K // group, group) * scale[..., None]).reshape(N, K)

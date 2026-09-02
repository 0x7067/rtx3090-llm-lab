#!/usr/bin/env python3
"""VRAM budget for Qwen3.8-27B-INT4-RedHatAI on one 24 GiB RTX 3090 (sm86).

Reimplements SGLang's hybrid-GDN pool sizing so we can predict, off-GPU,
whether a given flag set boots or dies with

    RuntimeError: Hybrid (mamba/linear-attention) state cache is too small to
    serve any requests. max_mamba_cache_size=K, mamba_ratio=R, ...

Sources (read at PR base 7088f2192):
  python/sglang/srt/mem_cache/kv_cache_configurator.py
    _calculate_mamba_ratio()            -> R
    _resolve_memory_pool_config()        -> K and the intermediate reservation
    MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO = 3
    MAMBA_CACHE_BASE_RATIO_DROP_ON_SKIP  = 1
    MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP      = 2
    MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP_LAZY = 1
    MAMBA_CACHE_V2_ADDITIONAL_RATIO_NO_OVERLAP   = 1
    MAMBA_CACHE_V2_ADDITIONAL_RATIO_NO_BUFFER    = 1
  https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B
    state_bytes 153.9 MB fp32 / 78.4 MB bf16; kv 32.8 KB fp8 / 65.5 KB bf16
"""

GIB = 1 << 30
GPU_TOTAL = 24576 / 1024 * GIB          # 24576 MiB reported by nvidia-smi
CUDA_CTX = 0.35 * GIB                    # driver + CUDA context + NCCL slack

# Measured from the checkpoint's safetensors headers, not estimated.
W_MAIN = 18_603_383_720                  # model.safetensors
W_MTP = 849_400_392                      # model_mtp.safetensors (draft weights)

STATE_BYTES = {"float32": 153.9e6, "bfloat16": 78.4e6, "float16": 78.4e6}
KV_BYTES_TOK = {"fp8_e5m2": 32.8e3, "fp8_e4m3": 32.8e3, "auto": 65.5e3,
                "bf16": 65.5e3, "bfloat16": 65.5e3}


def mamba_ratio(disable_radix, strategy, overlap, skip_decode_lock):
    """_calculate_mamba_ratio()."""
    if disable_radix:
        return 1
    base = 3 - (1 if skip_decode_lock else 0)
    extra = 0
    if strategy in ("extra_buffer", "extra_buffer_lazy", "auto"):
        if overlap:
            extra = 1 if strategy == "extra_buffer_lazy" else 2
        else:
            # "Lazy extra buffer requires overlap schedule" -> assertion error
            if strategy == "extra_buffer_lazy":
                return None
            extra = 1
    elif skip_decode_lock:
        extra = 1  # no_buffer under skip adds the base drop back
    return base + extra


def budget(mfs=0.90, ssm="float32", kvd="fp8_e5m2", D=4, replayssm=False,
           disable_radix=True, strategy="auto", overlap=False,
           skip_decode_lock=False, max_running=1, pin_K=None, ctx=65536,
           with_mtp=True):
    R = mamba_ratio(disable_radix, strategy, overlap, skip_decode_lock)
    if R is None:
        return {"error": "extra_buffer_lazy requires the overlap scheduler"}

    weights = W_MAIN + (W_MTP if with_mtp else 0)
    static = mfs * GPU_TOTAL - CUDA_CTX
    rest = static - weights
    if rest <= 0:
        return {"error": f"weights ({weights/GIB:.2f} GiB) exceed the static "
                         f"budget ({static/GIB:.2f} GiB)"}

    per_req = STATE_BYTES[ssm]
    Deff = 0 if replayssm else D          # replayssm allocates no intermediate

    if pin_K is not None:
        K = pin_K
    elif disable_radix and max_running is not None:
        K = max_running                   # from_max_running_requests branch
    else:
        mfr = 0.9                         # --mamba-full-memory-ratio default
        mb = rest * mfr / (1 + mfr)
        if Deff:
            K = int((mb - per_req * (1 + Deff)) // (per_req * (1 + Deff / R)))
        else:
            K = int((mb - per_req) // per_req)

    capped = min(max_running, K // R) if max_running else K // R
    main = (K + 1) * per_req
    inter = per_req * (capped + 1) * Deff if Deff else 0
    kv = rest - main - inter
    tokens = int(kv / KV_BYTES_TOK[kvd]) if kv > 0 else 0

    return {"R": R, "K": K, "max_num_reqs": capped,
            "weights_gib": weights / GIB, "rest_gib": rest / GIB,
            "main_mb": main / 1e6, "inter_mb": inter / 1e6,
            "kv_gib": kv / GIB, "kv_tokens": tokens,
            "boots": capped >= 1 and kv > 0,
            "fits_ctx": tokens >= ctx}


ROWS = [
    ("A  stock MTP, default flags (no tuning)",
     dict(mfs=0.85, disable_radix=False, strategy="auto", overlap=True,
          max_running=None, D=4)),
    ("B  stock MTP, --disable-radix-cache --max-running-requests 1, fp32",
     dict(mfs=0.90, disable_radix=True, max_running=1, D=4, ssm="float32")),
    ("C  = B at --mem-fraction-static 0.92",
     dict(mfs=0.92, disable_radix=True, max_running=1, D=4, ssm="float32")),
    ("D  = C + --mamba-ssm-dtype bfloat16",
     dict(mfs=0.92, disable_radix=True, max_running=1, D=4, ssm="bfloat16")),
    ("E  = C + --enable-linear-replayssm-spec",
     dict(mfs=0.92, disable_radix=True, max_running=1, D=4, ssm="float32",
          replayssm=True)),
    ("F  no-spec baseline (no MTP at all)",
     dict(mfs=0.92, disable_radix=True, max_running=1, D=0, ssm="float32",
          with_mtp=False)),
    ("G  hybrid MTP+ngram, L=9, fp32",
     dict(mfs=0.92, disable_radix=True, max_running=1, D=9, ssm="float32")),
    ("H  hybrid MTP+ngram, L=9, bfloat16",
     dict(mfs=0.92, disable_radix=True, max_running=1, D=9, ssm="bfloat16")),
    ("I  hybrid MTP+ngram, L=9, fp32 + replayssm",
     dict(mfs=0.92, disable_radix=True, max_running=1, D=9, ssm="float32",
          replayssm=True)),
    ("J  stock MTP, radix kept: no_buffer + SKIP_DECODE_LOCK, fp32",
     dict(mfs=0.92, disable_radix=False, strategy="no_buffer", overlap=True,
          skip_decode_lock=True, max_running=1, D=4)),
]

if __name__ == "__main__":
    print(f"GPU {GPU_TOTAL/GIB:.2f} GiB total, reserving "
          f"{CUDA_CTX/GIB:.2f} GiB for the CUDA context")
    print(f"weights: main {W_MAIN/GIB:.3f} + MTP {W_MTP/GIB:.3f} = "
          f"{(W_MAIN+W_MTP)/GIB:.3f} GiB "
          f"({(W_MAIN+W_MTP)/GPU_TOTAL*100:.1f}% of the card)\n")
    hdr = (f"{'config':<58}{'R':>3}{'K':>4}{'reqs':>5}"
           f"{'state':>8}{'inter':>8}{'KV GiB':>8}{'KV tok':>9}  verdict")
    print(hdr)
    print("-" * len(hdr))
    for name, kw in ROWS:
        r = budget(**kw)
        if "error" in r:
            print(f"{name:<58}{'':>3}{'':>4}{'':>5}{'':>8}{'':>8}{'':>8}"
                  f"{'':>9}  ERROR: {r['error']}")
            continue
        if not r["boots"]:
            v = "FAILS TO BOOT (max_num_reqs=0)"
        elif not r["fits_ctx"]:
            v = f"boots, but caps context at {r['kv_tokens']//1024}k"
        else:
            v = "boots, 64k context fits"
        print(f"{name:<58}{r['R']:>3}{r['K']:>4}{r['max_num_reqs']:>5}"
              f"{r['main_mb']:>7.0f}M{r['inter_mb']:>7.0f}M"
              f"{r['kv_gib']:>8.2f}{r['kv_tokens']:>9}  {v}")

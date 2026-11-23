"""Explicit tiny CUDA dependency check; no OCR model or document is loaded."""
import torch
from flash_attn import flash_attn_varlen_func
from paddlex.utils.deps import require_genai_engine_plugin


def main() -> None:
    assert torch.__version__.split("+")[0] == "2.8.0"
    assert torch.version.cuda == "12.8"
    assert torch._C._GLIBCXX_USE_CXX11_ABI is True
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    require_genai_engine_plugin("vllm-server")
    # Exercise the same variable-length FlashAttention function used by the
    # pinned PaddleX vision encoder, and compare with Torch's SDPA reference.
    torch.manual_seed(0)
    q, k, v = (torch.randn(16, 2, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    cumulative = torch.tensor([0, 16], device="cuda", dtype=torch.int32)
    actual = flash_attn_varlen_func(q, k, v, cumulative, cumulative, 16, 16)
    reference = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0),
    ).squeeze(0).transpose(0, 1)
    torch.cuda.synchronize()
    assert bool(torch.isfinite(actual).all())
    torch.testing.assert_close(actual, reference, atol=0.04, rtol=0.04)
    print("PaddleX vLLM plugin and FlashAttention BF16 CUDA smoke passed; no OCR model loaded.")


if __name__ == "__main__":
    main()

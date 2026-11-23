"""Start only the dedicated OCR model server after offline validation and GPU checks."""
import json
import os
from pathlib import Path
import sys

from preflight import INFERENCE, PreflightError, check_packages, read_manifest, require, validate_manifest, source_inventory


def main() -> None:
    require(os.environ.get("OCR_ENABLED") == "real", "Real OCR is disabled")
    Path.home().mkdir(mode=0o700, parents=True, exist_ok=True)
    check_packages("vllm")
    manifest = validate_manifest(read_manifest(Path(os.environ.get("OCR_RUNTIME_MANIFEST", "/etc/ocr/manifest.json"))))
    require(json.loads(Path("/opt/ocr-build/adapter-tree.json").read_text()) == source_inventory(), "Adapter files differ from the built image inventory")
    import torch
    require(torch.version.cuda == "12.8", "Unexpected Torch CUDA runtime")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Exactly one dedicated visible GPU is required")
    require((8, 0) <= torch.cuda.get_device_capability(0) < (10, 0), "GPU architecture is outside the reviewed range")
    free, _total = torch.cuda.mem_get_info(0)
    require(free >= INFERENCE["minFreeGpuMiB"] * 1024 * 1024, "Insufficient free GPU memory for the declared trial budget")
    config = {
        "trust-remote-code": True,  # Only the revision-pinned, inventoried, local model code.
        "served-model-name": INFERENCE["servedModel"],
        "gpu-memory-utilization": INFERENCE["gpuMemoryUtilization"],
        "max-model-len": INFERENCE["maxModelLen"],
        "max-num-batched-tokens": INFERENCE["maxNumBatchedTokens"],
        "max-num-seqs": 1, "api-server-count": 1, "tensor-parallel-size": 1,
        "dtype": INFERENCE["dtype"], "enforce-eager": True,
        "disable-log-requests": True,
    }
    path = Path("/tmp/ocr-vllm-backend.json")
    path.write_text(json.dumps(config))
    os.execv(sys.executable, [sys.executable, "-m", "paddlex.inference.genai.server",
        "--model_name", INFERENCE["servedModel"], "--model_dir", manifest["models"]["ocr"]["path"],
        "--backend", "vllm", "--host", "0.0.0.0", "--port", "8000", "--backend_config", str(path)])


if __name__ == "__main__":
    try:
        main()
    except (PreflightError, OSError, ValueError, TypeError) as error:
        print(f"OCR model server rejected startup: {error}", file=sys.stderr)
        raise SystemExit(2)

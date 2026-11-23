"""Fetch the DFlash2 drafter for single-user mode (SPEC=dflash2 in single-user/start_qwen.sh):
the W4A16-GPTQ requantization of incoai/Qwen3.8-27B-DFlash2 built by drafter/capture_dflash2.py
+ drafter/quant_dflash2.py (1.2 GB instead of 3.85 GB bf16; see drafter/README.md), prebuilt on
the Hub.

  venv/bin/python prepare/fetch_dflash2.py [dst_dir]         # default models/Qwen3.8-27B-DFlash2-W4A16
  venv/bin/python prepare/fetch_dflash2.py --bf16 [dst_dir]  # the original bf16 drafter instead (incoai)
"""
import os, sys
from huggingface_hub import snapshot_download

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
BF16 = "--bf16" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
REPO = "incoai/Qwen3.8-27B-DFlash2" if BF16 else "syvai/Qwen3.8-27B-DFlash2-W4A16"
D = args[0] if args else os.path.join(ROOT, "models", "Qwen3.8-27B-DFlash2" + ("" if BF16 else "-W4A16"))
os.makedirs(D, exist_ok=True)
snapshot_download(REPO, local_dir=D, allow_patterns=["*.json", "*.safetensors", "README.md"])
print("drafter ready:", D)
print("serve with: SPEC=dflash2 bash single-user/start_qwen.sh")

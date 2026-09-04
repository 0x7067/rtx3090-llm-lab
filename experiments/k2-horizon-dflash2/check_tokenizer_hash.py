"""Check that llama.cpp's converter will recognise the K2 7B tokenizer.

The EAGLE-3 conversion path reads the tokenizer from --target-model-dir and
runs the generic BPE vocab code, which identifies the pre-tokenizer by a
sha256 over a fixed check string. An unregistered hash aborts the conversion.
"""
import hashlib
import sys

from transformers import AutoTokenizer

CHK_TXT = open(
    "/tmp/claude-1000/-data-docker-services/e39adc1c-0fa6-4d11-85c2-93a34c34d655/scratchpad/chktxt.txt",
    encoding="utf-8",
).read()
REGISTERED = {
    "1f9825a388f700a6b591722f17d470cbbcf10973ece35d2fd14239a14110ae1a": "k2-horizon (0.9B)",
    "a9af07a84191f55098b248ae6f3dfe9e32d3190bebe8eafd91c1ddec9bc3449f": "k2-horizon (36B)",
}

path = sys.argv[1] if len(sys.argv) > 1 else "/data/buttercup_6tb/specforge-work/models/IFM/K2-Horizon-7B"
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
h = hashlib.sha256(str(tok.encode(CHK_TXT)).encode()).hexdigest()
print("tokenizer hash:", h)
print("recognized as:", REGISTERED.get(h, "*** NOT REGISTERED — conversion would abort ***"))

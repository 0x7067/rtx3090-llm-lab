"""Rewarm opt-in private token prefixes after a SparseKV backend starts."""
import hashlib
import json
import os
import time
from pathlib import Path
from urllib.request import Request, urlopen


def validate(bundle, model, tokenizer_file=None):
    if bundle["model"] != str(model):
        raise ValueError("Saved prefixes belong to a different model")
    tokenizer_file = tokenizer_file or model / "tokenizer.json"
    digest = hashlib.sha256(tokenizer_file.read_bytes()).hexdigest()
    if digest != bundle["tokenizer_sha256"]:
        raise ValueError("Tokenizer changed; recapture the shared prefixes")
    prefixes = bundle["prefixes"]
    if not 1 <= len(prefixes) <= 2:
        raise ValueError("Expected one or two prefixes")
    for tokens in prefixes:
        if not tokens or len(tokens) > 32768 or len(tokens) % 864:
            raise ValueError("Prefixes must end on an 864-token block boundary")
        if any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError("Invalid token ID")
    return prefixes


def main():
    path = Path(os.environ.get("SPARSEKV_WARM_PREFIX_FILE", "/cache/sparsekv-warm-prefixes.json"))
    if not path.exists():
        return
    try:
        if path.stat().st_size > 2 * 1024**2:
            raise ValueError("Saved prefix file exceeds size limit")
        tokenizer_file = os.environ.get("SPARSEKV_WARM_TOKENIZER_FILE")
        prefixes = validate(json.loads(path.read_text()), Path(os.environ["MODEL"]),
                            Path(tokenizer_file) if tokenizer_file else None)
        base = "http://127.0.0.1:" + os.environ.get("PORT", "5890")
        deadline = time.monotonic() + 900
        while True:
            try:
                with urlopen(base + "/health", timeout=2):
                    break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Backend startup exceeded prefix warm deadline")
                time.sleep(3)
        receipts = []
        for tokens in prefixes:
            body = {"model": os.environ.get("SPARSEKV_WARM_MODEL_ID", "qwen3.8-27b"), "prompt": tokens, "max_tokens": 1,
                    "temperature": 0, "vllm_xargs": {"sparse_pin_prefix": 1}}
            start = time.monotonic()
            with urlopen(Request(base + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}), timeout=120) as response:
                result = json.load(response)
            receipts.append({"tokens": len(tokens), "seconds": time.monotonic() - start,
                             "usage": result["usage"]})
        print("sparsekv prefix warm: " + json.dumps(receipts), flush=True)
    except Exception as error:
        # Warming is optional; errors leave the backend and its observer running.
        print("sparsekv prefix warm failed: " + str(error), flush=True)

if __name__ == "__main__":
    main()

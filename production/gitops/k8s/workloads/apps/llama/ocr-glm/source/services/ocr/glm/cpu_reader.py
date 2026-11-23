"""Bounded, explicitly attributed PP-OCR reader for future GLM recovery.

This component is not enabled by the deployed GLM worker. It reads the exact RGB
raster supplied by its caller; it does not choose documents, repair text, infer
fields or claim that recognition scores are calibrated confidence.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import tempfile
import time

from ..paddle.errors import WorkerError

READER_ID = "ppocr-v5-latin-v1"
CPU_SECONDS = 45
MAX_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_PNG_BYTES = 32 * 1024 * 1024
MAX_LINES = 10_000
MAX_TEXT = 1_000_000
CPU_PYTHON = "/opt/ocr/venv-layout/bin/python"
VERSIONS = {"paddlepaddle": "3.2.1", "paddleocr": "3.7.0", "paddlex": "3.7.2",
            "numpy": "2.2.6", "Pillow": "11.3.0", "pypdfium2": "4.30.0"}
PARAMETERS = {"text_detection_model_name": "PP-OCRv5_server_det",
    "text_recognition_model_name": "latin_PP-OCRv5_mobile_rec",
    "text_recognition_batch_size": 1, "use_doc_orientation_classify": False,
    "use_doc_unwarping": False, "use_textline_orientation": False,
    "text_det_limit_side_len": 1600, "text_det_limit_type": "max",
    "text_rec_score_thresh": 0.0, "return_word_box": False,
    "device": "cpu", "cpu_threads": 1, "enable_mkldnn": True}
MODELS = {
    "PP-OCRv5_server_det": {
        "revision": "ca867c897ecbca8873081573a802ad70d499cb94",
        "files": {
            "config.json": "436bf19010fe3879477e8599a870a0ec8ccd14ca39056805ec4ed20ab318ec04",
            "inference.json": "af5876933d8806a1b50d895867e0781e135cd92ff37381992828fc8d1b842d28",
            "inference.pdiparams": "183146fe9d9910352f68482f623bcbbb9fa7b9e8fa1463b9ad288cef00524d2d",
            "inference.yml": "28fb721efc3634fc8aa677e474b9602cb815a91cf569ef357a7a553d7b3ce685"}},
    "latin_PP-OCRv5_mobile_rec": {
        "revision": "ab2cd5cc5fa6309be2e5acdfe66eca2c2c127d57",
        "files": {
            "config.json": "e0f5ccc44fe86254956f27746058bc4cf88d86145668cecf22ba5dd962483c80",
            "inference.json": "820b2ee8e5072de5156c3d55124c2b03c82ef278d59f453f857c0789bacb4da1",
            "inference.pdiparams": "53cdc8b481a7394bb108f96d0fb3432b0a8f392e22c7d18f06dbb2d42b8b25f9",
            "inference.yml": "0bbe984570f597af3638e50bdf2e8276f3ab26a61966096538b3b0d1849f5c84"}},
}


def reader_identity():
    recipe = {"id": READER_ID, "models": MODELS, "versions": VERSIONS,
              "pythonVersion": "3.12.15", "parameters": PARAMETERS,
              "geometry": "input-rgb-pixels", "scores": "uncalibrated"}
    canonical = json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
    return {**deepcopy(recipe), "fingerprint": hashlib.sha256(canonical).hexdigest()}


def public_reader_identity():
    return {"id":READER_ID,"fingerprint":reader_identity()["fingerprint"],
        "detectionModel":{"id":"PaddlePaddle/PP-OCRv5_server_det","revision":MODELS["PP-OCRv5_server_det"]["revision"]},
        "recognitionModel":{"id":"PaddlePaddle/latin_PP-OCRv5_mobile_rec","revision":MODELS["latin_PP-OCRv5_mobile_rec"]["revision"]},
        "inferenceEngine":{"name":"paddlepaddle","version":VERSIONS["paddlepaddle"]}}


def strict_json(data):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result: raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def validate_observation(value, expected_input):
    """Reject partial, foreign-raster, malformed and excessive child output."""
    def require(condition):
        if not condition: raise WorkerError("invalid_provider_output")
    require(isinstance(expected_input,dict) and set(expected_input)=={"width","height","rgbSha256","pngSha256"})
    for key in ["rgbSha256","pngSha256"]:
        digest=expected_input[key]
        require(isinstance(digest,str) and len(digest)==64 and all(c in "0123456789abcdef" for c in digest))
    require(isinstance(value, dict) and set(value) == {
        "protocolVersion", "reader", "input", "text", "lines"})
    require(type(value["protocolVersion"]) is int and value["protocolVersion"] == 1)
    try:
        canonical = lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False)
        require(canonical(value["reader"]) == canonical(reader_identity())
                and canonical(value["input"]) == canonical(expected_input))
    except (ValueError, TypeError): raise WorkerError("invalid_provider_output") from None
    width, height = expected_input["width"], expected_input["height"]
    require(type(width) is int and type(height) is int and 0 < width <= 16_000
            and 0 < height <= 16_000 and width * height <= 40_000_000)
    lines = value["lines"]
    require(isinstance(lines, list) and len(lines) <= MAX_LINES)
    length = max(0, len(lines) - 1)
    for line in lines:
        require(isinstance(line, dict) and set(line) == {"text", "score", "polygon"})
        text, score, points = line["text"], line["score"], line["polygon"]
        require(isinstance(text, str) and len(text) <= 10_000)
        try: text.encode("utf-8", "strict")
        except UnicodeError: raise WorkerError("invalid_provider_output") from None
        length += len(text)
        require(length <= MAX_TEXT)
        require(type(score) in {int, float} and math.isfinite(score) and 0 <= score <= 1)
        require(isinstance(points, list) and len(points) == 4)
        for point in points:
            require(isinstance(point, list) and len(point) == 2)
            require(all(type(n) in {int, float} and math.isfinite(n) for n in point))
            require(0 <= point[0] <= width and 0 <= point[1] <= height)
        # Require a nondegenerate, convex perimeter in either orientation.
        crosses = []
        for i in range(4):
            a, b, c = points[i], points[(i + 1) % 4], points[(i + 2) % 4]
            crosses.append((b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0]))
        require(all(c > 0 for c in crosses) or all(c < 0 for c in crosses))
    require(isinstance(value["text"], str) and value["text"] == "\n".join(line["text"] for line in lines))
    return value


def _collect(command, payload, directory, deadline):
    """Child shares the OCR process group, so existing cancellation kills both.

    Never retain or expose dependency logs. Bound stdout while the child runs;
    communicate() alone would allow an arbitrary allocation before validation.
    """
    cache = str(directory / "cache")
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1", "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "MALLOC_ARENA_MAX": "2", "NVIDIA_VISIBLE_DEVICES": "void", "CUDA_VISIBLE_DEVICES": "",
        "XDG_CACHE_HOME": cache, "HF_HOME": cache + "/huggingface",
        "PADDLE_PDX_CACHE_HOME": cache + "/paddlex"}
    if time.monotonic() >= deadline: raise WorkerError("timeout", True)
    child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, cwd=directory, env=env, start_new_session=False)
    output = bytearray()
    try:
        child.stdin.write(payload); child.stdin.close()
        with selectors.DefaultSelector() as selector:
            selector.register(child.stdout, selectors.EVENT_READ)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0: raise WorkerError("timeout", True)
                for key, _ in selector.select(min(remaining, .1)):
                    chunk = os.read(key.fd, 65536)
                    if not chunk: selector.unregister(key.fileobj); continue
                    if len(output) + len(chunk) > MAX_OUTPUT_BYTES:
                        raise WorkerError("resource_limit", resource_limit="output_protocol")
                    output.extend(chunk)
        remaining = deadline - time.monotonic()
        if remaining <= 0: raise WorkerError("timeout", True)
        try: code = child.wait(timeout=remaining)
        except subprocess.TimeoutExpired: raise WorkerError("timeout", True) from None
        if code != 0: raise WorkerError("provider_failure")
        try: return strict_json(output.decode("utf-8", "strict"))
        except (ValueError, UnicodeError): raise WorkerError("invalid_provider_output") from None
    finally:
        if child.poll() is None: child.kill()
        child.wait()
        child.stdout.close()
        if not child.stdin.closed: child.stdin.close()


def read_cpu(image, model_dir, work_dir, budget, deadline):
    """Run at most one serial, 45-second read of an exact caller-owned RGB view."""
    model_dir, work_dir = Path(model_dir), Path(work_dir)
    if not model_dir.is_absolute() or not work_dir.is_absolute():
        raise WorkerError("provider_unavailable")
    if image.mode != "RGB": raise WorkerError("invalid_document")
    width, height = image.size
    if not (0 < width <= 16_000 and 0 < height <= 16_000 and width * height <= 40_000_000):
        raise WorkerError("resource_limit", resource_limit="page_dimensions")
    # Include RGB byte/hash, PNG encoding, child decode and BGR inference copies.
    area = width * height
    with budget.memory(5 * area):
        budget.reserve(5 * area)
        with tempfile.TemporaryDirectory(prefix="cpu-read-", dir=work_dir) as name:
            directory = Path(name)
            os.chmod(directory, 0o700)
            path = directory / "raster.png"
            with path.open("xb") as stream:
                os.chmod(path, 0o600); image.save(stream, format="PNG")
            if path.stat().st_size > MAX_PNG_BYTES:
                raise WorkerError("resource_limit", resource_limit="input_bytes")
            metadata = {"width": width, "height": height,
                "rgbSha256": hashlib.sha256(image.tobytes()).hexdigest(),
                "pngSha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            command = [CPU_PYTHON, str(Path(__file__).with_name("cpu_worker.py")),
                       "--input", str(path), "--models", str(model_dir)]
            output = _collect(command, json.dumps(metadata).encode(), directory,
                              min(deadline, time.monotonic() + CPU_SECONDS))
            return validate_observation(output, metadata)


def verify_cpu_models(model_dir,deadline):
    """Bounded auxiliary-only startup check; never scans or primes GLM artifacts."""
    if not Path(model_dir).is_absolute():raise WorkerError("provider_unavailable")
    with tempfile.TemporaryDirectory(prefix="ocr-cpu-verify-",dir="/dev/shm") as name:
        command=[CPU_PYTHON,str(Path(__file__).with_name("cpu_worker.py")),"--verify-models","--models",str(model_dir)]
        value=_collect(command,b"",Path(name),min(deadline,time.monotonic()+CPU_SECONDS))
        expected={"protocolVersion":1,"reader":reader_identity(),"modelsVerified":True}
        if json.dumps(value,sort_keys=True,allow_nan=False)!=json.dumps(expected,sort_keys=True):
            raise WorkerError("provider_unavailable")

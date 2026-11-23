"""Offline CPU child. Runs only under the existing pinned layout interpreter."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import signal
import socket
import stat
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "services.ocr.glm"

from .cpu_reader import (CPU_SECONDS, MAX_OUTPUT_BYTES, MAX_PNG_BYTES, MODELS,
    PARAMETERS, VERSIONS, reader_identity, strict_json, validate_observation)


def restrict():
    if platform.system() != "Linux" or platform.python_version() != "3.12.15":
        raise ValueError("Requires pinned Linux Python")
    os.umask(0o077)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024**2, 32 * 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    available = os.sched_getaffinity(0)
    if not available: raise ValueError("No CPU affinity available")
    os.sched_setaffinity(0, {min(available)})
    signal.alarm(CPU_SECONDS)
    def denied(*_args, **_kwargs): raise PermissionError("CPU OCR networking denied")
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.create_connection = denied


def regular_bytes(path, maximum):
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > maximum:
            raise ValueError("Invalid regular input")
        body = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
        if (info.st_size, info.st_mtime_ns, info.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError("Input changed")
        if len(body) != info.st_size: raise ValueError("Input size changed")
        return body


def verify_models(models):
    if not models.is_absolute() or models.is_symlink() or not models.is_dir():
        raise ValueError("Requires a trusted model directory")
    if {name: importlib.metadata.version(name) for name in VERSIONS} != VERSIONS:
        raise ValueError("CPU dependencies differ")
    for name, model in MODELS.items():
        base = models / name
        if base.is_symlink() or not base.is_dir(): raise ValueError("Invalid model directory")
        if {p.name for p in base.iterdir()} != set(model["files"]):
            raise ValueError("Model inventory differs")
        for filename, digest in model["files"].items():
            if hashlib.sha256(regular_bytes(base / filename, 96 * 1024**2)).hexdigest() != digest:
                raise ValueError("Model bytes differ")


def process(path, models, expected):
    if not path.is_absolute():raise ValueError("Requires a trusted absolute input path")
    if not isinstance(expected, dict) or set(expected) != {"width", "height", "pngSha256", "rgbSha256"}:
        raise ValueError("Invalid input binding")
    if any(type(expected[key]) is not int for key in ["width", "height"]):
        raise ValueError("Invalid dimensions")
    if not (0 < expected["width"] <= 16_000 and 0 < expected["height"] <= 16_000
            and expected["width"] * expected["height"] <= 40_000_000):
        raise ValueError("Excessive raster")
    verify_models(models)
    raw = regular_bytes(path, MAX_PNG_BYTES)
    if hashlib.sha256(raw).hexdigest() != expected["pngSha256"]:
        raise ValueError("Foreign raster")
    import io
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = 40_000_000
    with Image.open(io.BytesIO(raw)) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != (expected["width"], expected["height"]):
            raise ValueError("Invalid RGB raster")
        image.load()
        if hashlib.sha256(image.tobytes()).hexdigest() != expected["rgbSha256"]:
            raise ValueError("Decoded pixels differ")
        import numpy as np
        import cv2
        import paddle
        from paddleocr import PaddleOCR
        cv2.setNumThreads(1); paddle.set_device("cpu")
        pixels = np.asarray(image)[:, :, ::-1].copy()
    reader = PaddleOCR(**PARAMETERS,
        text_detection_model_dir=str(models / "PP-OCRv5_server_det"),
        text_recognition_model_dir=str(models / "latin_PP-OCRv5_mobile_rec"))
    try:
        iterator = iter(reader.predict(pixels))
        try:
            result = next(iterator)
            if next(iterator, None) is not None: raise ValueError("Unexpected extra result")
            texts, scores, polygons = result["rec_texts"], result["rec_scores"], result["rec_polys"]
            if not len(texts) == len(scores) == len(polygons) <= 10_000:
                raise ValueError("Inconsistent line output")
            lines = [{"text": text, "score": float(score), "polygon": np.asarray(polygon).tolist()}
                     for text, score, polygon in zip(texts, scores, polygons)]
            value = {"protocolVersion": 1, "reader": reader_identity(), "input": expected,
                     "text": "\n".join(line["text"] for line in lines), "lines": lines}
            return validate_observation(value, expected)
        finally:
            close = getattr(iterator, "close", None)
            if close: close()
    finally:
        pixels.fill(0)
        close = getattr(reader, "close", None)
        if close: close()


def main():
    protocol = os.dup(1)
    with open(os.devnull, "wb") as sink:
        os.dup2(sink.fileno(), 1); os.dup2(sink.fileno(), 2)
    code = 0
    try:
        restrict()
        parser = argparse.ArgumentParser(add_help=False)
        mode=parser.add_mutually_exclusive_group(required=True)
        mode.add_argument("--input",type=Path)
        mode.add_argument("--verify-models",action="store_true")
        parser.add_argument("--models", required=True, type=Path)
        args = parser.parse_args()
        payload = sys.stdin.buffer.read(4097)
        if len(payload) > 4096: raise ValueError("Excessive metadata")
        if args.verify_models:
            if payload:raise ValueError("Verification does not accept a document")
            verify_models(args.models)
            value={"protocolVersion":1,"reader":reader_identity(),"modelsVerified":True}
        else:value = process(args.input, args.models, strict_json(payload))
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8", "strict")
        if len(encoded) > MAX_OUTPUT_BYTES: raise ValueError("Excessive output")
    except BaseException:
        code = 2; encoded = b'{"protocolVersion":1,"error":"cpu_reader_failed"}'
    try:
        remaining = memoryview(encoded + b"\n")
        while remaining: remaining = remaining[os.write(protocol, remaining):]
    finally:
        signal.alarm(0); os.close(protocol)
    return code


if __name__ == "__main__": raise SystemExit(main())

"""Offline launch validation. Never downloads models, imports inference, or opens a socket."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
from pathlib import Path, PurePosixPath
from source_inventory import inventory as source_inventory

HERE = Path(__file__).resolve().parent
PYTHON_VERSION = "3.12.15"
MODELS = {
    "ocr": ("PaddlePaddle/PaddleOCR-VL-1.6", "c5630abae1d940eafe0697512a0325494b02ab42", "/models/ocr"),
    "layout": ("PaddlePaddle/PP-DocLayoutV3", "241f8bdfc77a7c7bee915a5057aaee58c235a8d3", "/models/layout"),
}
WORKER_CONFIG = {
    "configVersion": 1, "pipelineVersion": "v1.6",
    "layoutModelDir": "/models/layout", "vlModelDir": "/models/ocr",
    "vllmBaseUrl": "http://ocr-vllm:8000/v1",
    "servedModelName": "PaddleOCR-VL-1.6-0.9B", "pdfScale": 2, "layoutDevice": "cpu",
}
INFERENCE = {
    "endpoint": "http://ocr-vllm:8000/v1", "servedModel": "PaddleOCR-VL-1.6-0.9B",
    "minFreeGpuMiB": 10240, "gpuMemoryUtilization": 0.35,
    "maxModelLen": 8192, "maxNumBatchedTokens": 8192, "maxNumSeqs": 1,
    "apiServerCount": 1, "dtype": "bfloat16", "enforceEager": True,
    "trustInventoriedModelCode": True,
}


class PreflightError(ValueError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise PreflightError(message)


def keys(value: object, expected: set[str], name: str) -> None:
    require(isinstance(value, dict) and set(value) == expected, f"Invalid {name} fields")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_lock(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text().splitlines():
        if not line or line[0].isspace() or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+) \\", line)
        require(match is not None, "Lock contains an unsupported or unpinned requirement")
        name, version = match.groups()
        require(name not in result, "Lock repeats a package")
        result[name] = version
    require(bool(result), "Dependency lock is empty")
    return result


def check_packages(role: str) -> None:
    require(platform.system() == "Linux" and platform.machine() == "x86_64", "Requires Linux x86_64")
    require(platform.python_version() == PYTHON_VERSION, "Python patch does not match the pinned image")
    packages = parse_lock(HERE / f"requirements.{role}.lock")
    if role == "vllm":
        packages.update(parse_lock(HERE / "requirements.vllm-gpu.lock"))
    for name, version in packages.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            raise PreflightError(f"Missing locked package: {name}") from None
        require(actual == version, f"Installed package differs from lock: {name}")


def validate_manifest(manifest: object, verify_files: bool = True) -> dict:
    keys(manifest, {"manifestVersion", "platform", "pythonVersion", "normalizationRevision", "models", "artifacts", "inference", "workerConfig"}, "manifest")
    require(type(manifest["manifestVersion"]) is int and manifest["manifestVersion"] == 1, "Unsupported manifest version")
    require(manifest["platform"] == "linux/amd64", "Unexpected platform")
    require(manifest["pythonVersion"] == PYTHON_VERSION, "Unexpected Python version")
    require(manifest["normalizationRevision"] == "receipt-core-ptbr-v13", "Unexpected normalization rules")
    require(manifest["inference"] == INFERENCE, "Inference settings differ from the reviewed profile")
    require(manifest["workerConfig"] == WORKER_CONFIG, "Worker settings differ from the reviewed profile")
    artifacts = manifest["artifacts"]
    keys(artifacts, {"layoutImage", "vllmImage", "adapterTreeSha256", "layoutLockSha256", "vllmLockSha256"}, "artifacts")
    for field in ("layoutImage", "vllmImage"):
        value = artifacts[field]
        require(isinstance(value, str) and re.fullmatch(r"(?:[a-z0-9./:_-]+@)?sha256:[a-f0-9]{64}", value) is not None,
                f"Missing immutable built image: {field}")
    require(isinstance(artifacts["adapterTreeSha256"], str) and re.fullmatch(r"[a-f0-9]{64}", artifacts["adapterTreeSha256"]) is not None,
            "Missing adapter source tree digest")
    require(artifacts["adapterTreeSha256"] == source_inventory()["sha256"], "Adapter source tree hash mismatch")
    for role in ("layout", "vllm"):
        require(artifacts[f"{role}LockSha256"] == sha256_file(HERE / f"requirements.{role}.lock"), "Dependency lock hash mismatch")
    keys(manifest["models"], set(MODELS), "models")
    for name, (model_id, revision, directory) in MODELS.items():
        model = manifest["models"][name]
        keys(model, {"id", "revision", "path", "files"}, f"{name} model")
        require((model["id"], model["revision"], model["path"]) == (model_id, revision, directory), "Model identity or path mismatch")
        files = model["files"]
        require(isinstance(files, dict) and 1 <= len(files) <= 1000, "Model inventory is missing or too large")
        for relative, digest in files.items():
            path = PurePosixPath(relative)
            require(isinstance(relative, str) and relative == path.as_posix() and not path.is_absolute()
                    and ".." not in path.parts and relative not in ("", ".") and "\\" not in relative,
                    "Invalid model inventory path")
            require(isinstance(digest, str) and re.fullmatch(r"[a-f0-9]{64}", digest) is not None, "Invalid model inventory digest")
        if verify_files:
            root = Path(directory)
            require(root.is_dir() and not root.is_symlink(), "Missing regular model directory")
            actual_files = set()
            for path in root.rglob("*"):
                require(not path.is_symlink(), "Model inventory cannot contain symbolic links")
                if path.is_dir():
                    continue
                require(path.is_file(), "Model inventory contains a non-regular file")
                relative = path.relative_to(root).as_posix()
                actual_files.add(relative)
                require(relative in files, "Model directory contains a file outside the inventory")
                require(sha256_file(path) == files[relative], "Model bytes differ from the inventory")
            require(actual_files == set(files), "Model inventory contains missing files")
    return manifest


def runtime_config(manifest: dict) -> dict:
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    fingerprint = hashlib.sha256(canonical).hexdigest()
    return {"pipeline": {
        "id": f"paddleocr-vl-1.6-{fingerprint[:16]}", "fingerprint": fingerprint,
        "executionMode": "real", "ocrModel": {"id": MODELS["ocr"][0], "revision": MODELS["ocr"][1]},
        "layoutModel": {"id": MODELS["layout"][0], "revision": MODELS["layout"][1]},
        "interpretationModel": None, "inferenceEngine": {"name": "vllm", "version": "0.10.2"},
    }, "workerConfig": dict(WORKER_CONFIG)}


def read_manifest(path: Path) -> dict:
    require(path.is_file() and path.stat().st_size <= 1024 * 1024, "Missing or oversized manifest")
    def unique_object(pairs):
        obj = {}
        for key, value in pairs:
            require(key not in obj, "Duplicate JSON manifest key")
            obj[key] = value
        return obj
    return json.loads(path.read_text(), object_pairs_hook=unique_object)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--emit-runtime", action="store_true")
    parser.add_argument("--check-packages", choices=("layout", "vllm"))
    parser.add_argument("--role", choices=("layout", "vllm"), default="layout")
    args = parser.parse_args()
    try:
        if args.check_packages:
            check_packages(args.check_packages)
        else:
            require(args.manifest is not None, "A manifest path is required")
            check_packages(args.role)
            manifest = validate_manifest(read_manifest(args.manifest))
            build_record = json.loads(Path("/opt/ocr-build/adapter-tree.json").read_text())
            require(build_record == source_inventory(), "Adapter files differ from the built image inventory")
            if args.emit_runtime:
                print(json.dumps(runtime_config(manifest), separators=(",", ":")))
    except (PreflightError, OSError, ValueError, TypeError) as error:
        print(f"OCR preflight rejected startup: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


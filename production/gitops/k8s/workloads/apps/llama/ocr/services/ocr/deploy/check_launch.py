"""Check local built-image identity and staging. Start only with explicit --start."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

from preflight import HERE, PreflightError, read_manifest, require, sha256_file, validate_manifest


def check_credentials(namespace: str, token: str) -> None:
    require(re.fullmatch(r"[A-Za-z0-9_-]{1,100}", namespace) is not None, "Invalid namespace")
    require(re.fullmatch(r"[!-~]{24,4096}", token) is not None, "Invalid service token")


def check_image(record: dict, reference: str, source_hash: str) -> None:
    require(record.get("Os") == "linux" and record.get("Architecture") == "amd64", "Image platform mismatch")
    if reference.startswith("sha256:"):
        require(record.get("Id") == reference, "Built image ID mismatch")
    else:
        require(reference in record.get("RepoDigests", []), "Built repository digest mismatch")
    require(record.get("Config", {}).get("Labels", {}).get("org.finance-ocr.adapter-tree-sha256") == source_hash,
            "Built image does not identify the reviewed adapter snapshot")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--start", action="store_true", help="Explicitly enable and start the reviewed synthetic OCR profile")
    args = parser.parse_args()
    manifest = validate_manifest(read_manifest(args.manifest), verify_files=False)
    require(args.token_file.is_file() and not args.token_file.is_symlink(), "Missing regular token file")
    require(args.token_file.stat().st_size <= 4097, "Service token file is oversized")
    require(args.token_file.stat().st_mode & 0o007 == 0, "Token must not be accessible by other users")
    token = args.token_file.read_text().strip()
    check_credentials(args.namespace, token)
    for name, model in manifest["models"].items():
        root = args.models_dir / name
        require(root.is_dir() and not root.is_symlink(), "Missing regular staged model directory")
        actual = {}
        for path in root.rglob("*"):
            require(not path.is_symlink(), "Staged models cannot contain symlinks")
            if path.is_dir():
                continue
            require(path.is_file(), "Staged model entry must be regular")
            actual[path.relative_to(root).as_posix()] = sha256_file(path)
        require(actual == model["files"], "Staged model inventory mismatch")
    for role in ("layout", "vllm"):
        reference = manifest["artifacts"][f"{role}Image"]
        output = subprocess.run(["docker", "image", "inspect", reference], check=True, capture_output=True, text=True, timeout=30)
        records = json.loads(output.stdout)
        require(isinstance(records, list) and len(records) == 1, "Unexpected image inspection response")
        check_image(records[0], reference, manifest["artifacts"]["adapterTreeSha256"])
    env = os.environ.copy()
    env.update({
        "OCR_ENABLED": "real", "OCR_MANIFEST_PATH": str(args.manifest.resolve()),
        "OCR_MODELS_DIR": str(args.models_dir.resolve()), "OCR_TOKEN_PATH": str(args.token_file.resolve()),
        "OCR_NAMESPACE": args.namespace, "OCR_LAYOUT_IMAGE": manifest["artifacts"]["layoutImage"],
        "OCR_VLLM_IMAGE": manifest["artifacts"]["vllmImage"],
        "OCR_ADAPTER_TREE_SHA256": manifest["artifacts"]["adapterTreeSha256"],
    })
    command = ["docker", "compose", "-f", str(HERE / "compose.yaml"), "--profile", "ocr-real"]
    subprocess.run(command + ["config", "--quiet"], env=env, check=True, timeout=30)
    print("OCR staging, image identity, source snapshot and Compose configuration verified; no inference quality claim.")
    if args.start:
        subprocess.run(command + ["up", "--no-build", "--pull", "never"], env=env, check=True)


if __name__ == "__main__":
    try:
        main()
    except (PreflightError, OSError, ValueError, TypeError, subprocess.SubprocessError) as error:
        raise SystemExit(f"OCR launch rejected: {error}")

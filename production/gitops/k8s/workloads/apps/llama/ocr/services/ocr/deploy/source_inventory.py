"""Fingerprint or package exactly the source files copied into both OCR images."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[3]


def source_paths(root: Path = ROOT) -> list[Path]:
    paths = [root / "package.json", root / "package-lock.json"]
    for directory in (root / "shared", root / "services/ocr"):
        for path in directory.rglob("*"):
            relative = path.relative_to(root)
            if (relative.parts[:4] == ("services", "ocr", "deploy", "local")
                    or "__pycache__" in relative.parts or path.name.endswith((".pyc", ".test.ts"))
                    or (path.name.startswith("test_") and path.suffix == ".py")):
                continue
            if path.is_symlink():
                raise ValueError("Source snapshot cannot contain symbolic links")
            if path.is_file():
                paths.append(path)
    if not all(path.is_file() and not path.is_symlink() for path in paths):
        raise ValueError("Missing regular source file")
    return sorted(paths)


def inventory(root: Path = ROOT) -> dict:
    files = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in source_paths(root)}
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"algorithm": "sha256-path-content-v1", "sha256": hashlib.sha256(canonical).hexdigest(), "files": files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect")
    parser.add_argument("--write", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    result = inventory()
    if args.expect is not None and args.expect != result["sha256"]:
        raise SystemExit("OCR adapter source differs from the requested build snapshot")
    if args.write:
        args.write.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    if args.archive:
        # No .git, .data, node_modules, root environment files, secrets or model mounts.
        with tarfile.open(args.archive, "w:gz") as archive:
            for path in source_paths():
                archive.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
    print(result["sha256"])


if __name__ == "__main__":
    main()

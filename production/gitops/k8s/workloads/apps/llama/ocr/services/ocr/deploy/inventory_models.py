"""Inventory already-staged model files; does not download or execute them."""
import argparse
import json
from pathlib import Path

from preflight import MODELS, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, required=True)
    args = parser.parse_args()
    models = {}
    for name, (model_id, revision, runtime_path) in MODELS.items():
        root = args.models_dir / name
        if not root.is_dir() or root.is_symlink():
            raise SystemExit(f"Missing regular {name} model directory")
        files = {}
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise SystemExit("Copy snapshot bytes into staging; symbolic links are not accepted")
            if path.is_dir():
                continue
            if not path.is_file():
                raise SystemExit("Only regular model files are accepted")
            files[path.relative_to(root).as_posix()] = sha256_file(path)
        if not files or len(files) > 1000:
            raise SystemExit("Missing or oversized model inventory")
        models[name] = {"id": model_id, "revision": revision, "path": runtime_path, "files": files}
    print(json.dumps(models, indent=2))


if __name__ == "__main__":
    main()

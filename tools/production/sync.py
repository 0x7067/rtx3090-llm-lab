#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pyyaml==6.0.2"]
# ///
"""Mirror and lock the inputs of the RTX 3090 llama service.

Subcommands:

  sync    Copy the GitOps files listed in SOURCES from one GitOps commit into
          production/gitops/, then regenerate production/LOCK.json and
          production/STATUS.md. With --host-scan, also hash every read-only
          hostPath input of the llama pod (weights, runtime bundles). With
          --check, change nothing and exit 1 if the result would differ.
  verify  Check production/LOCK.json against the files in production/. Needs
          neither GitOps nor the host, so CI can run it.
  hash-tree PATH
          Print one host input's digest as JSON. sync runs this under sudo for
          inputs that the invoking user cannot read.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

LAB = Path(__file__).resolve().parents[2]
PRODUCTION = LAB / "production"
MIRROR = PRODUCTION / "gitops"
RECIPES = PRODUCTION / "recipes"
LOCK = PRODUCTION / "LOCK.json"
STATUS = PRODUCTION / "STATUS.md"
LLAMA = "k8s/workloads/apps/llama"

# GitOps paths that together rebuild and launch the llama pod.
SOURCES = [
    f"{LLAMA}/",
    "benchmarks/",
    "docs/model-storage.md",
    "scripts/publish-image-to-k3s.sh",
    "scripts/benchmark-llama.sh",
]
# Research records (over 100 MB of logs) stay in GitOps. Encrypted secrets are
# never mirrored.
EXCLUDE = [
    f"{LLAMA}/optimization/*",
    "*.enc.yaml",
    "*/__pycache__/*",
    "*.pyc",
]
# hostPath inputs that change outside a release, or that hold state.
HOST_SKIP_PREFIXES = ("/etc/", "/dev/", "/run/", "/proc/", "/sys/")

IMAGE_RE = re.compile(r"(?<![\w./-])([a-z0-9][\w.-]*(?::\d+)?/[\w./-]+(?::[\w.-]+)?@sha256:[0-9a-f]{64})")
CACHE_PATH = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "rtx3090-lab-sync" / "hash-cache.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        fd = handle.fileno()
        while chunk := handle.read(16 << 20):
            digest.update(chunk)
        # Hashing reads 180+ GB; do not keep it in the page cache.
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except (AttributeError, OSError):
            pass
    return digest.hexdigest()


def excluded(path: str) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in EXCLUDE)


def git(gitops: str, *args: str) -> bytes:
    return subprocess.run(["git", "-C", gitops, *args], check=True, capture_output=True).stdout


def export_gitops(gitops: str, ref: str) -> tuple[dict[str, bytes], dict]:
    commit = git(gitops, "rev-parse", "--verify", f"{ref}^{{commit}}").decode().strip()
    committed = git(gitops, "show", "-s", "--format=%cI", commit).decode().strip()
    archive = git(gitops, "archive", "--format=tar", commit, "--", *SOURCES)
    files: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar:
            if member.isdir() or excluded(member.name):
                continue
            if member.issym():
                files[member.name] = ("symlink:" + member.linkname).encode()
                continue
            extracted = tar.extractfile(member)
            assert extracted is not None, member.name
            files[member.name] = extracted.read()
            if member.mode & 0o111:
                files[member.name + "\0x"] = b""
    return files, {"commit": commit, "committed": committed}


def write_mirror(root: Path, files: dict[str, bytes]) -> None:
    mirror = root / "gitops"
    if mirror.exists():
        shutil.rmtree(mirror)
    executable = {name[:-2] for name in files if name.endswith("\0x")}
    for name, content in files.items():
        if name.endswith("\0x"):
            continue
        target = mirror / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if content.startswith(b"symlink:"):
            target.symlink_to(content[len("symlink:"):].decode())
            continue
        target.write_bytes(content)
        target.chmod(0o755 if name in executable else 0o644)


def tracked_files(root: Path) -> dict[str, str]:
    result = {}
    for base in ("gitops", "recipes"):
        for path in sorted((root / base).rglob("*")):
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                result[relative] = "symlink:" + os.readlink(path)
            elif path.is_file():
                result[relative] = sha256_file(path)
    return result


def image_refs(root: Path) -> dict[str, list[str]]:
    refs: dict[str, set[str]] = {}
    for base in ("gitops", "recipes"):
        for path in sorted((root / base).rglob("*")):
            if not path.is_file() or path.is_symlink() or path.stat().st_size > 2 << 20:
                continue
            try:
                text = path.read_text()
            except UnicodeDecodeError:
                continue
            for ref in IMAGE_RE.findall(text):
                refs.setdefault(ref, set()).add(path.relative_to(root).as_posix())
    return {ref: sorted(where) for ref, where in sorted(refs.items())}


def load_yaml_documents(path: Path) -> list:
    import yaml

    return [document for document in yaml.safe_load_all(path.read_text()) if document]


def pod_spec(root: Path) -> dict:
    for document in load_yaml_documents(root / "gitops" / LLAMA / "deployment.yaml"):
        if document.get("kind") == "Deployment":
            return document["spec"]["template"]["spec"]
    raise SystemExit("deployment.yaml has no Deployment")


def host_inputs(spec: dict) -> dict[str, str]:
    """Return read-only hostPath inputs as {path: volume name}."""
    writable = set()
    for container in spec.get("initContainers", []) + spec.get("containers", []):
        for mount in container.get("volumeMounts", []):
            if not mount.get("readOnly", False):
                writable.add(mount["name"])
    inputs = {}
    for volume in spec.get("volumes", []):
        host = volume.get("hostPath")
        if not host or volume["name"] in writable:
            continue
        path = host["path"]
        if path.startswith(HOST_SKIP_PREFIXES):
            continue
        inputs[path] = volume["name"]
    return dict(sorted(inputs.items()))


def hash_tree(path: Path, cache: dict[str, str] | None) -> dict:
    """Digest a file or directory: sizes, file count and one tree SHA-256."""

    def file_hash(file: Path, info: os.stat_result) -> str:
        key = f"{info.st_dev}:{info.st_ino}:{info.st_size}:{info.st_mtime_ns}"
        if cache is not None and key in cache:
            return cache[key]
        value = sha256_file(file)
        if cache is not None:
            cache[key] = value
        return value

    info = path.stat()
    if stat.S_ISREG(info.st_mode):
        return {"kind": "file", "bytes": info.st_size, "sha256": file_hash(path, info)}
    lines = []
    total = count = 0
    hf_commits = set()
    for directory, subdirectories, names in os.walk(path):
        subdirectories.sort()
        for name in sorted(names):
            file = Path(directory, name)
            relative = file.relative_to(path).as_posix()
            if file.is_symlink():
                lines.append(f"l\t{relative}\t{os.readlink(file)}")
                continue
            file_info = file.stat()
            if not stat.S_ISREG(file_info.st_mode):
                continue
            total += file_info.st_size
            count += 1
            lines.append(f"f\t{relative}\t{file_info.st_size}\t{file_hash(file, file_info)}")
            if "/.cache/huggingface/download/" in f"/{relative}" and name.endswith(".metadata"):
                first = file.read_text(errors="replace").splitlines()[:1]
                if first and re.fullmatch(r"[0-9a-f]{40}", first[0]):
                    hf_commits.add(first[0])
    result = {
        "kind": "directory",
        "files": count,
        "bytes": total,
        "tree_sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest(),
    }
    if hf_commits:
        result["huggingface_download_commits"] = sorted(hf_commits)
    return result


def scan_host(paths: dict[str, str]) -> dict[str, dict]:
    try:
        cache = json.loads(CACHE_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        cache = {}
    results = {}
    for path, volume in paths.items():
        print(f"hashing {path}", file=sys.stderr)
        try:
            entry = hash_tree(Path(path), cache)
        except PermissionError:
            completed = subprocess.run(
                ["sudo", "-n", "/usr/bin/python3", "-I", str(Path(__file__).resolve()), "hash-tree", path],
                check=True,
                capture_output=True,
                text=True,
            )
            entry = json.loads(completed.stdout)
        except FileNotFoundError:
            entry = {"kind": "missing"}
        results[path] = {"volume": volume, **entry}
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = CACHE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(cache))
    temporary.replace(CACHE_PATH)
    return results


def swap_models(root: Path) -> tuple[dict, list[str]]:
    import yaml

    for document in load_yaml_documents(root / "gitops" / LLAMA / "swap-configmap.yaml"):
        if document.get("kind") == "ConfigMap":
            config = yaml.safe_load(document["data"]["config.yaml"])
            preload = config.get("hooks", {}).get("on_startup", {}).get("preload", [])
            return config.get("models", {}), preload
    raise SystemExit("swap-configmap.yaml has no ConfigMap")


def render_status(root: Path, lock: dict) -> str:
    models, preload = swap_models(root)
    spec = pod_spec(root)
    gitops = lock["gitops"]
    out = [
        "# Production status",
        "",
        f"Generated by `tools/production/sync.py` from GitOps `{gitops['commit'][:12]}`",
        f"({gitops['committed']}). Do not edit; rerun the sync.",
        "",
        "## Models",
        "",
        "| Model | Preloaded | TTL (s) | Listed | Max length | Launcher |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, model in models.items():
        command = " ".join(str(model.get("cmd", "")).split())
        launcher = next((part for part in command.split() if part.endswith((".sh", ".py", "llama-server"))), command[:40])
        max_len = (model.get("metadata") or {}).get("max_model_len", "")
        out.append(
            f"| `{name}` | {'yes' if name in preload else ''} | {model.get('ttl', '')} | "
            f"{'no' if model.get('unlisted') else 'yes'} | {max_len} | `{launcher}` |"
        )
    for name in preload:
        model = models.get(name, {})
        out += ["", f"### `{name}` environment", "", "```text"]
        out += [str(item) for item in model.get("env", []) if not str(item).startswith("PATH=")]
        out += ["```"]
    out += ["", "## Containers", "", "| Container | Image |", "| --- | --- |"]
    for kind in ("initContainers", "containers"):
        for container in spec.get(kind, []):
            label = container["name"] + (" (init)" if kind == "initContainers" else "")
            out.append(f"| `{label}` | `{container['image']}` |")
    out += [
        "",
        "## Host inputs",
        "",
        "Read-only hostPath mounts. Hashes come from the last `--host-scan`.",
        "",
        "| Path | Volume | Size | Digest |",
        "| --- | --- | --- | --- |",
    ]
    for path, entry in lock["host_inputs"].items():
        size = f"{entry['bytes'] / 2**30:.1f} GiB" if "bytes" in entry else entry.get("kind", "")
        digest = entry.get("tree_sha256") or entry.get("sha256") or entry.get("kind", "")
        out.append(f"| `{path}` | `{entry['volume']}` | {size} | `{digest[:16]}` |")
    return "\n".join(out) + "\n"


def build(root: Path, gitops_meta: dict, host: dict[str, dict] | None, previous: dict) -> dict:
    spec = pod_spec(root)
    inputs = host_inputs(spec)
    old_host = previous.get("host_inputs", {})
    host_entries = {}
    for path, volume in inputs.items():
        if host is not None:
            host_entries[path] = host[path]
        elif path in old_host and old_host[path].get("volume") == volume:
            host_entries[path] = old_host[path]
        else:
            host_entries[path] = {"volume": volume, "kind": "unscanned"}
    lock = {
        "gitops": {"repository": "Denguinho/docker-services", **gitops_meta},
        "images": image_refs(root),
        "host_inputs": host_entries,
        "files": tracked_files(root),
    }
    (root / "LOCK.json").write_text(json.dumps(lock, indent=1, sort_keys=False) + "\n")
    (root / "STATUS.md").write_text(render_status(root, lock))
    return lock


def compare_trees(left: Path, right: Path, names: list[str]) -> list[str]:
    differences = []
    for name in names:
        a, b = left / name, right / name
        if a.is_dir() or b.is_dir():
            files_a = {p.relative_to(a).as_posix() for p in a.rglob("*") if not p.is_dir()} if a.exists() else set()
            files_b = {p.relative_to(b).as_posix() for p in b.rglob("*") if not p.is_dir()} if b.exists() else set()
            for item in sorted(files_a ^ files_b):
                differences.append(f"{name}/{item}")
            for item in sorted(files_a & files_b):
                pa, pb = a / item, b / item
                if pa.is_symlink() or pb.is_symlink():
                    same = pa.is_symlink() and pb.is_symlink() and os.readlink(pa) == os.readlink(pb)
                else:
                    same = pa.read_bytes() == pb.read_bytes()
                if not same:
                    differences.append(f"{name}/{item}")
        elif not (a.exists() and b.exists() and a.read_bytes() == b.read_bytes()):
            differences.append(name)
    return differences


def command_sync(args: argparse.Namespace) -> int:
    files, meta = export_gitops(args.gitops, args.ref)
    previous = json.loads(LOCK.read_text()) if LOCK.exists() else {}
    with tempfile.TemporaryDirectory() as scratch:
        staged = Path(scratch) / "production"
        staged.mkdir()
        write_mirror(staged, files)
        shutil.copytree(RECIPES, staged / "recipes", symlinks=True)
        host = scan_host(host_inputs(pod_spec(staged))) if args.host_scan else None
        build(staged, meta, host, previous)
        names = ["gitops", "LOCK.json", "STATUS.md"]
        differences = compare_trees(staged, PRODUCTION, names)
        if args.check:
            for item in differences:
                print(f"out of date: production/{item}")
            return 1 if differences else 0
        for name in names:
            target = PRODUCTION / name
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
            source = staged / name
            if source.is_dir():
                shutil.copytree(source, target, symlinks=True)
            else:
                shutil.copy2(source, target)
    print(f"synced GitOps {meta['commit'][:12]}: {len(differences)} paths changed")
    return 0


def command_verify(_: argparse.Namespace) -> int:
    lock = json.loads(LOCK.read_text())
    actual = tracked_files(PRODUCTION)
    problems = [f"hash mismatch or missing: production/{p}" for p in sorted(set(lock["files"]) | set(actual)) if lock["files"].get(p) != actual.get(p)]
    if lock["images"] != image_refs(PRODUCTION):
        problems.append("images section does not match the image references in production/")
    inputs = host_inputs(pod_spec(PRODUCTION))
    if list(lock["host_inputs"]) != list(inputs):
        problems.append("host_inputs does not match the deployment's read-only hostPath volumes")
    if STATUS.read_text() != render_status(PRODUCTION, lock):
        problems.append("STATUS.md is not the rendering of LOCK.json and the mirror")
    for problem in problems:
        print(problem)
    print("production/ is consistent" if not problems else f"{len(problems)} problems")
    return 1 if problems else 0


def command_hash_tree(args: argparse.Namespace) -> int:
    print(json.dumps(hash_tree(Path(args.path), None)))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sync = sub.add_parser("sync")
    sync.add_argument("--gitops", required=True, help="path to a docker-services clone")
    sync.add_argument("--ref", default="origin/main")
    sync.add_argument("--host-scan", action="store_true")
    sync.add_argument("--check", action="store_true")
    sync.set_defaults(func=command_sync)
    verify = sub.add_parser("verify")
    verify.set_defaults(func=command_verify)
    tree = sub.add_parser("hash-tree")
    tree.add_argument("path")
    tree.set_defaults(func=command_hash_tree)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

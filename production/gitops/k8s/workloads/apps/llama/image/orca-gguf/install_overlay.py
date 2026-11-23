"""Stage verified source replacements in an Orca-only Python package path."""
import hashlib
import json
import os
import shutil
from pathlib import Path

root = Path('/opt/orca')
site = Path('/opt/sparsekv/venv/lib/python3.12/site-packages')
runtime = root / 'python'
manifest = json.loads((root / 'sources.json').read_text())['overlay']
packages = {'sparsekv', 'vllm'}
replacements = {}
for relative, hashes in manifest.items():
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts or len(path.parts) < 2 or path.parts[0] not in packages:
        raise ValueError(f'Invalid overlay path: {relative}')
    original = site / path
    replacement = root / 'overlay' / path
    if hashlib.sha256(original.read_bytes()).hexdigest() != hashes['original_sha256']:
        raise ValueError(f'Base source differs: {relative}')
    if hashlib.sha256(replacement.read_bytes()).hexdigest() != hashes['patched_sha256']:
        raise ValueError(f'Patch source differs: {relative}')
    replacements[path] = replacement

for package in sorted(packages):
    shutil.copytree(site / package, runtime / package, symlinks=True,
                    copy_function=os.symlink, ignore=shutil.ignore_patterns('__pycache__'))

for relative, replacement in replacements.items():
    path = runtime / relative
    if not path.parent.resolve().is_relative_to(runtime):
        raise ValueError(f'Overlay parent escapes the Orca package path: {relative}')
    path.unlink()
    shutil.copyfile(replacement, path)

print('Staged', len(replacements), 'hash-verified replacements in', runtime)

"""Create a small private venv while sharing immutable dependency files.

The mutable vLLM/PTH Python packages are real copies, never hard links.
CUDA libraries and unrelated packages stay at their pinned base-image paths.
"""
from pathlib import Path
import hashlib
import json
import shutil

base = Path('/app/venv')
private = Path('/opt/sparsekv/venv')
relative = Path('lib/python3.12/site-packages')
private.mkdir(parents=True)
shutil.copy2(base / 'pyvenv.cfg', private / 'pyvenv.cfg')
(private / 'bin').mkdir()
for item in (base / 'bin').iterdir():
    dest = private / 'bin' / item.name
    if item.name.startswith('python'):
        if item.is_symlink():
            dest.symlink_to(item.readlink())
        else:
            shutil.copy2(item, dest)
    elif item.name == 'vllm':
        dest.write_bytes(item.read_bytes().replace(b'#!/app/venv/bin/', b'#!/opt/sparsekv/venv/bin/', 1))
        dest.chmod(item.stat().st_mode)
    else:
        dest.symlink_to(item)

(private / relative).mkdir(parents=True)

def copy_package(source, target):
    source, target = Path(source), Path(target)
    if source.suffix == '.so':
        target.symlink_to(source)
    else:
        shutil.copy2(source, target)
    return str(target)

for item in (base / relative).iterdir():
    target = private / relative / item.name
    if item.name == 'vllm':
        shutil.copytree(item, target, copy_function=copy_package,
                        ignore=shutil.ignore_patterns('__pycache__'))
    elif item.name.startswith('pth_flash_attn'):
        if item.is_dir():
            shutil.copytree(item, target, ignore=shutil.ignore_patterns('__pycache__'))
        else:
            shutil.copy2(item, target)
    else:
        target.symlink_to(item)

# The image check compares these after installation to prove the old runtime survived.
paths = [base / relative / name for name in ('vllm', 'pth_flash_attn')]
hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
          for directory in paths for p in directory.rglob('*.py')}
Path('/opt/sparsekv/base-runtime-hashes.json').write_text(json.dumps(hashes, sort_keys=True))

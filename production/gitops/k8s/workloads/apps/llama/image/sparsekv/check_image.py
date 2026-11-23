"""CPU check: SparseKV imports privately and the original runtime is intact."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

root = Path('/opt/sparsekv')
assert sys.prefix == str(root / 'venv'), sys.prefix
for name, digest in json.loads((root / 'base-runtime-hashes.json').read_text()).items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
import vllm
import pth_flash_attn.sparse_kv
import sparsekv.tier_connector
assert Path(vllm.__file__).is_relative_to(root), vllm.__file__
assert Path(pth_flash_attn.sparse_kv.__file__).is_relative_to(root)
assert Path(sparsekv.tier_connector.__file__).is_relative_to(root)
assert hasattr(sparsekv.tier_connector, 'HostTierConnector')
subprocess.run(['bash', str(root / 'sparsekv/install.sh'), str(root / 'venv'), '--check'], check=True)
assert hashlib.sha256((root / 'venv/share/sparsekv/basis27.pt').read_bytes()).hexdigest() == 'dee8b28281d5837ad512a560e4d5c31af245cbe42dc1dea66dc1510775b4b99b'
print('SPARSEKV_CPU_CHECK: private runtime verified; base runtime unchanged', flush=True)

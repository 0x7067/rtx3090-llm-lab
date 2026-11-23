"""CPU-only image identity checks plus isolated launcher argv qualification."""
import ast
import glob
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
from apply_patch import patched
import check_cpu


def main():
    assert not glob.glob('/dev/nvidia*')
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert os.environ.get('NVIDIA_VISIBLE_DEVICES') == 'void'
    assert os.environ.get('NORMAL120_GDN_FP16', '0') == '0'
    site = Path(sysconfig.get_paths()['purelib'])
    assert not (site/'normal120_gdn_plugin.py').exists()
    assert not any(e.name == 'normal120_gdn_fp16' for e in importlib.metadata.entry_points(group='vllm.general_plugins'))
    source = (site/'vllm/_custom_ops.py').read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='fused_gdn_decode_post_conv_mtp')
    wrapper = ''.join(source.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
    wrapper_hash = hashlib.sha256(wrapper.encode()).hexdigest()
    assert wrapper_hash == '1b88624476906a176cf267ca19a36a51733f3f7415f4d3ffff5a9d160bf3d654'
    original = (Path(__file__).parent/'upstream/single-user/start_qwen.sh').read_bytes()
    installed = Path('/app/single-user/start_qwen.sh').read_bytes()
    assert installed == patched(original)
    subprocess.run(['bash','-n','/app/single-user/start_qwen.sh'],check=True)
    check_cpu.check()
    assert 'torch' not in sys.modules and 'vllm' not in sys.modules
    report = {'cpu_checks':'passed','gpu_execution':False,'model_requests':False,
              'installed_launcher_sha256':hashlib.sha256(installed).hexdigest(),
              'original_wrapper_sha256':wrapper_hash,'custom_gdn_plugin_absent':True,
              'gdn_gate_default_off':True,'torch_vllm_not_imported':True,
              'argv_defaults_k7_and_opt_in_k3_k5':True,'qualification':False}
    print('NORMAL120_CPU_RECEIPT\n'+json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()

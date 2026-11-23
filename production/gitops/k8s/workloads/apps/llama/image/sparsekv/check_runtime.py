"""Check the published image and production launcher without a GPU or server."""
from pathlib import Path
import json
import re
import shutil
import subprocess
import tempfile
import yaml

HERE = Path(__file__).resolve().parent
APP = HERE.parents[1]
OUT = HERE / 'qualification'
meta = json.loads((OUT / 'build-metadata.json').read_text())
image = '127.0.0.1:5000/home-llm-strata:sparsekv-v1@' + meta['containerimage.digest']
ctr = ['sudo', '-n', 'k3s', 'ctr', '-n', 'k8s.io']
with (OUT / 'image-pull.log').open('w') as log:
    subprocess.run([*ctr, 'images', 'pull', '--plain-http', image], check=True,
                   stdout=log, stderr=subprocess.STDOUT, timeout=600)
pull_log = OUT / 'image-pull.log'
pull_log.write_text('\n'.join(re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', line).rstrip()
                              for line in pull_log.read_text().splitlines()) + '\n')
config = yaml.safe_load((APP / 'swap-configmap.yaml').read_text())['data']['config.yaml']
model = yaml.safe_load(config)['models']['qwen3.8-27b-hyperqwen']
env = dict(item.split('=', 1) for item in model['env'])
env.update(PORT='5890', PRINT_ARGV='1', CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='void',
           LLAMA_API_TOKEN='validation-placeholder', HOME='/tmp', HQ_TMP='/tmp/sparsekv',
           TORCH_EXTENSIONS_DIR='/tmp/torch', XDG_CACHE_HOME='/tmp', HF_HOME='/tmp/hf',
           VLLM_OFFLOAD_KEEP_SHM='1', PYTHONDONTWRITEBYTECODE='1')
with tempfile.TemporaryDirectory(prefix='sparsekv-runtime-') as temp:
    temp = Path(temp)
    (temp / 'config.yaml').write_text(config)
    shutil.copy2(APP / 'chat_template.jinja', temp / 'chat_template.jinja')
    command = [*ctr, 'run', '--rm', '--user', '1000:1000', '--cpus', '2',
               '--memory-limit', '2147483648',
               '--mount', f'type=bind,src={temp},dst=/chat-template,options=rbind:ro',
               '--mount', 'type=bind,src=/srv/llama-nvme/hyperqwen-bakeoff/models,dst=/app/models,options=rbind:ro']
    for name, value in env.items():
        command += ['--env', name + '=' + value]
    command += [image, 'sparsekv-adoption-cpu', '/bin/bash', '-euc',
                '/opt/sparsekv/venv/bin/python /opt/sparsekv/check_image.py\n'
                '/usr/local/bin/llama-swap -config /chat-template/config.yaml -validate\n'
                'bash /opt/sparsekv/serve-sparse.sh']
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
    (OUT / 'runtime-check.stdout').write_text(result.stdout)
    (OUT / 'runtime-check.stderr').write_text(result.stderr)
    result.check_returncode()
    lines = result.stdout.splitlines()
    start = lines.index('venv/bin/vllm')
    argv = lines[start:]
    expected = (HERE / 'benchmarked.argv').read_text().splitlines()
    expected = expected[expected.index('venv/bin/vllm'):]
    expected[expected.index('--port') + 1] = '5890'
    assert argv == expected, {'actual': argv, 'benchmarked': expected}
    def option(name):
        return argv[argv.index(name) + 1]
    assert option('--host') == '127.0.0.1', argv
    assert option('--port') == '5890', argv
    assert option('--max-model-len') == '200000', argv
    assert option('--max-num-seqs') == '4', argv
    assert option('--chat-template') == '/chat-template/chat_template.jinja', argv
    assert 'HostTierConnector' in option('--kv-transfer-config'), argv
    assert 'SPARSEKV_CPU_CHECK: private runtime verified; base runtime unchanged' in result.stdout
    (OUT / 'runtime-check.json').write_text(json.dumps({
        'passed': True, 'image': image, 'gpu_execution': False, 'argv': argv,
        'base_runtime_unchanged': True, 'llama_swap_config_valid': True,
        'benchmark_argv_match_except_port': True}, indent=2) + '\n')
print('Published image, private runtime, and production launcher checks passed')

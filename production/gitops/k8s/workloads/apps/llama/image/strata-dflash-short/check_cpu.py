"""CPU-only exact-source launcher argv checks; no GPU/model imports or host secrets."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from apply_patch import patched

ROOT = Path(__file__).resolve().parent

def check():
    provenance = json.loads((ROOT / 'provenance.json').read_text())
    for name, digest in provenance['sha256'].items():
        assert hashlib.sha256((ROOT / 'upstream' / name).read_bytes()).hexdigest() == digest
    original = (ROOT / 'upstream/single-user/start_qwen.sh').read_bytes()
    try:
        patched(original + b'\n')
    except ValueError:
        pass
    else:
        raise AssertionError('Drifted launcher accepted')
    with tempfile.TemporaryDirectory(prefix='normal120-dflash-cpu-') as tmp:
        tree = Path(tmp)
        shutil.copytree(ROOT / 'upstream', tree, dirs_exist_ok=True)
        bin_dir = tree / 'venv/bin'
        bin_dir.mkdir(parents=True)
        stubs = {
            'python': '#!/bin/sh\ncase "$*" in *pth_flash_attn*) exit 0;; *) echo "Unexpected Python invocation" >&2; exit 97;; esac\n',
            'nvidia-smi': '#!/bin/sh\nexit 98\n',
            'nvcc': '#!/bin/sh\nexit 98\n',
            'vllm': '#!/bin/sh\nprintf "@@ARGV\\n"\nprintf "%s\\n" "$@"\nprintf "@@ENV\\n"\nenv | sort | sed -n "/^VLLM_/p"\n',
        }
        for name, body in stubs.items():
            path = bin_dir / name
            path.write_text(body)
            path.chmod(0o755)
        draft = tree / 'models/Qwen3.8-27B-DFlash2-W4A16'
        draft.mkdir(parents=True)
        (draft / 'model.safetensors').touch()
        script = tree / 'single-user/start_qwen.sh'
        # ConfigMap sources are read-only; only the disposable copy is mutable.
        script.chmod(0o600)
        env = dict(PATH=f'{bin_dir}:/usr/bin:/bin', HOME=str(tree), CUDA_HOME='/not-used',
            VLLM_OFFLOAD_KEEP_SHM='1', VLLM_API_KEY='cpu-test-placeholder',
            SPEC='dflash2', CTX='long', PTH_FA='1', FA2_FP8KV='0',
            MODEL='/cpu-test/target', MAX_SEQS='4', DFLASH_MAX_LEN='150000',
            KV_MEM='6442450944', VISION='1', VISION_OFFLOAD='1', EMBED_UVA='1',
            INT8_ACT='int8', INT8_LAYERS='mlp', PREFIX_CACHE='1', HOST='127.0.0.1')
        def run(data, **overrides):
            script.write_bytes(data)
            subprocess.run(['bash', '-n', str(script)], check=True)
            return subprocess.run(['bash', str(script)], env={**env, **overrides},
                                  capture_output=True, text=True, timeout=15)
        new = patched(original)
        control = run(original)
        default = run(new)
        assert control.returncode == default.returncode == 0, (control.stderr, default.stderr)
        assert control.stdout == default.stdout and control.stderr == default.stderr
        assert run(new, NORMAL120_PTH_SHORT_DFLASH='1').stdout == default.stdout
        for k in ('3', '5'):
            assert run(original, DFLASH_TOKENS=k).returncode != 0
            assert run(new, DFLASH_TOKENS=k).returncode != 0
            assert run(new, DFLASH_TOKENS=k, NORMAL120_PTH_SHORT_DFLASH='2').returncode != 0
            result = run(new, DFLASH_TOKENS=k, NORMAL120_PTH_SHORT_DFLASH='1')
            assert result.returncode == 0, result.stderr
            argv = result.stdout.split('@@ARGV\n', 1)[1].split('@@ENV\n', 1)[0].splitlines()
            after = lambda key: argv[argv.index(key)+1]
            spec = json.loads(after('--speculative-config'))
            assert spec['num_speculative_tokens'] == int(k)
            assert spec['attention_backend'] == 'CUSTOM' and spec['draft_sample_method'] == 'probabilistic'
            assert after('--attention-backend') == 'CUSTOM' and after('--kv-cache-dtype') == 'int8_per_token_head'
            assert after('--max-model-len') == '150000' and after('--max-num-seqs') == '4'
            assert '--kv-cache-memory=6442450944' in argv and '--language-model-only' not in argv
            assert 'VLLM_EMBED_UVA=1\n' in result.stdout and 'VLLM_VISION_CPU_OFFLOAD_GB=1\n' in result.stdout
            assert json.loads(after('--compilation-config'))['max_cudagraph_capture_size'] == 4*(int(k)+1)
            for override in ({'LOOKUP':'0'}, {'DYNAMIC_K':'[[1,4,3]]'},
                             {'DRAFT_METHOD':'dspark'}, {'EXTRA_ARGS':'--attention-backend=FLASH_ATTN'},
                             {'EXTRA_ARGS':'--kv-offloading-size=4'}, {'CTX':'fast'}):
                rejected = run(new, DFLASH_TOKENS=k, NORMAL120_PTH_SHORT_DFLASH='1', **override)
                assert rejected.returncode != 0 and '@@ARGV' not in rejected.stdout, override
        for k in ('1','2','4','6','8','9','15'):
            assert run(new, DFLASH_TOKENS=k, NORMAL120_PTH_SHORT_DFLASH='1').returncode != 0
        for overrides in ({'SPEC':'mtp','DFLASH_TOKENS':'5'}, {'PTH_FA':'0','DFLASH_TOKENS':'5'}):
            before=run(original, **overrides)
            after=run(new, NORMAL120_PTH_SHORT_DFLASH='1', **overrides)
            assert (before.returncode,before.stdout,before.stderr)==(after.returncode,after.stdout,after.stderr)
    print(json.dumps({'cpu_checks':'passed','source_sha256':hashlib.sha256(original).hexdigest(),
                      'patched_sha256':hashlib.sha256(new).hexdigest(),
                      'gpu_execution':False,'qualification':False}, indent=2))

if __name__ == '__main__':
    check()

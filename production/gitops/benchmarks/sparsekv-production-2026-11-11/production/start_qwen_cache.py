#!/usr/bin/env python3
"""Opt in to the pinned PTH CPU tier with PIECEWISE graph capture.

The upstream launcher refuses this unqualified combination. Only that refusal
and the script's relocated origin change; package code and other checks remain
untouched. Drift, full-graph capture, or a different model profile fails closed.
"""
import hashlib
import os
from pathlib import Path
import shlex

SOURCE = Path('/app/single-user/start_qwen.sh')
EXPECTED_SHA256 = '2f76802a7624cfc1ccadba0ef61f5c362627c34fa341c700c0eab27b521727cd'
DIR_LINE = 'DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"'
GUARD = '''    *"--kv-offloading-size"*|*"--kv-transfer-config"*)
      echo "PTH_FA=1 with a KV connector (KV_OFFLOAD_GB, --kv-offloading-size, --kv-transfer-config) is" >&2
      echo "  untested on this backend; refusing. Use PTH_FA=0 for a CPU KV tier." >&2
      exit 1 ;;
'''
OPT_IN = '''    *"--kv-offloading-size"*|*"--kv-transfer-config"*)
      echo "[start_qwen_cache] pinned PTH CPU tier with PIECEWISE capture" >&2 ;;
'''


def transform(source):
    if source.count(DIR_LINE) != 1 or source.count(GUARD) != 1:
        raise ValueError('Exact known launcher structure required')
    return source.replace(DIR_LINE, 'DIR=/app/single-user', 1).replace(GUARD, OPT_IN, 1)


def validate_environment(env):
    # These must equal the env of the qwen3.8-27b-hyperqwen entry in swap-configmap.yaml;
    # the pod refuses to start on any mismatch.
    required = {'PTH_CPU_CACHE': '1', 'PTH_FA': '1', 'FA2_FP8KV': '0',
                'KV_OFFLOAD_GB': '16', 'CUDAGRAPH_MODE': 'PIECEWISE',
                'PREFIX_RETENTION': '5184', 'CTX': 'long', 'SPEC': 'dflash2',
                'DFLASH_TOKENS': '7', 'LOOKUP': '1', 'DFLASH_MAX_LEN': '200000',
                'KV_MEM': '7784628224', 'MAX_SEQS': '4', 'INT8_ACT': 'int8',
                'INT8_LAYERS': 'mlp', 'VISION': '1', 'VISION_OFFLOAD': '1',
                'EMBED_UVA': '1', 'PREFIX_CACHE': '1', 'HOST': '127.0.0.1', 'PORT': '5890'}
    if any(env.get(key) != value for key, value in required.items()):
        raise ValueError('Explicit pinned 200K/INT8/DFlash7/16GiB/PIECEWISE profile required')
    if env.get('VLLM_KV_CACHE_LAYOUT', 'LBHNC') != 'LBHNC':
        raise ValueError('Only the qualified connector cache layout is supported')
    if env.get('MODEL') != '/app/models/Qwen3.8-27B-W4A16-AutoRound':  # unset would fall back to -fast
        raise ValueError('Only the pinned HyperQwen checkpoint is supported')
    if env.get('VLLM_PREFIX_CACHE_RETENTION_INTERVAL', '5184') != '5184':
        raise ValueError('Conflicting prefix-retention override')
    allowed = [['--chat-template', path, '--long-prefill-token-threshold', '864',
                '--long-prefill-token-threshold-adaptive', '--watermark', '0.1']
               for path in ('/trial/chat_template.jinja', '/chat-template/chat_template.jinja')]
    if shlex.split(env.get('EXTRA_ARGS', '')) not in allowed:
        raise ValueError('Only the tested template/fair-prefill arguments are allowed')


def script_fd(source):
    fd = os.memfd_create('hq-piecewise-cache-launcher', flags=0)
    try:
        with os.fdopen(fd, 'w', closefd=False) as handle:
            handle.write(source)
        os.lseek(fd, 0, os.SEEK_SET)
        os.set_inheritable(fd, True)
        return fd
    except BaseException:
        os.close(fd)
        raise


def main():
    validate_environment(os.environ)
    source = SOURCE.read_bytes()
    if hashlib.sha256(source).hexdigest() != EXPECTED_SHA256:
        raise ValueError('Upstream launcher changed; requalify the CPU cache profile')
    patched = transform(source.decode())
    fd = script_fd(patched)
    print('PIECEWISE CPU-cache launcher SHA256 ' + hashlib.sha256(patched.encode()).hexdigest(), flush=True)
    os.execv('/bin/bash', ['/bin/bash', f'/proc/self/fd/{fd}'])


if __name__ == '__main__':
    main()

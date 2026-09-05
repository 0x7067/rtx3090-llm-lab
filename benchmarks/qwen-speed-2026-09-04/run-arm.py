#!/usr/bin/env python3
"""Screen one serving change, with a fresh process and bounded cleanup."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument('tag')
p.add_argument('--image')
p.add_argument('--draft', type=int, default=7)
p.add_argument('--parallel', type=int, default=1)
p.add_argument('--threads', type=int, default=8)
p.add_argument('--batch', type=int, default=512)
p.add_argument('--ubatch', type=int, default=512)
p.add_argument('--prefill-tokens', type=int, default=0)
p.add_argument('--prefill-cublas', type=int, default=0)
p.add_argument('--sort-chunk-mib', type=int, default=0)
p.add_argument('--ngram-max', type=int, default=64)
p.add_argument('--q4-draft', action='store_true')
p.add_argument('--cpu-vision', action='store_true')
p.add_argument('--depth', type=int, default=0)
p.add_argument('--qualify', action='store_true')
p.add_argument('--context-check', action='store_true')
p.add_argument('--parallel-context-check', action='store_true')
p.add_argument('--stress-context-check', action='store_true')
p.add_argument('--checks-only', action='store_true')
a = p.parse_args()
base = json.loads((ROOT / 'baseline-command.json').read_text())
if a.image:
    base['image'] = a.image
cmd = base['args'][:]
def replace(flag, value):
    cmd[cmd.index(flag) + 1] = str(value)
replace('--spec-draft-n-max', a.draft)
replace('--parallel', a.parallel)
replace('--threads', a.threads)
replace('--threads-batch', a.threads)
replace('--batch-size', a.batch)
replace('--ubatch-size', a.ubatch)
if a.ngram_max != 64:
    cmd += ['--spec-ngram-mod-n-max', str(a.ngram_max)]
if a.parallel > 1:
    cmd += ['--kv-unified', '--kv-unified-per-slot', '131072']
if a.cpu_vision:
    cmd += ['--no-mmproj-offload']
if a.q4_draft:
    replace('--model-draft', '/trial/DFlash2-Q4_K_M.gguf')
name = 'qwen-speed-trial'
(ROOT / (a.tag + '-command.json')).write_text(json.dumps(cmd, indent=2) + '\n')
run = ['docker', 'run', '-d', '--name', name, '--gpus', 'all', '--network', 'host',
       '-v', '/data/buttercup_6tb/k3s/llama-models:/models:ro',
       '-v', '/tmp/qwen-speed:/trial:ro',
       '-e', 'GGML_CUDA_MMVQ_NE11_MAX=3', '-e', 'GGML_CUDA_MMQ_SMALLN=3',
       '--entrypoint', '/usr/local/bin/llama-server', base['image'], *cmd]
if a.prefill_cublas:
    run[2:2] = ['-e', f'GGML_CUDA_PREFILL_CUBLAS_MIN={a.prefill_cublas}']
if a.sort_chunk_mib:
    run[2:2] = ['-e', f'GGML_CUDA_SORT_CHUNK_MIB={a.sort_chunk_mib}']
(ROOT / (a.tag + '-launch.json')).write_text(json.dumps(run, indent=2) + '\n')
try:
    subprocess.run(run, check=True, stdout=subprocess.DEVNULL)
    ready = False
    for _ in range(150):
        try:
            req = urllib.request.Request('http://127.0.0.1:18089/health', headers={'User-Agent':'OpenAI File Downloader, XaiImageApiFetch/1.0'})
            with urllib.request.urlopen(req, timeout=2) as r:
                ready = r.status == 200
            if ready:
                break
        except Exception:
            pass
        if subprocess.check_output(['docker','inspect','-f','{{.State.Running}}',name],text=True).strip() != 'true':
            raise RuntimeError('server exited during load')
        time.sleep(2)
    if not ready:
        raise RuntimeError('server startup exceeded 300 seconds')
    shared = ['--base-url','http://127.0.0.1:18089/v1','--tag',a.tag,'--reasoning','medium','--out',str(ROOT / 'screening.jsonl')]
    if a.prefill_tokens:
        subprocess.run([sys.executable,str(ROOT/'bench.py'),'prefill','--tokens',str(a.prefill_tokens),*shared],check=True,timeout=420)
    for job in ([] if a.checks_only else [ ['quality'], ['session','--turns','8','--max-tokens','4096','--preamble-tokens',str(a.depth)], ['concurrent','--n','4'], ['sustained','--max-tokens','2048'] ]):
        subprocess.run([sys.executable,str(ROOT/'bench.py'),*job,*shared],check=True,timeout=420,env={**os.environ,"QWEN_OUTPUTS":str(ROOT / (a.tag + "-" + job[0] + "-outputs.jsonl"))})
    if a.qualify:
        subprocess.run([sys.executable,str(ROOT/'qualify-vision.py'),a.tag],check=True,timeout=600)
    if a.context_check:
        subprocess.run([sys.executable,str(ROOT/'check-context.py'),a.tag],check=True,timeout=720)
    if a.parallel_context_check:
        subprocess.run([sys.executable,str(ROOT/'check-context.py'),a.tag,'parallel'],check=True,timeout=720)
    if a.stress_context_check:
        subprocess.run([sys.executable,str(ROOT/'check-context.py'),a.tag,'stress'],check=True,timeout=900)
finally:
    with (ROOT / (a.tag + '-server.log')).open('w') as f:
        subprocess.run(['docker','logs',name],stdout=f,stderr=f)
    subprocess.run(['docker','rm','-f',name],stdout=subprocess.DEVNULL)

#!/usr/bin/env python3
"""Matched existing-Orca extension upgrade and separate DFlash2 candidate."""
import importlib.util
import json
import os
from pathlib import Path

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'orcasaq-latest-2026-10-07'
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
p=module('prefix',OLD/'run-prefix.py')
r=module('resident',HERE.parents[1]/'benchmarks/gsq-resident-2026-10-08/run-frozen.py')
w=p.w
r.w=w
r.command=w.command
r.h=w.h
w.monitor=r.guarded_monitor
p.ROOT=HERE/'qualification'
original_command=w.command
original_launch=p.launch
original_sessions=p.session_tests
OLD_IMAGE='sha256:bf0c4eebc48d0db6d8c779be8b81018ced33acfbcc1ac98011ad9ac82c4ee45c'
NEW_IMAGE='sha256:55c96bdd11d4e09ffea2ee6cc85605b18e1f5eb84fdb8a64b51271d1c1661872'
current=''

def command(args,*a,**kw):
    if args[:3]==['docker','run','-d']:
        args=list(args)
        args[args.index('--name')+1]='orca-exl16-'+current
        w.active='orca-exl16-'+current
        args[-1]=OLD_IMAGE if current=='control' else NEW_IMAGE
        for i,arg in enumerate(args):
            if arg=='/tmp/orcasaq-latest-tuning-cache:/cache':
                args[i]=f'/tmp/orca-exl16-cache/{current}:/cache'
            if current=='dflash2' and arg.startswith('EXTRA='):
                draft={'method':'dflash','model':'/models/dflash2','num_speculative_tokens':7,'kv_cache_dtype':'fp8_e4m3','attention_backend':'FLASH_ATTN'}
                args[i]='EXTRA=--speculative-config '+json.dumps(draft,separators=(',',':'))+' --mamba-cache-mode align --enable-prompt-tokens-details --prefix-match-unit 64'
        args[-1:-1]=['-e','VLLM_USE_V2_MODEL_RUNNER=1']
        if current=='dflash2':args[-1:-1]=['-v','/data/models/qwen3.8/z-lab/Qwen3.8-27B-DFlash2:/models/dflash2:ro']
        (w.ROOT/'actual-launch.json').write_text(json.dumps(args,indent=2))
    return original_command(args,*a,**kw)
w.command=command
r.command=command

def launch(name,*a,**kw):
    global current
    current=name
    r.ROOT=p.ROOT/name
    thread=original_launch(name,*a,**kw)
    r.thread=thread
    return thread
p.launch=launch

def stop(thread=None):
    r.stop()
p.tm.stop=stop

def sessions(arm,generation_effort='low'):
    w.CONCURRENCY_CASES=[(1,0,False),(2,0,False)]
    w.concurrency(arm)
    original_sessions(arm,generation_effort)
p.session_tests=sessions

def main():
    os.environ['KUBECONFIG']='/home/denguinho/.kube/config'
    config=command(['kubectl','-n','apps','get','configmap','llama-swap-config','-o','jsonpath={.data}'])
    HERE.mkdir(exist_ok=True)
    (HERE/'config-before.json').write_text(config)
    try:
        p.main(profiles=[('control',False,64,True,True),('exl16',False,64,True,True),('dflash2',False,64,True,True)],gpu_fraction='0.92',generation_effort='low')
    finally:
        same=command(['kubectl','-n','apps','get','configmap','llama-swap-config','-o','jsonpath={.data}'])==config
        (HERE/'config-restoration.json').write_text(json.dumps({'unchanged':same}))
        if not same:raise RuntimeError('Production ConfigMap changed')

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Qualify one screened candidate without changing production configuration."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('screen',HERE/'run-screen.py')
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
r=d.r
name=sys.argv[1]
assert name in ('ampere-no-spec','ampere-mtp2-f32','ampere-mtp2-q8-state')
screen=json.loads((HERE/'screen/screen-results.json').read_text())
assert next(x for x in screen if x['arm']==name).get('screen_complete'), 'Candidate did not complete capacity and smoke gates'
d.ROOT=HERE/('qualification-'+name)
d.ROOT.mkdir()
r.ROOT=d.ROOT;r.w.ROOT=d.ROOT
r.IMAGE=d.CANDIDATE
d.current={'name':name,'fork':True,'ubatch':128,'spec':'mtp-file-none' if name=='ampere-no-spec' else 'mtp','depth':2,'state':'q8_0' if name.endswith('q8-state') else 'f32'}
arm={'name':name,'model':'gsq-rco-27b','base_url':'http://127.0.0.1:18185/v1','metadata':{'kv':'q8_0','shared_pool':320000,'parallel':2,'image':r.IMAGE,'microbatch':128,'state':d.current['state'],'draft_window':8192 if d.current['spec']=='mtp' else 0}}
command=d.command

def main():
    os.environ['KUBECONFIG']='/home/denguinho/.kube/config'
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,r.w.interrupted)
    for unit in ('vram-thermal-guard.service','vram-fan-curve.service'):
        command(['systemctl','is-active','--quiet',unit])
    assert not Path('/run/gpu-thermal-trip').exists()
    before={key:command(args) for key,args in {
      'replicas':['kubectl','-n','apps','get','deployment','llama','-o','jsonpath={.spec.replicas}'],
      'suspend':['kubectl','-n','flux-system','get','kustomization','apps','-o','jsonpath={.spec.suspend}'],
      'template':['kubectl','-n','apps','get','deployment','llama','-o','jsonpath={.spec.template}'],
      'config':['kubectl','-n','apps','get','configmap','llama-swap-config','-o','jsonpath={.data}'],
    }.items()}
    r.h.write_new(d.ROOT/'state-before.json',before)
    plan=copy.deepcopy(r.h.load_plan(HERE.parents[1]/'benchmarks/gsq-resident-2026-10-08/plan-tools.json'))
    plan.pop('sha256');plan['arms']=[arm];plan['sha256']=r.h.digest(plan)
    r.h.write_new(d.ROOT/'plan-tools.json',plan)
    try:
        if before['suspend']!='true':command(['flux','suspend','kustomization','apps'])
        command(['kubectl','-n','apps','scale','deployment','llama','--replicas=0'])
        command(['kubectl','-n','apps','wait','--for=delete','pod','-l','app.kubernetes.io/name=llama','--timeout=10m'],660)
        for _ in range(60):
            if not command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']):break
            time.sleep(5)
        else:raise RuntimeError('Unowned GPU process')
        r.launch(arm,320000,'quality')
        try:
            r.full(plan,arm)
            r.q.run(r.w,arm,d.ROOT/(name+'-concurrent-quality.jsonl'))
        finally:r.stop()
        r.long(arm,320000,9300,2,20)
    finally:
        r.stop()
        command(['kubectl','-n','apps','scale','deployment','llama','--replicas='+before['replicas']])
        if before['suspend']!='true':command(['flux','resume','kustomization','apps'])
        command(['kubectl','-n','apps','rollout','status','deployment/llama','--timeout=15m'],960)
        checks={
            'template':command(['kubectl','-n','apps','get','deployment','llama','-o','jsonpath={.spec.template}'])==before['template'],
            'config':command(['kubectl','-n','apps','get','configmap','llama-swap-config','-o','jsonpath={.data}'])==before['config'],
            'replicas':command(['kubectl','-n','apps','get','deployment','llama','-o','jsonpath={.spec.replicas}'])==before['replicas'],
            'suspend':command(['kubectl','-n','flux-system','get','kustomization','apps','-o','jsonpath={.spec.suspend}'])==before['suspend'],
        }
        r.h.write_new(d.ROOT/'restoration.json',checks)
        if not all(checks.values()):raise RuntimeError('Restoration mismatch')
if __name__=='__main__':main()

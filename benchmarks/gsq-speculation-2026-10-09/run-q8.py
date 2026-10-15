#!/usr/bin/env python3
"""Bounded capacity and matched speed screening; preserve production ownership."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import time

ROOT = Path(__file__).resolve().parent/'q8-draft'
ROOT.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location('resident', ROOT.parent.parent/'gsq-resident-2026-10-08/run-frozen.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
r.ROOT = ROOT
r.w.ROOT = ROOT
original_command = r.command
current = None

def command(args, *a, **kw):
    if args[:3] == ['docker','run','-d']:
        args = list(args)
        args[args.index('--ubatch-size')+1] = '256'
        args += ['--spec-draft-type-k','q8_0','--spec-draft-type-v','q8_0']
        if current['spec'] == 'mtp':
            args[args.index('--model')+1] = '/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf'
            args += ['--spec-type','draft-mtp','--spec-draft-n-max',str(current['depth']), '--spec-draft-sampling','probabilistic']
        elif current['spec'] == 'dflash':
            idx = args.index(r.IMAGE)
            args[idx:idx] = ['-v','/data/models/qwen3.8/z-lab/Qwen3.8-27B-DFlash2-GGUF:/draft:ro']
            args += ['--spec-type','draft-dflash','--spec-draft-model','/draft/Qwen3.8-27B-DFlash2-Q4_K_M.gguf','--spec-draft-n-max','7','--spec-draft-ngl',str(current['draft_ngl']),'--spec-draft-type-k','q8_0','--spec-draft-type-v','q8_0','--spec-draft-sampling','probabilistic']
        (ROOT/(r.w.active+'-actual-launch.json')).write_text(json.dumps(args,indent=2))
    return original_command(args,*a,**kw)
r.command = command
r.w.command = command

def main():
    global current
    os.environ['KUBECONFIG']='/home/denguinho/.kube/config'
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,r.w.interrupted)
    for unit in ('vram-thermal-guard.service','vram-fan-curve.service'):
        command(['systemctl','is-active','--quiet',unit])
    before = {key:command(args) for key,args in {
        'replicas':['kubectl','-n','apps','get','deployment','llama','-o','jsonpath={.spec.replicas}'],
        'suspend':['kubectl','-n','flux-system','get','kustomization','apps','-o','jsonpath={.spec.suspend}'],
        'template':['kubectl','-n','apps','get','deployment','llama','-o','jsonpath={.spec.template}'],
        'config':['kubectl','-n','apps','get','configmap','llama-swap-config','-o','jsonpath={.data}'],
    }.items()}
    r.h.write_new(ROOT/'state-before.json',before)
    arms=[{'name':'baseline-ub256','spec':'none'}, {'name':'mtp1-q8-ub256','spec':'mtp','depth':1}, {'name':'mtp2-q8-ub256','spec':'mtp','depth':2}]
    r.h.write_new(ROOT/'screen-plan.json',{'pool':320000,'kv':'q8_0','draft_kv':'q8_0','ubatch':256,'arms':arms,'image':r.IMAGE,'speed_cases':[[1,0,False],[2,0,False],[2,2400,False]],'sampling':'target T1/top-p .95/top-k20; probabilistic rejection for speculative arms'})
    r.w.CONCURRENCY_CASES=[(1,0,False),(2,0,False),(2,2400,False)]
    results=[]
    try:
        if before['suspend']!='true':command(['flux','suspend','kustomization','apps'])
        command(['kubectl','-n','apps','scale','deployment','llama','--replicas=0'])
        command(['kubectl','-n','apps','wait','--for=delete','pod','-l','app.kubernetes.io/name=llama','--timeout=10m'],660)
        if command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']):raise RuntimeError('Unowned GPU process')
        for current in arms:
            arm={**current,'model':'gsq-rco-27b','base_url':'http://127.0.0.1:18185/v1','metadata':{'kv':'q8_0'}}
            status={'arm':current['name']}
            try:
                r.launch(arm,320000,'screen')
                status['ready']=True
                smoke=r.checked_request(arm,{'messages':[{'role':'user','content':'Return only the result of 17+25.'}],'temperature':0,'max_tokens':256,'reasoning_effort':'low','stream':True,'stream_options':{'include_usage':True}})
                r.h.write_new(ROOT/(arm['name']+'-smoke.json'),smoke)
                status['smoke_pass']=smoke.get('content','').strip()=='42'
                if not status['smoke_pass']:raise RuntimeError('Arithmetic smoke failed')
                r.w.concurrency(arm)
                status['screen_complete']=True
            except Exception as exc:
                status['error']=str(exc)
                print('ARM FAILED '+json.dumps(status),flush=True)
            finally:
                r.stop()
                results.append(status)
                (ROOT/'screen-results.json').write_text(json.dumps(results,indent=2))
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
        r.h.write_new(ROOT/'restoration.json',checks)
        if not all(checks.values()):raise RuntimeError('Production restoration mismatch')

if __name__=='__main__':main()

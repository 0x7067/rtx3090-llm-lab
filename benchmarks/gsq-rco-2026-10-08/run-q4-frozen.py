#!/usr/bin/env python3
"""Matched GSQ cache qualification with an independently restored production owner."""
import concurrent.futures as cf
import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'orca-thinkingcap-2026-10-07'
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result
w = module('window', SOURCE / 'run.py')
q = module('concurrent_quality', SOURCE / 'concurrent_quality.py')
h = w.h
w.ROOT = ROOT
IMAGE = 'ghcr.io/ggml-org/llama.cpp@sha256:a863a2d1dc2cbc5326c995ae380eec9c57f37a7e1cac944f0c6f43e4e9afff23'
MODEL_DIR = '/data/models/qwen3.8/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF'
MODEL = 'Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf'
command = w.command
thread = None
sequence = 0

def guarded_monitor(container, output):
    paused=False
    pause_started=None
    last_global=0
    try:
        with output.open('x') as f:
            while not w.stop_monitor.wait(2 if paused else 5):
                available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))
                psi=float(Path('/proc/pressure/memory').read_text().splitlines()[0].split()[1].split('=')[1])
                pid=int(command(['docker','inspect',container,'--format','{{.State.Pid}}']))
                group=Path('/sys/fs/cgroup')/Path(f'/proc/{pid}/cgroup').read_text().split(':',2)[2].strip().lstrip('/')
                cg={name:(group/name).read_text().strip() for name in ['memory.current','memory.peak','memory.events','memory.pressure','memory.swap.current'] if (group/name).exists()}
                cgpsi=float(cg.get('memory.pressure','some avg10=0').splitlines()[0].split()[1].split('=')[1])
                gpu=command(['nvidia-smi','--query-gpu=memory.used,temperature.gpu,clocks.current.graphics,clocks.current.memory,power.draw','--format=csv,noheader,nounits'],20)
                trip=Path('/run/gpu-thermal-trip').exists()
                row={'time':time.time(),'available_kib':available,'psi_avg10':psi,'gpu':gpu,'thermal_trip':trip,'cgroup':cg,'paused':paused}
                f.write(json.dumps(row)+'\n');f.flush()
                persistent=paused and time.monotonic()-pause_started>45
                if available<12582912 or cgpsi>5 or trip or persistent:
                    w.abort.set()
                    if paused:command(['docker','unpause',container],30,False);paused=False
                    command(['docker','stop','-t','10',container],30,False)
                    return
                if psi>5 and not paused:
                    command(['docker','pause',container],30)
                    paused=True;pause_started=time.monotonic()
                    with (ROOT/'pressure-pauses.jsonl').open('a') as events:events.write(json.dumps({'time':time.time(),'container':container,'event':'pause','global_psi_avg10':psi})+'\n')
                    print('RESOURCE PAUSE '+container,flush=True)
                elif paused and psi<1 and last_global<1:
                    command(['docker','unpause',container],30)
                    paused=False
                    with (ROOT/'pressure-pauses.jsonl').open('a') as events:events.write(json.dumps({'time':time.time(),'container':container,'event':'resume','duration_s':time.monotonic()-pause_started})+'\n')
                    print('RESOURCE RESUME '+container,flush=True)
                last_global=psi
    except Exception as exc:
        w.abort.set()
        with (ROOT/'monitor-errors.jsonl').open('a') as events:events.write(json.dumps({'time':time.time(),'error':str(exc)})+'\n')
        if paused:command(['docker','unpause',container],30,False)
        command(['docker','stop','-t','10',container],30,False)
    finally:
        if paused:command(['docker','unpause',container],30,False)
w.monitor=guarded_monitor

def stop():
    global thread
    if w.active:
        with (ROOT / (w.active + '-server.log')).open('w') as log:
            subprocess.run(['docker', 'logs', w.active], stdout=log, stderr=log, timeout=60)
        command(['docker', 'rm', '-f', w.active], 60, False)
        w.active = None
    w.stop_monitor.set()
    if thread:
        thread.join(timeout=30)
        thread = None

def launch(arm, pool, phase):
    global thread, sequence
    sequence += 1
    w.active = f"gsq-q4only-{arm['name']}-{phase}-{sequence}"
    args = ['docker','run','-d','--name',w.active,'--gpus','all','--network','host','--memory','32g','--memory-swap','32g','--cpus','8','--shm-size','8g',
            '-e','GGML_CUDA_NO_PINNED=1','-v',MODEL_DIR+':/models:ro',IMAGE,'--model','/models/'+MODEL,'--ctx-size',str(pool),'--parallel','2','--kv-unified','--kv-unified-per-slot','262144',
            '--no-context-shift','--fit','off','--n-gpu-layers','999','--flash-attn','on','--cache-type-k',arm['metadata']['kv'],'--cache-type-v',arm['metadata']['kv'],
            '--jinja','--host','127.0.0.1','--port','18185','--alias',arm['model'],'--threads','8','--threads-batch','8','--batch-size','2048','--ubatch-size','512',
            '--cache-ram','8192','--temp','1','--top-p','0.95','--top-k','20','--min-p','0','--reasoning-effort','medium']
    h.write_new(ROOT / (w.active+'-launch.json'),args)
    command(args)
    w.stop_monitor.clear(); w.abort.clear()
    thread = threading.Thread(target=w.monitor,args=(w.active,ROOT/(w.active+'-telemetry.jsonl')))
    thread.start()
    w.wait_ready()
    print(f"READY {arm['name']} pool={pool} phase={phase}",flush=True)

def checked_request(arm, body):
    if w.abort.is_set():
        raise RuntimeError('Resource guard aborted')
    result = h.request(arm,body,3600)
    result.pop('events',None)
    if w.abort.is_set() or result['status'] != 'ok':
        h.write_new(ROOT / f'infrastructure-failure-{time.time_ns()}.json',result)
        raise RuntimeError('Request/resource failure: '+result['status'])
    return result

def full(plan,arm):
    out=ROOT/(arm['name']+'-full'); out.mkdir()
    h.write_new(out/'manifest.json',{'plan':plan,'arm':arm['name'],'started_unix':time.time(),'timeout_s':3600})
    h.write_new(out/'warmup.json',checked_request(arm,{'messages':[{'role':'user','content':'Reply OK.'}],'max_tokens':32,'temperature':0,'seed':42,'stream':True,'stream_options':{'include_usage':True}}))
    with (out/'results.jsonl').open('x') as f:
        for index,job in enumerate(plan['jobs']):
            result=checked_request(arm,job['request'])
            score, detail=(None,'performance probe')
            if job['case']['phase']=='quality':
                score,detail=h.grade(job['case']['grader'],result,plan['suite']['code_image'])
            row={'id':job['id'],'plan_sha256':plan['sha256'],'request_sha256':job['request_sha256'],'arm':arm['name'],'score':score,'grade_detail':detail,'result':result}
            f.write(json.dumps(row)+'\n');f.flush()
            print(f"{arm['name']} full {index+1}/{len(plan['jobs'])} {job['id']} score={score}",flush=True)
    h.write_new(ROOT/(arm['name']+'-full-summary.json'),h.summarize(plan,out))

def bodies(lines,callers):
    result=[]; expected=[]
    for caller in range(callers):
        records=[f'Record {i:05d}: ordinary neutral text with no instructions.' for i in range(lines)]
        keys={place:f'{place.upper()}_{h.digest([lines,caller,place])[:16]}' for place in ('early','middle','latest')}
        for place,idx in [('early',3),('middle',lines//2),('latest',lines-3)]:
            records[idx]=f"The {place} project reference is {keys[place]}."
        text=f'Independent coding agent {caller}, ledger {lines}.\n'+'\n'.join(records)+'\nReturn only the early, middle, and latest project references, in that order, separated by commas and no spaces.'
        result.append({'messages':[{'role':'user','content':text}],'temperature':1,'top_p':.95,'top_k':20,'reasoning_effort':'low','max_tokens':1024,'seed':42,'stream':True,'stream_options':{'include_usage':True}})
        expected.append(','.join(keys.values()))
    return result,expected

def batch(arm,requests,expected,label,out):
    barrier=threading.Barrier(len(requests))
    def submit(item):
        idx,body=item
        barrier.wait(timeout=30)
        result=checked_request(arm,body)
        return {'caller':idx,'request_sha256':h.digest(body),'expected':expected[idx],'pass':result.get('content','').strip()==expected[idx] if expected[idx] is not None else None,'result':result}
    started=time.monotonic()
    with cf.ThreadPoolExecutor(max_workers=len(requests)) as pool:
        rows=list(pool.map(submit,enumerate(requests)))
    wall=time.monotonic()-started
    record={'start_unix':time.time()-wall,'end_unix':time.time(),'label':label,'batch_wall_s':wall,'aggregate_output_tps':sum(r['result'].get('usage',{}).get('completion_tokens',0) for r in rows)/wall,'rows':rows}
    out.write(json.dumps(record)+'\n');out.flush()
    print(f"{arm['name']} {label} wall={wall:.2f}s pass={[r['pass'] for r in rows]} prompt={[r['result'].get('usage',{}).get('prompt_tokens') for r in rows]}",flush=True)
    return rows

def long(arm,pool,lines,callers,turns):
    label=f'long-{callers}x{lines}'
    launch(arm,pool,label)
    try:
        requests,expected=bodies(lines,callers)
        h.write_new(ROOT/f'{arm["name"]}-{label}-manifest.json',{'initial_requests':requests,'expected':expected,'turns':turns,'description':'Early/middle/latest retrieval, then revised latest with alternating original/revised retrieval. No reference repeated in question.'})
        with (ROOT/f'{arm["name"]}-{label}.jsonl').open('x') as out:
            rows=batch(arm,requests,expected,'cold',out)
            # Use canonical prior answers in both arms so subsequent histories remain matched even after a quality failure.
            for body,answer in zip(requests,expected):
                body['messages'].append({'role':'assistant','content':answer})
            for turn in range(1,turns+1):
                targets=[]
                for caller,body in enumerate(requests):
                    original=expected[caller].split(',')
                    revision=f'REVISION_{h.digest([callers,lines,caller,turn])[:16]}'
                    target=original[0] if turn%2 else revision
                    body['messages'].append({'role':'user','content':f'Latest project reference is now {revision}. Return only '+('the original early project reference.' if turn%2 else 'the current latest project reference.')})
                    targets.append(target)
                rows=batch(arm,requests,targets,f'followup-{turn}',out)
                for body,target in zip(requests,targets):
                    body['messages'].append({'role':'assistant','content':target})
            for body in requests:
                body['max_tokens']=512
                body['messages'].append({'role':'user','content':'Write a detailed explanation of implementing a reliable task queue. Continue until the output allowance is exhausted.'})
            batch(arm,requests,[None]*callers,'cached-generation-512',out)
    finally:
        stop()

def main():
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,w.interrupted)
    command(['systemctl','is-active','--quiet','vram-thermal-guard.service'])
    command(['systemctl','is-active','--quiet','vram-fan-curve.service'])
    assert not Path('/run/gpu-thermal-trip').exists()
    assert not (ROOT/'state-before-q4.json').exists(),'Never overwrite a campaign'
    base=h.load_plan(SOURCE/'plan-full.json')
    command(['docker','image','inspect',base['suite']['code_image']])
    replicas=command(['kubectl','-n','apps','get','deployment','llama','-o','jsonpath={.spec.replicas}'])
    suspended=command(['kubectl','-n','flux-system','get','kustomization','apps','-o','jsonpath={.spec.suspend}'])=='true'
    template=command(['kubectl','-n','apps','get','deployment','llama','-o','jsonpath={.spec.template}'])
    h.write_new(ROOT/'state-before-q4.json',{'replicas':replicas,'flux_suspended':suspended,'template_sha256':h.digest(template)})
    arms=[{'name':'gsq_'+dtype,'model':'gsq-rco-27b','base_url':'http://127.0.0.1:18185/v1','metadata':{'kv':dtype,'image':IMAGE,'weight_sha256':'64b53b64c7aa39f20a7e54bd80582fe595b1d745624ee8a72e92508c0326d810','vision':False,'speculation':False,'cuda_no_pinned':True}} for dtype in ('q8_0','q4_0')]
    pool=320000
    try:
        if not suspended: command(['flux','suspend','kustomization','apps'])
        command(['kubectl','-n','apps','scale','deployment','llama','--replicas=0'])
        command(['kubectl','-n','apps','wait','--for=delete','pod','-l','app.kubernetes.io/name=llama','--timeout=10m'],660)
        for _ in range(60):
            if not command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']):break
            time.sleep(5)
        else:raise RuntimeError('Unowned GPU process remains')
        plan=h.load_plan(ROOT/'plan-full.json')
        arms=[a for a in plan['arms'] if a['name']=='gsq_q4_0'];pool=arms[0]['metadata']['shared_pool']
        for arm in arms:
            if not (ROOT/(arm['name']+'-full-summary.json')).exists():
                launch(arm,pool,'quality')
                try:full(plan,arm)
                finally:stop()
            launch(arm,pool,'concurrent-coding')
            try:q.run(w,arm,ROOT/(arm['name']+'-concurrent-quality.jsonl'))
            finally:stop()
            launch(arm,pool,'concurrency')
            try:
                w.CONCURRENCY_CASES=[(1,0,False),(2,0,False),(2,2400,False)]
                w.concurrency(arm)
            finally:stop()
            long(arm,pool,15700,1,2)
            pair_lines=min(9300,(pool-12000)//32)
            long(arm,pool,pair_lines,2,20)
        h.write_new(ROOT/'comparison-full.json',h.compare(plan,ROOT/'gsq_q8_0-full',ROOT/'gsq_q4_0-full'))
    finally:
        stop()
        failures=[]
        for args in ([['kubectl','-n','apps','scale','deployment','llama','--replicas='+replicas]]+([] if suspended else [['flux','resume','kustomization','apps']])):
            try:command(args)
            except Exception as exc:failures.append(str(exc))
        try:
            command(['kubectl','-n','apps','rollout','status','deployment/llama','--timeout=15m'],960)
            actual=command(['kubectl','-n','apps','get','deployment','llama','-o','jsonpath={.spec.template}'])
            if h.digest(actual)!=h.digest(template):failures.append('Pod template changed')
        except Exception as exc:failures.append(str(exc))
        h.write_new(ROOT/'state-restored-q4.json',{'restored':not failures,'failures':failures,'replicas':replicas,'flux_suspended':suspended})
        if failures:raise RuntimeError('Production restoration failure: '+str(failures))
if __name__=='__main__':main()

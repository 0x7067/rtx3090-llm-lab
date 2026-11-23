"""Explicitly launched CPU-only build and qualification; never deploys serving."""
import argparse
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import time
import urllib.request
import urllib.error

from buildlib import memory_preflight, run_logged

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'jobs'
TAG='127.0.0.1:5000/home-llm-strata:normal120-dflash-short-v1'
NAMESPACE='normal120-dflash-short-cpu'

def kubectl(*args,check=True,input=None):
    return subprocess.run(['kubectl',*args],input=input,text=True,capture_output=True,timeout=45,check=check)

def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2)+'\n')

def status(stage,**kw):
    save('status.json',dict(stage=stage,pid=os.getpid(),unix=time.time(),**kw))

def run():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'build-metadata.json').exists(), 'Do not reuse a published tag'
    request=urllib.request.Request('http://127.0.0.1:5000/v2/home-llm-strata/manifests/normal120-dflash-short-v1',method='HEAD',headers={'Accept':'application/vnd.oci.image.index.v1+json,application/vnd.oci.image.manifest.v1+json'})
    try:
        with urllib.request.urlopen(request,timeout=30):
            raise RuntimeError('Tag already exists; choose a new reviewed tag')
    except urllib.error.HTTPError as error:
        assert error.code==404, f'Registry preflight failed: {error.code}'
    memory_preflight(OUT,status)
    status('building',deadline_seconds=1800)
    run_logged(['docker','buildx','build','--builder','strata-local-registry',
        '--network=none','--resource=memory=8g','--resource=cpu-quota=200000','--progress=plain',
        '--metadata-file',str(OUT/'build-metadata.json'),'--output','type=image,name='+TAG+',push=true',str(ROOT)],
        OUT/'build.log',1800)
    meta=json.loads((OUT/'build-metadata.json').read_text())
    digest=meta['containerimage.digest'];image=TAG+'@'+digest
    status('checking-cpu',image=image)
    assert not kubectl('get','namespace',NAMESPACE,'--ignore-not-found','-o','name').stdout.strip()
    created=False
    try:
        kubectl('create','namespace',NAMESPACE);created=True
        policy={'apiVersion':'networking.k8s.io/v1','kind':'NetworkPolicy','metadata':{'name':'deny-all','namespace':NAMESPACE},'spec':{'podSelector':{},'policyTypes':['Ingress','Egress'],'ingress':[],'egress':[]}}
        kubectl('create','-f','-',input=json.dumps(policy))
        names=['apply_patch.py','check_cpu.py','check_image.py','provenance.json']
        names += ['upstream/'+p for p in json.loads((ROOT/'provenance.json').read_text())['sha256']]
        data={name.replace('/','__'):(ROOT/name).read_text() for name in names}
        config={'apiVersion':'v1','kind':'ConfigMap','metadata':{'name':'checks','namespace':NAMESPACE},'data':data}
        kubectl('create','-f','-',input=json.dumps(config))
        pod={'apiVersion':'v1','kind':'Pod','metadata':{'name':'check','namespace':NAMESPACE},'spec':{
            'restartPolicy':'Never','activeDeadlineSeconds':240,'automountServiceAccountToken':False,'enableServiceLinks':False,
            'securityContext':{'runAsNonRoot':True,'runAsUser':1000,'runAsGroup':1000,'fsGroup':1000},
            'containers':[{'name':'check','image':image,'imagePullPolicy':'IfNotPresent','command':['/app/venv/bin/python','/checks/check_image.py'],
            'env':[{'name':k,'value':v} for k,v in {'CUDA_VISIBLE_DEVICES':'','NVIDIA_VISIBLE_DEVICES':'void','HOME':'/tmp','PYTHONDONTWRITEBYTECODE':'1','VLLM_OFFLOAD_KEEP_SHM':'1'}.items()],
            'securityContext':{'readOnlyRootFilesystem':True,'allowPrivilegeEscalation':False,'capabilities':{'drop':['ALL']}},
            'resources':{'requests':{'cpu':'1','memory':'1Gi'},'limits':{'cpu':'2','memory':'8Gi'}},
            'volumeMounts':[{'name':'tmp','mountPath':'/tmp'},{'name':'checks','mountPath':'/checks','readOnly':True}]}],
            'volumes':[{'name':'tmp','emptyDir':{'medium':'Memory','sizeLimit':'256Mi'}},{'name':'checks','configMap':{'name':'checks','items':[{'key':name.replace('/','__'),'path':name} for name in names]}}]}}
        save('pod.json',pod)
        kubectl('create','-f','-',input=json.dumps(pod))
        deadline=time.monotonic()+300
        while True:
            result=json.loads(kubectl('get','pod','check','-n',NAMESPACE,'-o','json').stdout)
            if result.get('status',{}).get('phase') in ('Succeeded','Failed'):break
            assert time.monotonic()<deadline,'CPU pod deadline exceeded'
            time.sleep(5)
        save('pod-result.json',result)
        log=kubectl('logs','check','-n',NAMESPACE).stdout;(OUT/'cpu-check.log').write_text(log)
        container=result['status']['containerStatuses'][0]
        assert container['state']['terminated']['exitCode']==0, 'CPU qualification failed'
        request=urllib.request.Request('http://127.0.0.1:5000/v2/home-llm-strata/manifests/'+digest,headers={'Accept':'application/vnd.oci.image.index.v1+json,application/vnd.oci.image.manifest.v1+json'})
        with urllib.request.urlopen(request,timeout=30) as response:manifest=response.read()
        assert 'sha256:'+hashlib.sha256(manifest).hexdigest()==digest
        (OUT/'registry-manifest.json').write_bytes(manifest)
        accepted={digest}|{m['digest'] for m in json.loads(manifest).get('manifests',[]) if m.get('platform',{}).get('architecture')=='amd64' and m.get('platform',{}).get('os')=='linux'}
        assert any(container['imageID'].endswith(d) for d in accepted)
        receipt=json.loads(log.split('NORMAL120_CPU_RECEIPT\n',1)[1])
        assert receipt['cpu_checks']=='passed' and receipt['gpu_execution'] is False
        receipt.update(image_id=image,runtime_image_id=container['imageID'],exit_code=0,
            check_sources_sha256={name:hashlib.sha256(data[name.replace('/','__')].encode()).hexdigest() for name in names})
        save('cpu-receipt.json',receipt)
    finally:
        if created:
            kubectl('delete','namespace',NAMESPACE,'--wait=false')
            deadline=time.monotonic()+120
            while kubectl('get','namespace',NAMESPACE,'--ignore-not-found','-o','name').stdout.strip():
                assert time.monotonic()<deadline,'CPU namespace cleanup incomplete'
                time.sleep(2)
    status('complete',image=image,deployed=False,gpu_execution=False,cleanup_complete=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',required=True,help='Explicit build authorization required')
    parser.parse_args()
    def interrupted(signum, frame):
        raise RuntimeError(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:run()
    except Exception as error:
        OUT.mkdir(exist_ok=True);status('failed',error=f'{type(error).__name__}: {error}')
        raise

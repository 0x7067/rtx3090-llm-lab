"""Authenticated production and native-client verification, without changing defaults."""
import concurrent.futures as cf
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import urllib.error
import urllib.request
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('h',ROOT.parent/'paired-harness/harness.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
TOKEN=(Path.home()/'.pi/agent/config/llama-swap-api-key').read_text().strip()
os.environ['GSQ_VERIFY_TOKEN']=TOKEN
BASE='http://100.64.0.2/v1'
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):return None
OPENER=urllib.request.build_opener(NoRedirect())
def request(path,body=None,auth=True,base=BASE):
    headers={'Content-Type':'application/json','User-Agent':'home-llm-benchmark-verification/1.0'}
    if auth:headers['Authorization']='Bearer '+TOKEN
    try:
        with OPENER.open(urllib.request.Request(base+path,headers=headers,data=json.dumps(body).encode() if body else None),timeout=900) as response:return response.status,json.load(response)
    except urllib.error.HTTPError as exc:return exc.code,None

def client(item):
    name,folder,provider,marker=item
    settings=json.loads((Path.home()/folder/'agent/settings.json').read_text())
    args=[name,'--offline','--print','--mode','json','--no-session','--no-tools','--no-extensions','--extension',str(Path.home()/folder/'agent/extensions/home-models.ts'),'--no-skills','--no-prompt-templates','--no-context-files','--system-prompt','Follow the user request exactly.','--provider',provider,'--model','gsq-rco-27b','--thinking','medium','Reply exactly '+marker+'.']
    result=subprocess.run(args,cwd='/tmp',capture_output=True,text=True,timeout=900)
    (ROOT/(name+'-gsq.jsonl')).write_text(result.stdout)
    (ROOT/(name+'-gsq.stderr')).write_text(result.stderr)
    final={}
    for line in result.stdout.splitlines():
        try:event=json.loads(line)
        except json.JSONDecodeError:continue
        message=event.get('message',{})
        if event.get('type')=='message_end' and message.get('role')=='assistant':final=message
    answer=''.join(p.get('text','') for p in final.get('content',[]) if p.get('type')=='text').strip()
    after=json.loads((Path.home()/folder/'agent/settings.json').read_text())
    defaults_unchanged=all(after.get(k)==settings.get(k) for k in ['defaultModel','defaultProvider','defaultThinkingLevel'])
    return name,{'defaults_unchanged':defaults_unchanged,'returncode':result.returncode,'model':final.get('model'),'provider':final.get('provider'),'answer':answer,'default_model':settings['defaultModel'],'default_provider':settings.get('defaultProvider'),'default_thinking':settings.get('defaultThinkingLevel'),'passed':defaults_unchanged and result.returncode==0 and final.get('model')=='gsq-rco-27b' and final.get('provider')==provider and answer==marker}

def main():
    report={}
    unauthorized,_=request('/models',auth=False)
    status,catalog=request('/models')
    models={m['id']:m for m in (catalog or {}).get('data',[])}
    report['catalog']={'unauthenticated_status':unauthorized,'authenticated_status':status,'models':models,'passed':unauthorized==401 and status==200 and all(m in models for m in ['gsq-rco-27b','orca-saq2-27b','humanlike-2.0'])}
    unauthorized,_=request('/models',auth=False,base='https://llama.denguinho.org/v1')
    status,public=request('/models',base='https://llama.denguinho.org/v1')
    report['public_catalog']={'unauthenticated_status':unauthorized,'authenticated_status':status,'passed':unauthorized==401 and status==200 and 'gsq-rco-27b' in [m['id'] for m in (public or {}).get('data',[])]}
    for model,marker in [('humanlike-2.0','HUMANLIKE_READY'),('gsq-rco-27b','GSQ_READY')]:
        arm={'name':model,'model':model,'base_url':BASE,'api_key_env':'GSQ_VERIFY_TOKEN'}
        result=h.request(arm,{'messages':[{'role':'user','content':'Reply exactly '+marker+'.'}],'temperature':0,'reasoning_effort':'medium','max_tokens':1024,'stream':True,'stream_options':{'include_usage':True}},900)
        result.pop('events',None)
        report[model]={'result':result,'passed':result['status']=='ok' and result.get('content','').strip()==marker}
    status,props=request('/upstream/gsq-rco-27b/props',base=BASE.removesuffix('/v1'))
    runtime={k:(props or {}).get(k) for k in ['total_slots','model_alias','build_info']}
    runtime['n_ctx']=(props or {}).get('default_generation_settings',{}).get('n_ctx')
    runtime['status']=status
    runtime['passed']=status==200 and runtime['total_slots']==2 and runtime['n_ctx']==262144 and runtime['build_info']=='b11429-d81235049'
    report['runtime']=runtime
    messages=[{'role':'user','content':'Call report_ready with tag GSQ_PRODUCTION_READY. Do not answer in prose.'}]
    tools=[{'type':'function','function':{'name':'report_ready','description':'Report readiness','parameters':{'type':'object','properties':{'tag':{'type':'string'}},'required':['tag']}}}]
    status,response=request('/chat/completions',{'model':'gsq-rco-27b','temperature':0,'reasoning_effort':'medium','max_tokens':1024,'messages':messages,'tools':tools,'tool_choice':'auto'})
    message=(response or {}).get('choices',[{}])[0].get('message',{})
    calls=message.get('tool_calls',[])
    correct=len(calls)==1 and calls[0]['function']['name']=='report_ready' and json.loads(calls[0]['function']['arguments'])=={'tag':'GSQ_PRODUCTION_READY'}
    report['tool_call']={'status':status,'message':message,'passed':status==200 and correct}
    if correct:
        messages += [message,{'role':'tool','tool_call_id':calls[0]['id'],'content':'Readiness accepted. Reply exactly GSQ_TOOL_ROUNDTRIP_READY.'}]
        status,response=request('/chat/completions',{'model':'gsq-rco-27b','temperature':0,'reasoning_effort':'medium','max_tokens':1024,'messages':messages,'tools':tools})
        content=(response or {}).get('choices',[{}])[0].get('message',{}).get('content','')
        report['tool_followup']={'status':status,'content':content,'passed':status==200 and content.strip()=='GSQ_TOOL_ROUNDTRIP_READY'}
    with cf.ThreadPoolExecutor(max_workers=2) as pool:
        for name,evidence in pool.map(client,[('pi','.pi','llama-swap','GSQ_PI_READY'),('prime-agent','.prime','home-qwen','GSQ_PRIME_READY')]):report[name]=evidence
    status,response=request('/chat/completions',{'model':'orca-saq2-27b','temperature':0,'reasoning_effort':'low','max_tokens':256,'messages':[{'role':'user','content':'Reply exactly ORCA_DEFAULT_READY.'}]})
    content=(response or {}).get('choices',[{}])[0].get('message',{}).get('content','')
    report['orca_default']={'status':status,'content':content,'passed':status==200 and content.strip()=='ORCA_DEFAULT_READY'}
    report['passed']=all(item['passed'] for item in report.values())
    (ROOT/'production-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:({'passed':value['passed']} if isinstance(value,dict) else value) for key,value in report.items()}))
    assert report['passed'],'Production verification failed'
if __name__=='__main__':main()

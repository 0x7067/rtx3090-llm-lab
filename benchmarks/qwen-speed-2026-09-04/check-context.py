#!/usr/bin/env python3
"""Test exact token capacity and retrieval from the beginning of a long prompt."""
import json
from pathlib import Path
import sys
import time
import urllib.request
ROOT=Path(__file__).resolve().parent
label=sys.argv[1]
base='http://127.0.0.1:18089'
def post(path,data):
    req=urllib.request.Request(base+path,json.dumps(data).encode(),headers={
        'Content-Type':'application/json','User-Agent':'OpenAI File Downloader, XaiImageApiFetch/1.0'})
    with urllib.request.urlopen(req,timeout=600) as r:return json.load(r)
def tokens(text):return post('/tokenize',{'content':text,'parse_special':True})['tokens']
stress=len(sys.argv)>2 and sys.argv[2]=='stress'
parallel=stress or (len(sys.argv)>2 and sys.argv[2]=='parallel')
secret='Q38_CTX_START_7F3A'
prompt=post('/apply-template',{'messages':[{'role':'user','content':
    'Remember the reference code '+secret+'. Read the background below, then return only the reference code.\nFILLER_LOCATION\nReturn the reference code from the first line, exactly.'}],
    'reasoning_effort':'medium'})['prompt']
a,b=prompt.split('FILLER_LOCATION')
prefix,suffix=tokens(a),tokens(b)
import importlib.util
spec=importlib.util.spec_from_file_location('trial',ROOT/'bench.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
filler=tokens(m.bench.build_preamble(10000))
requested=70000 if parallel else 125000
n=requested-len(prefix)-len(suffix)
ids=prefix+(filler*((n+len(filler)-1)//len(filler)))[:n]+suffix
assert len(ids)==requested
def evaluate(ids, expected):
    start=time.monotonic()
    try:
        r=post('/completion',{'prompt':ids,'n_predict':8192 if stress else 512,'ignore_eos':stress,'temperature':0,'cache_prompt':False,'return_tokens':True})
        return {'tag':label,'input_tokens':len(ids),'elapsed_s':time.monotonic()-start,
                'content':None if stress else r.get('content'),'timings':r.get('timings'),'truncated':r.get('truncated'),
                'tokens_evaluated':r.get('tokens_evaluated'),
                'passed':r.get('timings',{}).get('predicted_n',0)>=8192 if stress else r.get('content','').rsplit('</think>',1)[-1].strip()==expected}
    except Exception as e:
        return {'tag':label,'input_tokens':len(ids),'elapsed_s':time.monotonic()-start,'passed':False,'error':str(e)}
if parallel:
    from concurrent.futures import ThreadPoolExecutor
    other='Q38_CTX_OTHER_9B2C'
    other_prefix=tokens(a.replace(secret,other))
    m=requested-len(other_prefix)-len(suffix)
    other_ids=other_prefix+(filler*((m+len(filler)-1)//len(filler)))[:m]+suffix
    with ThreadPoolExecutor(2) as ex:
        futures=[ex.submit(evaluate,ids,secret),ex.submit(evaluate,other_ids,other)]
        records=[f.result() for f in futures]
else:
    records=[evaluate(ids,secret)]
name=label+('-context-stress.json' if stress else '-context-overlap.json' if parallel else '-context.json')
(ROOT/name).write_text(json.dumps(records,indent=2)+'\n')
for record in records:print(json.dumps(record),flush=True)
assert all(r['passed'] and not r.get('truncated') for r in records)

import argparse
import concurrent.futures
import json
import time
import urllib.request
import uuid

parser = argparse.ArgumentParser(description="Qualify Qwen Pi prefix reuse, tools and long-context retrieval.")
parser.add_argument("--port", type=int, default=5894)
parser.add_argument("--context", type=int, choices=(131072, 172032), default=131072)
parser.add_argument("--callers", type=int, choices=(4, 8), default=4)
args = parser.parse_args()
BASE = f"http://127.0.0.1:{args.port}"
result={'checks':[]}
def call(name,messages,expected=None,tools=None,temperature=0.8):
    payload={'model':'qwen3.8-pi','messages':messages,'temperature':temperature,'top_p':1.0,'seed':123,'max_tokens':64}
    if tools:payload.update(tools=tools,tool_choice='auto')
    request=urllib.request.Request(BASE+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    start=time.monotonic()
    with urllib.request.urlopen(request,timeout=300) as response:data=json.load(response)
    message=data['choices'][0]['message']
    if expected is not None:assert (message.get('content') or '').strip()==expected,(name,message)
    check={'name':name,'seconds':time.monotonic()-start,'usage':data['usage'],'cached_tokens':data['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0),'passed':True}
    result['checks'].append(check);print(name,'PASS',round(check['seconds'],3),'cached',check['cached_tokens'],flush=True)
    return message
parents=[]
for family, value in enumerate(('CONTEXT_A_162','CONTEXT_B_749')):
    source='Run '+uuid.uuid4().hex+'\nReference value: '+value+'\n'+'\n'.join(f'record_{i:04d} = {{"value": {i*7%997}, "enabled": {str(i%3==0).lower()}}}' for i in range(1500))
    parent=[{'role':'user','content':source+'\nAcknowledge this document by replying exactly READY.'}]
    call('shared_parent_prime_'+str(family),parent,'READY')
    parents.append((parent+[{'role':'assistant','content':'READY'}],value))
with concurrent.futures.ThreadPoolExecutor(max_workers=args.callers) as pool:
    futures=[]
    for i in range(args.callers):
        history,value=parents[i%2]
        marker=f'FORK_{i}_CORRECT_839'
        expected=value+':'+marker
        query='Look up the Reference value from the document. Reply with that value, a colon, then '+marker+'. Do not add quotes, spaces, markdown, or commentary.'
        futures.append(pool.submit(call,'concurrent_branch_'+str(i),history+[{'role':'user','content':query}],expected))
    for future in futures:future.result()
checks=[c for c in result['checks'] if c['name'].startswith('concurrent_branch_')]
assert len(checks)==args.callers and all(c['cached_tokens']>28000 for c in checks),checks

tools=[{'type':'function','function':{'name':'read_probe','description':'Read a probe value','parameters':{'type':'object','properties':{'key':{'type':'string'}},'required':['key'],'additionalProperties':False}}}]
messages=[{'role':'user','content':'Call read_probe with key streaming-cache-check. After it returns, reply with its exact value only, without formatting or commentary.'}]
payload={'model':'qwen3.8-pi','messages':messages,'tools':tools,'tool_choice':'auto','temperature':0.8,'top_p':1.0,'seed':123,'max_tokens':128,'stream':True}
request=urllib.request.Request(BASE+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
assembled={};done=False
with urllib.request.urlopen(request,timeout=30) as response:
    for line in response:
        if not line.startswith(b'data: '):continue
        raw=line[6:].strip()
        if raw==b'[DONE]':done=True;break
        chunk=json.loads(raw)
        for choice in chunk.get('choices',[]):
            for piece in choice.get('delta',{}).get('tool_calls',[]):
                item=assembled.setdefault(piece['index'],{'id':'','name':'','arguments':''})
                if piece.get('id'):item['id']=piece['id']
                fn=piece.get('function',{})
                item['name']+=fn.get('name','');item['arguments']+=fn.get('arguments','')
assert done and len(assembled)==1,assembled
item=assembled[0];assert item['name']=='read_probe' and json.loads(item['arguments'])=={'key':'streaming-cache-check'},item
call('streamed_tool_result',messages+[{'role':'assistant','content':None,'tool_calls':[{'id':item['id'],'type':'function','function':{'name':item['name'],'arguments':item['arguments']}}]},{'role':'tool','tool_call_id':item['id'],'content':'STREAM_TOOL_472_OK'}],'STREAM_TOOL_472_OK',tools)
result['streamed_tool_call_passed']=True

def count_tokens(messages):
    payload = {"model": "qwen3.8-pi", "messages": messages}
    request = urllib.request.Request(
        BASE + "/tokenize", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)["count"]

if args.context == 131072:
    full='Run '+uuid.uuid4().hex+'\nRecovery identifier: FULL_WINDOW_8741\n'+'\n'.join(f'record_{i:04d} = {{"value": {i*7%997}, "enabled": {str(i%3==0).lower()}}}' for i in range(5750))
    full_messages=[{'role':'user','content':full+'\nWhat is the recovery identifier? Reply with only that identifier.'}]
    call('full_context_cold',full_messages,'FULL_WINDOW_8741')
    call('full_context_repeat',full_messages,'FULL_WINDOW_8741')
    assert result['checks'][-1]['usage']['prompt_tokens']>119000
    assert result['checks'][-1]['usage']['prompt_tokens']<131072
    assert result['checks'][-1]['cached_tokens']>119000
else:
    expected = uuid.uuid4().hex[:10]
    header = f"This is an archive of test logs. The release authorization code is {expected}.\n"
    paragraph = "The fixture records queue depth and memory limits for every worker in the cluster.\n"
    question = "\nReturn the release authorization code from the start of the archive. Output just the code."

    def archive(paragraphs, suffix=""):
        return [{"role": "user", "content": header + paragraph * paragraphs + question + suffix}]

    target = args.context - 256
    base = count_tokens(archive(0))
    tokens_per_paragraph = (count_tokens(archive(100)) - base) / 100
    paragraphs = int((target - base) / tokens_per_paragraph)
    count = count_tokens(archive(paragraphs))
    assert target - 128 <= count <= args.context - 128, count

    call("full_context_cold", archive(paragraphs), expected, temperature=0)
    call("full_context_repeat", archive(paragraphs, "\nUse lowercase."), expected, temperature=0)
    assert result["checks"][-2]["usage"]["prompt_tokens"] >= target - 128
    assert result["checks"][-1]["usage"]["prompt_tokens"] >= target - 128
    assert result["checks"][-1]["cached_tokens"] >= 0.95 * count

result['passed']=True
print('RESULT_JSON '+json.dumps(result),flush=True)

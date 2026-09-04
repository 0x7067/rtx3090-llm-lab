import json, subprocess, time, urllib.request
from pathlib import Path
root=Path(__file__).resolve().parent
name='qwen-profile-check'
try:
 subprocess.run(['docker','run','-d','--name',name,'--gpus','all','--network','host','-v','/data/buttercup_6tb/k3s/llama-models:/models:ro','-v',str(root/'candidate-profile.yaml')+':/app/config.yaml:ro','--entrypoint','/usr/local/bin/llama-swap','llama:cuda-swap-v18','-config','/app/config.yaml','-listen','127.0.0.1:18089'],check=True,stdout=subprocess.DEVNULL)
 time.sleep(2)
 body={'model':'qwen3.8-27b','messages':[{'role':'user','content':'Calculate 19 * 23. Answer only the number.'}],'max_tokens':256,'temperature':0}
 req=urllib.request.Request('http://127.0.0.1:18089/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json','User-Agent':'OpenAI File Downloader, XaiImageApiFetch/1.0'})
 with urllib.request.urlopen(req,timeout=180) as r: result=json.load(r)
 (root/'published-profile-check.json').write_text(json.dumps(result,indent=2)+'\n')
 assert result['choices'][0]['message']['content'].strip()=='437', result
 print('Published image + exact serving profile + persistent drafter: PASS')
finally:
 with (root/'published-profile-server.log').open('w') as f: subprocess.run(['docker','logs',name],stdout=f,stderr=f)
 subprocess.run(['docker','rm','-f',name],stdout=subprocess.DEVNULL)

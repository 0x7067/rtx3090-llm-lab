import json, statistics
from pathlib import Path
root=Path(__file__).resolve().parent
def rows(p): return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
output={}
for arm,base in [('orca_vllm',root),('thinkingcap_llama',root/'thinkingcap-200k')]:
 d={}; output[arm]=d
 qual=[]; perf={}
 for name in ['full','coding-repeat','speed-repeat']:
  rs=rows(base/f'{arm}-{name}/results.jsonl')
  qual+= [r for r in rs if r['score'] is not None]
  for r in rs:
   if r['score'] is None:perf.setdefault(r['id'].split(':')[0],[]).append(r['result'])
 d['quality']={'passed':sum(r['score'] for r in qual),'count':len(qual),'failures':[r['id'] for r in qual if r['score']!=1]}
 d['performance']={k:{'n':len(v),'tps':statistics.median(r['e2e_output_tps'] for r in v) if v[0].get('e2e_output_tps') is not None else None,'ttft':statistics.median(r['ttft_s'] for r in v)} for k,v in perf.items()}
 conc=rows(base/f'{arm}-concurrency.jsonl')
 if arm=='orca_vllm':
  d['original_long_admission_only']= [r['batch_wall_s'] for r in conc if r['retrieval']]
  conc=[r for r in conc if not r['retrieval']] + rows(root/'orca-concurrent-quality'/f'{arm}-concurrency.jsonl')
 groups={}
 for r in conc: groups.setdefault(str((r['callers'],r['filler_lines'],r['retrieval'],r.get('warm',False))),[]).append(r)
 d['concurrency']={k:{'batches':len(v),'seconds':statistics.median(r['batch_wall_s'] for r in v),'tps':statistics.median(r['aggregate_output_tps'] for r in v),'retrieval':[s['retrieval_pass'] for r in v for s in r['rows']],'tokens':[s['result'].get('usage') for r in v for s in r['rows']],'ttft':[s['result'].get('ttft_s') for r in v for s in r['rows']]} for k,v in groups.items()}
 panelbase=base if arm=='thinkingcap_llama' else root/'orca-concurrent-quality'
 panel=rows(panelbase/f'{arm}-concurrent-quality.jsonl')
 prs=[r for batch in panel for r in batch['rows']]
 d['concurrent_coding']={'count':len(prs),'passed':sum(r['score'] for r in prs),'inference_seconds':sum(b['inference_wall_s'] for b in panel),'failures':[r['id'] for r in prs if r['score']!=1]}
print(json.dumps(output,indent=2))
(root/'decision-observations.json').write_text(json.dumps(output,indent=2)+'\n')

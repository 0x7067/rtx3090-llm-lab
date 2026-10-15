"""Summarize raw inference, executable grades and resource observations."""
import json
from pathlib import Path
import statistics
ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
rows=lambda p:[json.loads(s) for s in p.read_text().splitlines()]
result={'scope':'Matched Q8/Q4 within GSQ; historical Orca observations are a separate campaign. No statistical quality-equivalence claim.','arms':{}}
for arm in ('gsq_q8_0','gsq_q4_0'):
    report={}
    summary=ROOT/f'{arm}-full-summary.json'
    if summary.exists():
        report['full']=read(summary)
        report['full_quality_passed']=sum(v['full_score'] for v in report['full']['quality'].values())
        report['full_quality_attempted']=sum(v['attempted'] for v in report['full']['quality'].values())
    path=ROOT/f'{arm}-concurrent-quality.jsonl'
    if path.exists():
        items=rows(path); grades=[r for batch in items for r in batch['rows']]
        report['concurrent_coding']={'passed':sum(r['score']==1 for r in grades),'attempted':len(grades),'failures':[{'id':r['id'],'detail':r['detail']} for r in grades if r['score']<1],'inference_wall_s':sum(b['inference_wall_s'] for b in items)}
    path=ROOT/f'{arm}-concurrency.jsonl'
    resumed=ROOT.parent/'gsq-q4-resume-2026-10-08'
    if arm=='gsq_q4_0' and (resumed/path.name).exists():path=resumed/path.name
    if path.exists():
        groups={}
        interrupted=[r for r in rows(path) if any(x['result']['status']!='ok' for x in r['rows'])]
        report['interrupted_concurrency_runs']=len(interrupted)
        for row in rows(path):
            if any(x['result']['status']!='ok' for x in row['rows']):continue
            groups.setdefault(f"{row['callers']}callers-{row['filler_lines']}lines",[]).append(row)
        report['concurrency']={key:{'runs':len(items),'median_wall_s':statistics.median(r['batch_wall_s'] for r in items),'median_aggregate_output_tps':statistics.median(r['aggregate_output_tps'] for r in items),'output_token_counts':[[item['result'].get('usage',{}).get('completion_tokens') for item in r['rows']] for r in items]} for key,items in groups.items()}
    report['long_context']={}
    long_root=ROOT.parent/'gsq-q4-resume-2026-10-08' if arm=='gsq_q4_0' else ROOT
    for path in sorted(long_root.glob(f'{arm}-long-*.jsonl')):
        items=rows(path)
        if not items:continue
        checks=[r for b in items for r in b['rows'] if r['pass'] is not None]; warm=[b['batch_wall_s'] for b in items if b['label'].startswith('followup-')]
        report['long_context'][path.stem]={'checks_passed':sum(r['pass'] for r in checks),'checks_attempted':len(checks),'max_prompt_tokens':max((r['result'].get('usage',{}).get('prompt_tokens',0) for b in items for r in b['rows']),default=0),'cold_wall_s':items[0]['batch_wall_s'],'median_followup_wall_s':statistics.median(warm) if warm else None,'generation':items[-1] if items[-1]['label']=='cached-generation-512' else None}
    telemetry_roots=[ROOT]+([ROOT.parent/'gsq-q4-resume-2026-10-08'] if arm=='gsq_q4_0' else [])
    telemetry=[r for directory in telemetry_roots for p in directory.glob(f'*{arm}-*-telemetry.jsonl') for r in rows(p) if 'gpu' in r]
    if telemetry:report['resources']={'peak_gpu_mib':max(int(r['gpu'].split(',')[0]) for r in telemetry),'peak_gpu_celsius':max(int(r['gpu'].split(',')[1]) for r in telemetry),'minimum_host_available_kib':min(r['available_kib'] for r in telemetry),'peak_host_psi_avg10':max(r['psi_avg10'] for r in telemetry)}
    result['arms'][arm]=report
restored=ROOT.parent/'gsq-q4-resume-2026-10-08'/'state-restored-q4.json'
if not restored.exists():restored=ROOT/'state-restored-q4.json'
if not restored.exists():restored=ROOT/'state-restored-resume.json'
if not restored.exists():restored=ROOT/'state-restored.json'
if restored.exists():result['restoration']=read(restored)
result['pressure_pauses']=rows(ROOT/'pressure-pauses.jsonl') if (ROOT/'pressure-pauses.jsonl').exists() else []
(ROOT/'qualification-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({arm:{k:value[k] for k in ['full_quality_passed','full_quality_attempted','concurrent_coding','resources'] if k in value} for arm,value in result['arms'].items()},indent=2))

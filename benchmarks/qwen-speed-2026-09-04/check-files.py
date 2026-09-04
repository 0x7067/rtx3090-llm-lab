#!/usr/bin/env python3
"""Exercise generated code inside an isolated, disposable Python container."""
import json
from pathlib import Path
import subprocess
import sys

CHECK = r'''
import json, re, sys, types
from pathlib import Path
rows=json.load(sys.stdin)
for i,row in enumerate(rows,1):
    assert row['finish_reason']=='stop', f'turn {i}: incomplete'
    code=re.search(r'```python\s*\n(.*?)```',row['content'],re.S).group(1)
    mod=types.ModuleType('generated')
    sys.modules['generated']=mod
    exec(compile(code,'generated.py','exec'), mod.__dict__)
    store=mod.SqliteTaskStore(':memory:')
    store.upsert(mod.Task('a','first',2))
    store.upsert(mod.Task('b','second',9))
    assert store.get('a').title=='first'
    assert store.pending_count()==2
    assert store.mark_status('a','done') is True
    assert store.pending_count()==1
    assert store.mark_status('missing','done') is False
    if i>=2:
        assert store.delete('missing') is False
        assert store.delete('a') is True
        assert store.get('a') is None
    if i>=5:
        assert [t.task_id for t in store.high_priority(9)]==[]
        assert [t.task_id for t in store.high_priority(8)]==['b']
    if i>=6:
        out=Path('/tmp/export.json')
        mod.export_json(store,out)
        data=json.loads(out.read_text())
        assert data['pending_count']==1 and len(data['tasks'])==1
    if i>=8:
        assert store.requeue('missing') is False
        store.mark_status('b','done')
        old=store.get('b').updated_at
        assert store.requeue('b') is True
        assert store.get('b').status=='pending'
        assert store.get('b').updated_at>=old
    print(f'turn {i}: PASS')
'''
for filename in sys.argv[1:]:
    rows=[json.loads(x) for x in Path(filename).read_text().splitlines()]
    r=subprocess.run(['docker','run','--rm','-i','--network','none','--read-only',
                      '--cap-drop','ALL','--security-opt','no-new-privileges',
                      '--memory','256m','--pids-limit','64','--user','1000:1000',
                      '--tmpfs','/tmp:rw,noexec,nosuid,size=16m','python:3.12-slim',
                      'python','-c',CHECK],input=json.dumps(rows),text=True,
                      capture_output=True,timeout=30)
    print(filename, r.stdout, r.stderr,sep='\n')
    if r.returncode:
        raise SystemExit(r.returncode)

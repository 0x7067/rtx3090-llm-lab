"""Executable coding under two-caller load; keep inference and grading separate."""
import concurrent.futures
import copy
import json
from pathlib import Path
import threading
import time


def run(window, arm, output):
    source = Path(__file__).resolve().parent
    plan = window.h.load_plan(source / 'plan-full.json')
    jobs = [copy.deepcopy(j) for j in plan['jobs'] if j['case']['tier'] == 'coding']
    for case_id in ('code-sse-frames', 'code-histogram'):
        original = next(j for j in jobs if j['case']['id'] == case_id)
        for seed in (314, 2718):
            item = copy.deepcopy(original)
            item['id'] = f'{case_id}:concurrent:{seed}'
            item['request']['seed'] = seed
            jobs.append(item)
    for job in jobs:
        if job['case']['id'] == 'code-sse-frames':
            job['request']['messages'][-1]['content'] += ' Return a list of strings, not an iterator or generator.'
        job['request']['messages'][0]['content'] = 'Concurrent coding evaluation: ' + job['id']
        job['request_sha256'] = window.h.digest(job['request'])
    output.parent.mkdir(exist_ok=True)
    window.h.write_new(output.with_suffix('.manifest.json'), {'arm': arm, 'source_plan_sha256': plan['sha256'], 'jobs': jobs, 'callers': 2,
        'contract_change': 'Explicitly require a list return for SSE frames. New panel; historical strict scores unchanged.'})
    with output.open('x') as f:
        for index in range(0, len(jobs), 2):
            batch = jobs[index:index + 2]
            barrier = threading.Barrier(len(batch))
            def submit(job):
                barrier.wait(timeout=30)
                result = window.h.request(arm, job['request'], 1800)
                result.pop('events', None)
                return result
            start = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(submit, batch))
            wall = time.monotonic() - start
            rows = []
            for job, result in zip(batch, results):
                score, detail = window.h.grade(job['case']['grader'], result, plan['suite']['code_image'])
                rows.append({'id': job['id'], 'request_sha256': job['request_sha256'], 'score': score, 'detail': detail, 'result': result})
            f.write(json.dumps({'batch': index // 2, 'inference_wall_s': wall, 'rows': rows}) + '\n'); f.flush()
            print(f"{arm['name']} concurrent coding {index + len(batch)}/{len(jobs)} score={sum(r['score'] for r in rows)}/{len(rows)}", flush=True)
            if window.abort.is_set() or any(r['result']['status'] != 'ok' for r in rows):
                raise RuntimeError('Concurrent coding infrastructure failure')

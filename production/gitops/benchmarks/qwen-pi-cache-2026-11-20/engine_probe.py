import concurrent.futures
import json
import time
import urllib.request
import uuid
import sys

BASE = 'http://127.0.0.1:5894'
MODEL = 'qwen3.8-pi'
result = {'profile': sys.argv[1], 'checks': []}

def post(path, payload, timeout=150):
    request = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)

def call(name, messages, cap=64, expected=None, tools=None):
    payload = {'model': MODEL, 'messages': messages, 'temperature': 0.8, 'top_p': 1.0, 'seed': 123, 'max_tokens': cap}
    if tools:
        payload['tools'] = tools
        payload['tool_choice'] = 'auto'
    start = time.monotonic()
    data = post('/v1/chat/completions', payload)
    elapsed = time.monotonic() - start
    message = data['choices'][0]['message']
    if expected is not None:
        assert expected in (message.get('content') or ''), (name, message)
    item = {'name': name, 'seconds': elapsed, 'usage': data['usage'], 'finish_reason': data['choices'][0]['finish_reason']}
    item['cached_tokens'] = data['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0)
    result['checks'].append(item)
    print(name, 'PASS', round(elapsed, 3), 's', 'cached', item['cached_tokens'], 'of', data['usage']['prompt_tokens'], flush=True)
    return message

def streamed(name, messages):
    payload = {'model': MODEL, 'messages': messages, 'temperature': 0.8, 'top_p': 1.0, 'seed': 123, 'max_tokens': 256, 'stream': True, 'stream_options': {'include_usage': True}}
    request = urllib.request.Request(BASE + '/v1/chat/completions', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    start = time.monotonic()
    first = None
    usage = None
    done = False
    with urllib.request.urlopen(request, timeout=150) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            raw = line[6:].strip()
            if raw == b'[DONE]':
                done = True
                break
            chunk = json.loads(raw)
            if chunk.get('usage'):
                usage = chunk['usage']
            if first is None and any(c.get('delta', {}).get('content') for c in chunk.get('choices', [])):
                first = time.monotonic()
    elapsed = time.monotonic() - start
    assert done and first and usage and usage['completion_tokens'] > 0
    item = {'name': name, 'seconds': elapsed, 'ttft_seconds': first-start, 'usage': usage, 'output_tokens_per_second': usage['completion_tokens']/elapsed}
    result['checks'].append(item)
    print(name, 'PASS', round(item['output_tokens_per_second'], 2), 'tokens/s', flush=True)
    return item

nonce = uuid.uuid4().hex
small = 'Run marker: ' + nonce + '\n' + '\n'.join(f'record_{i:04d} = {{"value": {i*7%997}, "enabled": {str(i%3==0).lower()}}}' for i in range(150))
small_messages = [{'role': 'user', 'content': small + '\nReply exactly SMALL_OK.'}]
call('small_cold', small_messages, expected='SMALL_OK')
call('small_repeat', small_messages, expected='SMALL_OK')
call('small_repeat_again', small_messages, expected='SMALL_OK')
history = small_messages + [{'role': 'assistant', 'content': 'SMALL_OK'}]
for marker in ('BRANCH_ALPHA_72', 'BRANCH_BETA_93'):
    call('fork_'+marker, history+[{'role': 'user', 'content': 'Reply exactly '+marker+'.'}], expected=marker)

short = [{'role': 'user', 'content': 'Explain hash table collision resolution with clear Python examples.'}]
streamed('short_stream', short)
long_doc = 'Run marker: ' + nonce + '\nRecovery identifier: CACHE_RECOVERY_5729\n' + '\n'.join(f'record_{i:04d} = {{"value": {i*7%997}, "enabled": {str(i%3==0).lower()}}}' for i in range(2200))
long_prefix = [{'role': 'user', 'content': long_doc+'\nWhat is the recovery identifier? Reply with that identifier only.'}]
call('long_cold', long_prefix, expected='CACHE_RECOVERY_5729')
call('long_repeat', long_prefix, expected='CACHE_RECOVERY_5729')
long_messages = long_prefix + [{'role': 'assistant', 'content': 'CACHE_RECOVERY_5729'}, {'role': 'user', 'content': 'Explain how to validate these records efficiently in Python, including code and complexity.'}]
call('long_append_warmup', long_messages, cap=32)
streamed('long_stream', long_messages)

parallel_messages = [{'role': 'user', 'content': 'Write a detailed explanation of merge sort with a Python implementation.'}]
start = time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    futures = [pool.submit(call, 'parallel_'+str(i), parallel_messages, 128) for i in range(4)]
    for future in futures:
        future.result()
elapsed = time.monotonic()-start
tokens = sum(c['usage']['completion_tokens'] for c in result['checks'] if c['name'].startswith('parallel_'))
result['parallel'] = {'seconds': elapsed, 'output_tokens_per_second': tokens/elapsed, 'tokens': tokens}

# Throughput and shared-prefix correctness are distinct gates. Fork the same
# warm long conversation concurrently and require each branch's own answer.
shared_history = long_prefix + [{'role': 'assistant', 'content': 'CACHE_RECOVERY_5729'}]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    futures = []
    for index in range(4):
        marker = f'PARALLEL_BRANCH_{index}_N17'
        messages = shared_history + [{'role': 'user', 'content': 'Reply exactly '+marker+'.'}]
        futures.append(pool.submit(call, 'parallel_cache_'+str(index), messages, 64, marker))
    for future in futures:
        future.result()
shared_checks = [check for check in result['checks'] if check['name'].startswith('parallel_cache_')]
assert len(shared_checks) == 4 and all(check['cached_tokens'] > 0.9 * check['usage']['prompt_tokens'] for check in shared_checks), shared_checks

tools = [{'type': 'function', 'function': {'name': 'read_probe', 'description': 'Read the requested probe value.', 'parameters': {'type': 'object', 'properties': {'key': {'type': 'string'}}, 'required': ['key'], 'additionalProperties': False}}}]
tool_messages = [{'role': 'user', 'content': 'Call read_probe with key cache-test. Then return its value as the final answer.'}]
message = call('tool_request', tool_messages, tools=tools)
assert message.get('tool_calls') and len(message['tool_calls']) == 1, message
request = message['tool_calls'][0]
assert request['function']['name'] == 'read_probe' and json.loads(request['function']['arguments']) == {'key': 'cache-test'}, request
tool_messages += [message, {'role': 'tool', 'tool_call_id': request['id'], 'content': 'PROBE_TOOL_RESULT_641'}]
call('tool_result', tool_messages, tools=tools, expected='PROBE_TOOL_RESULT_641')
with urllib.request.urlopen(BASE+'/metrics', timeout=10) as response:
    metrics = response.read().decode()
result['metrics'] = [line for line in metrics.splitlines() if line.startswith('vllm:') and any(field in line for field in ('spec_decode_num_draft_tokens_total', 'spec_decode_num_accepted_tokens_total', 'prefix_cache_hits_total', 'prefix_cache_queries_total', 'num_preemptions_total'))]
result['passed'] = True
print('RESULT_JSON '+json.dumps(result), flush=True)

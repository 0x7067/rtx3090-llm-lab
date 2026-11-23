"""Run inside llama-swap after rollout; never prints the endpoint credential.

Uses the authenticated public API shape on loopback, without exposing a key
on the command line. Produces one JSON receipt per check; exits on failure.
"""
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import os
import struct
import sys
import threading
import time
import urllib.request
import zlib

MODEL = 'qwen3.8-27b-hyperqwen'
BASE = 'http://127.0.0.1:8080'
TOKEN = os.environ['LLAMA_API_TOKEN']


def request(path, body=None, timeout=600):
    data = None if body is None else json.dumps(body).encode()
    return urllib.request.urlopen(urllib.request.Request(BASE + path, data=data,
        headers={'Authorization': 'Bearer ' + TOKEN, 'Content-Type': 'application/json'}), timeout=timeout)


def payload(messages, **extra):
    return {'model': MODEL, 'messages': messages, 'temperature': 0, 'max_tokens': 128,
            'reasoning_effort': 'none', **extra}


def chat(messages, **extra):
    with request('/v1/chat/completions', payload(messages, **extra)) as response:
        return json.load(response)


def user(text):
    return [{'role': 'user', 'content': text}]


def content(response):
    return response['choices'][0]['message']['content'].strip()


def record(name, **details):
    print(json.dumps({'check': name, 'passed': True, **details}), flush=True)


def run():
    deadline = time.monotonic() + 600
    while True:
        try:
            with urllib.request.urlopen('http://127.0.0.1:5890/health', timeout=5) as response:
                assert response.status == 200
            break
        except (OSError, AssertionError):
            if time.monotonic() >= deadline:
                raise RuntimeError('SparseKV backend did not become healthy within 600 seconds')
            time.sleep(5)
    with urllib.request.urlopen('http://127.0.0.1:5890/v1/models', timeout=10) as response:
        backend = json.load(response)['data']
    assert backend and all(m['max_model_len'] == 200000 for m in backend), backend
    record('backend-ready', models=backend)
    with request('/v1/models') as response:
        models = json.load(response)['data']
    model = next(m for m in models if m['id'] == MODEL)
    assert model['meta']['llamaswap']['max_model_len'] == 200000, model
    record('metadata', model=model)
    answer = chat(user('What is 6 times 7? Reply with only the integer.'))
    assert content(answer) == '42', answer
    record('completion', answer=answer)

    text = ''
    done = False
    with request('/v1/chat/completions', payload(user('Count from one to five.'), stream=True)) as response:
        for line in response:
            if line.strip() == b'data: [DONE]':
                done = True
                break
            if line.startswith(b'data: '):
                data = json.loads(line[6:])
                for choice in data.get('choices', []):
                    text += choice.get('delta', {}).get('content') or ''
    assert done and text, text
    record('streaming', content=text)

    messages = user('Use get_temperature for Porto, then tell me the temperature.')
    tool = {'type': 'function', 'function': {'name': 'get_temperature',
        'description': 'Get temperature in Celsius.', 'parameters': {'type': 'object',
        'properties': {'city': {'type': 'string'}}, 'required': ['city'], 'additionalProperties': False}}}
    answer = chat(messages, tools=[tool], tool_choice={'type': 'function', 'function': {'name': 'get_temperature'}})
    message = answer['choices'][0]['message']
    calls = message['tool_calls']
    assert len(calls) == 1 and calls[0]['function']['name'] == 'get_temperature', answer
    assert json.loads(calls[0]['function']['arguments'])['city'].lower() == 'porto', answer
    answer = chat([*messages, message, {'role': 'tool', 'tool_call_id': calls[0]['id'],
        'content': '{"temperature_celsius":17}'}], tools=[tool])
    assert '17' in content(answer), answer
    record('tool-round-trip', answer=answer)

    answer = chat(user('Return a JSON object with answer equal to the number 42.'),
                  response_format={'type': 'json_object'})
    assert json.loads(content(answer))['answer'] == 42, answer
    record('json', answer=answer)

    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data))
    png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', 128, 128, 8, 2, 0, 0, 0))
    png += chunk(b'IDAT', zlib.compress((b'\0' + b'\xff\0\0' * 128) * 128)) + chunk(b'IEND', b'')
    answer = chat([{'role': 'user', 'content': [
        {'type': 'text', 'text': 'What solid color is this image? Reply with one word.'},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(png).decode()}}]}])
    assert content(answer).lower().strip('.!') == 'red', answer
    record('vision', answer=answer)

    # Distinct leading keys avoid treating a shared cached prefix as four independent requests.
    barrier = threading.Barrier(4)
    def concurrent(index):
        key = f'ORCHID_{index}_7182'
        messages = user(f'Archive {index}. The key is {key}.\n' +
                        'The routine status is unchanged.\n' * 4500 +
                        '\nReturn the archive key only.')
        barrier.wait(timeout=10)
        started = time.monotonic()
        answer = chat(messages)
        assert content(answer) == key, answer
        assert 30000 < answer['usage']['prompt_tokens'] < 34000, answer['usage']
        return {'client': index, 'wall_s': time.monotonic() - started, 'usage': answer['usage']}
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(concurrent, range(4)))
    record('four-clients', clients=rows)

    received = False
    with request('/v1/chat/completions', payload(user('Write a detailed 2000 word guide to sorting algorithms.'),
                                               stream=True, max_tokens=4096)) as response:
        for line in response:
            if line.startswith(b'data: ') and line.strip() != b'data: [DONE]':
                data = json.loads(line[6:])
                if any(c.get('delta', {}).get('content') for c in data.get('choices', [])):
                    received = True
                    break
    assert received, 'Stream ended before cancellation'
    time.sleep(2)
    answer = chat(user('What is 20 plus 22? Reply with only the integer.'))
    assert content(answer) == '42', answer
    record('request-after-cancellation', answer=answer)

    messages = user('Start key: MAPLE_3951\n' + 'The routine status is unchanged.\n' * 27000 +
                    '\nEnd key: CEDAR_8246\nWhat is the start key? Reply only with the key.')
    answer = chat(messages, max_tokens=64)
    assert content(answer) == 'MAPLE_3951', answer
    assert 185000 <= answer['usage']['prompt_tokens'] < 200000, answer['usage']
    record('near-window-retrieval', answer=answer)
    answer = chat([*messages, answer['choices'][0]['message'],
                   {'role': 'user', 'content': 'What is the end key? Reply only with the key.'}], max_tokens=64)
    assert content(answer) == 'CEDAR_8246', answer
    record('near-window-follow-up', answer=answer)
    record('complete')


try:
    run()
except Exception as error:
    print(json.dumps({'passed': False, 'error': f'{type(error).__name__}: {error}'}), flush=True)
    sys.exit(1)

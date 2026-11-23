"""Serving checks for structured output, tool use, prefix reuse and cancellation."""

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

p = argparse.ArgumentParser()
p.add_argument("--url", default="http://127.0.0.1:18892")
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
base = {
    "model": "qwen3.8-27b",
    "temperature": 0,
    "max_tokens": 256,
    "chat_template_kwargs": {"enable_thinking": False},
}


def call(messages, **extra):
    body = dict(base, messages=messages, **extra)
    with urlopen(
        Request(
            args.url + "/v1/chat/completions",
            json.dumps(body).encode(),
            {"Content-Type": "application/json"},
        ),
        timeout=300,
    ) as response:
        return json.load(response)


def content(result):
    return result["choices"][0]["message"]["content"].strip()


checks = []
r = call([{"role": "user", "content": "Return only the result of 17+25."}])
assert content(r) == "42"
checks.append("arithmetic")
schema = {
    "type": "object",
    "properties": {"result": {"type": "integer"}},
    "required": ["result"],
    "additionalProperties": False,
}
r = call(
    [{"role": "user", "content": "Calculate 17+25 and return result as JSON."}],
    response_format={
        "type": "json_schema",
        "json_schema": {"name": "answer", "strict": True, "schema": schema},
    },
)
assert json.loads(content(r)) == {"result": 42}
checks.append("json_schema")
tool = {
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two integers",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    },
}
messages = [
    {
        "role": "user",
        "content": "Use add to calculate 17+25. After receiving its result, return only the number.",
    }
]
r = call(
    messages,
    tools=[tool],
    tool_choice={"type": "function", "function": {"name": "add"}},
)
m = r["choices"][0]["message"]
tc = m["tool_calls"][0]
assert tc["function"]["name"] == "add" and json.loads(tc["function"]["arguments"]) == {
    "a": 17,
    "b": 25,
}
messages.extend([m, {"role": "tool", "tool_call_id": tc["id"], "content": "42"}])
r = call(messages, tools=[tool], tool_choice="none")
assert content(r) == "42"
checks.append("tool_round_trip")
prompt = (
    "Secret code: ORCHID-42.\n"
    + ("alpha beta gamma delta.\n" * 6000)
    + "\nReturn only the secret code."
)
for i in range(2):
    r = call([{"role": "user", "content": prompt}])
    assert content(r) == "ORCHID-42"
    checks.append("long_prefix_" + str(i))
body = dict(
    base,
    messages=[
        {"role": "user", "content": "List integers from 1 through 1000, one per line."}
    ],
    stream=True,
    max_tokens=2048,
)
with urlopen(
    Request(
        args.url + "/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    ),
    timeout=120,
) as response:
    for line in response:
        if line.startswith(b"data: ") and line[6:].strip() != b"[DONE]":
            event = json.loads(line[6:])
            if any(c.get("delta", {}).get("content") for c in event.get("choices", [])):
                break
r = call([{"role": "user", "content": "Return only the result of 17+25."}])
assert content(r) == "42"
checks.append("completion_after_cancel")
args.output.write_text(json.dumps({"passed": True, "checks": checks}, indent=2) + "\n")
print(json.dumps(checks), flush=True)

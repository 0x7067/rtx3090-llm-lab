#!/usr/bin/env python3
"""Qualification probes for k2-horizon-7b behind llama-swap (OpenAI-compatible)."""
import json, os, sys, time, urllib.request
BASE=os.environ.get("BASE","http://100.64.0.2/v1"); TOK=os.environ.get("TOK","")
MODEL=os.environ.get("MODEL","k2-horizon-7b")
def post(body, path="/chat/completions"):
    req=urllib.request.Request(BASE+path, data=json.dumps(body).encode(), headers={"Content-Type":"application/json","Authorization":f"Bearer {TOK}"})
    t=time.time(); r=json.load(urllib.request.urlopen(req, timeout=1800)); return r, time.time()-t
def show(tag, r, dt):
    m=r["choices"][0]["message"]; u=r.get("usage",{})
    print(f"\n== {tag} ({dt:.1f}s, prompt={u.get('prompt_tokens')} completion={u.get('completion_tokens')}, finish={r['choices'][0].get('finish_reason')})")
    print(" reasoning_content:", (m.get("reasoning_content") or m.get("reasoning") or "")[:300].replace("\n"," "))
    print(" content:", (m.get("content") or "")[:400].replace("\n"," "))
    print(" tool_calls:", json.dumps(m.get("tool_calls"))[:400])
    return m
# 1. exact answer, low effort
r,dt=post({"model":MODEL,"messages":[{"role":"user","content":"Reply with exactly the text K2_OK and nothing else."}],"max_tokens":512,"temperature":0,"reasoning_effort":"low"}); m=show("exact/low",r,dt)
assert (m.get("content") or "").strip()=="K2_OK", "exact answer failed"
# 2. tool call, default (xml) format
tools=[{"type":"function","function":{"name":"get_weather","description":"Get weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]
for fmt in ["xml","json","xml_typed"]:
    r,dt=post({"model":MODEL,"messages":[{"role":"user","content":"What's the weather in Curitiba? Use the tool."}],"tools":tools,"tool_choice":"required","max_tokens":2048,"temperature":0,"reasoning_effort":"low","chat_template_kwargs":{"tool_call_format":fmt}})
    m=show(f"toolcall/{fmt}",r,dt)
    tc=m.get("tool_calls") or []
    ok = bool(tc) and tc[0]["function"]["name"]=="get_weather" and "Curitiba" in tc[0]["function"]["arguments"]
    print(" PARSED_OK" if ok else " PARSE_FAIL")
# 3. multi-turn with tool result
r,dt=post({"model":MODEL,"messages":[{"role":"user","content":"What's the weather in Curitiba?"},{"role":"assistant","content":"","reasoning_content":"I should call the tool.","tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Curitiba\"}"}}]},{"role":"tool","tool_call_id":"call_1","content":"{\"temp_c\":14,\"sky\":\"rain\"}"}],"tools":tools,"max_tokens":1024,"temperature":0,"reasoning_effort":"low"}); show("tool-result-turn",r,dt)
# 4. coding, high effort
r,dt=post({"model":MODEL,"messages":[{"role":"user","content":"Write a Python function `lru(capacity)` returning an object with get(k) and put(k,v) in O(1). Only code, no prose."}],"max_tokens":8192,"temperature":1.0,"top_p":0.95,"reasoning_effort":"high"}); show("code/high",r,dt)
print("\nDONE")

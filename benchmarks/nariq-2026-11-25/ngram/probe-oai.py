"""ngram speculation probe: agent-style edit tasks vs fresh generation. Greedy, thinking off, 2 reps."""
import json, sys, time, urllib.request, pathlib
url, arm, out = sys.argv[1], sys.argv[2], sys.argv[3]
import os
model = os.environ.get("MODEL"); token = os.environ.get("TOKEN")
fx = pathlib.Path(__file__).parent / "fixtures"
sp = (fx / "speed-probe.py").read_text(); ag = (fx / "aggregate.py").read_text()
tasks = {
 "full-rewrite": f"Here is a Python file:\n```python\n{sp}\n```\nRename every occurrence of the variable `ctx` to `context_len` (only that identifier). Output the complete updated file in one code block, nothing else.",
 "add-feature": f"Here is a Python file:\n```python\n{ag}\n```\nAdd a `--csv PATH` option that also writes the table rows as CSV. Output the complete updated file in one code block, nothing else.",
 "search-replace": f"Here is a Python file:\n```python\n{sp}\n```\nAdd type hints to every function signature. Reply only with SEARCH/REPLACE blocks (<<<<<<< SEARCH / ======= / >>>>>>> REPLACE) that apply the change.",
 "explain": f"Here is a Python file:\n```python\n{ag}\n```\nExplain in prose how this script works and what could go wrong with it. Do not quote code.",
 "fresh-gen": "Write a Python module implementing a thread-safe LRU cache with TTL expiry, a size-in-bytes limit and statistics, with docstrings and pytest tests.",
}
res = []
for name, prompt in tasks.items():
  for rep in range(2):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 1536, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}, }
    if model: body["model"] = model
    hdr = {"Content-Type": "application/json"}
    if token: hdr["Authorization"] = "Bearer " + token
    req = urllib.request.Request(url, json.dumps(body).encode(), hdr)
    t0 = time.time(); r = json.load(urllib.request.urlopen(req, timeout=900)); wall = time.time() - t0
    t = r.get("timings") or {"predicted_n": r["usage"]["completion_tokens"], "predicted_per_second": r["usage"]["completion_tokens"] / wall}
    row = {"arm": arm, "task": name, "rep": rep, "n": t.get("predicted_n"), "tg": round(t.get("predicted_per_second", 0), 1),
           "pp": round(t.get("prompt_per_second", 0), 1), "draft": t.get("draft_n"), "accepted": t.get("draft_n_accepted"),
           "wall": round(wall, 1), "finish": r["choices"][0].get("finish_reason")}
    res.append(row); print(json.dumps(row), flush=True)
    if rep == 0: pathlib.Path(out).with_suffix(f".{name}.txt").write_text(r["choices"][0]["message"]["content"])
pathlib.Path(out).write_text("\n".join(json.dumps(x) for x in res) + "\n")

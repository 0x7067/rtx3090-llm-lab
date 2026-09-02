"""Collect a diverse prompt set (EN chat, code, DA instructions, DA reasoning, math) for
self-distillation data generation. Output: data/prompts.jsonl with {"id","src","messages","think"}."""
import json, random, os, glob, sys
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

R = random.Random(1234)
OUT = os.path.join(HERE, "data", "prompts.jsonl")
CACHE = os.path.join(HERE, "data", "hf")


def dl(repo, patterns):
    return snapshot_download(repo, repo_type="dataset", allow_patterns=patterns,
                             local_dir=f"{CACHE}/{repo.replace('/', '__')}")


def parquet_rows(root, cols=None):
    out = []
    for f in sorted(glob.glob(f"{root}/**/*.parquet", recursive=True)):
        try:
            out.extend(pq.read_table(f, columns=cols).to_pylist())
        except Exception as e:
            print("skip", f, e)
    return out


prompts = []


def add(src, msgs):
    prompts.append({"src": src, "messages": msgs})


# 1) UltraChat 200k (EN chat, first user turn; 25% keep 3 turns of history)
d = dl("HuggingFaceH4/ultrachat_200k", ["data/train_sft-00000-of-*.parquet"])
uc = parquet_rows(d, ["messages"])
R.shuffle(uc)
n = 0
for r in uc:
    m = r["messages"]
    if not m or m[0]["role"] != "user" or len(m[0]["content"]) < 20:
        continue
    if len(m) >= 3 and R.random() < 0.25:
        add("ultrachat", [{"role": x["role"], "content": x["content"]} for x in m[:3]])
    else:
        add("ultrachat", [{"role": "user", "content": m[0]["content"]}])
    n += 1
    if n >= 2300:
        break
print("ultrachat", n)

# 2) Magicoder OSS-Instruct (code)
d = dl("ise-uiuc/Magicoder-OSS-Instruct-75K", ["*.parquet", "data/*.parquet", "*.jsonl", "data/*.jsonl"])
mc = parquet_rows(d, ["problem", "lang"])
if not mc:
    for f in glob.glob(f"{d}/**/*.jsonl", recursive=True):
        mc.extend(json.loads(l) for l in open(f))
R.shuffle(mc)
for r in mc[:1100]:
    add("magicoder", [{"role": "user", "content": r["problem"]}])
print("magicoder", min(1100, len(mc)))

# 3) syvai/da-instruction (DA tasks; prefer ones with longer answers)
d = dl("syvai/da-instruction", ["*.parquet", "data/*.parquet"])
da = parquet_rows(d)
R.shuffle(da)
n = 0
seen_task = {}
for r in da:
    c = r["conversations"]
    if not c or c[0]["role"] != "user":
        continue
    ans = c[1]["content"] if len(c) > 1 else ""
    task = r.get("task", "")
    if len(ans) < 150 and R.random() < 0.85:
        continue  # mostly long-answer tasks
    if seen_task.get(task, 0) >= 200:
        continue
    seen_task[task] = seen_task.get(task, 0) + 1
    add("da-instruction", [{"role": "user", "content": c[0]["content"]}])
    n += 1
    if n >= 1100:
        break
print("da-instruction", n, seen_task)

# 4) syvai/reasoning-v1 (DA reasoning prompts)
d = dl("syvai/reasoning-v1", ["*.parquet", "data/*.parquet"])
rv = parquet_rows(d)
R.shuffle(rv)
n = 0
for r in rv:
    c = r["conversations"]
    if not c or c[0]["role"] != "user":
        continue
    add("da-reasoning", [{"role": "user", "content": c[0]["content"]}])
    n += 1
    if n >= 800:
        break
print("da-reasoning", n)

# 5) skolegpt-instruct (DA flan-style)
d = dl("kobprof/skolegpt-instruct", ["*.parquet", "data/*.parquet"])
sk = parquet_rows(d, ["system_prompt", "question"])
R.shuffle(sk)
for r in sk[:1000]:
    m = []
    if r.get("system_prompt"):
        m.append({"role": "system", "content": r["system_prompt"]})
    m.append({"role": "user", "content": r["question"]})
    add("skolegpt", m)
print("skolegpt", min(1000, len(sk)))

# 6) GSM8K train (EN math)
gs = parquet_rows(os.path.join(REPO, "bench", "quality-data", "gsm8k"), ["question"])
if not gs:
    d = dl("openai/gsm8k", ["main/train-*.parquet"])
    gs = parquet_rows(d, ["question"])
R.shuffle(gs)
for r in gs[:500]:
    add("gsm8k", [{"role": "user", "content": r["question"]}])
print("gsm8k", min(500, len(gs)))

R.shuffle(prompts)
with open(OUT, "w") as f:
    for i, p in enumerate(prompts):
        p["id"] = i
        base = 0.7 if p["src"] in ("da-reasoning", "gsm8k") else 0.4
        p["think"] = R.random() < base
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
print("total", len(prompts), "think", sum(p["think"] for p in prompts))

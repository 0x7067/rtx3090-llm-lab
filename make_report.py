# /// script
# requires-python = ">=3.11"
# ///
"""Builds the final comparison report from the speed JSON and sandbox results."""
import json, statistics as st

def load(p):
    return json.load(open(p))

def jsonl(p):
    return [json.loads(l) for l in open(p)]

q, t = load("results_speed_qwen.json"), load("results_speed_tiel.json")
qr, tr = jsonl("work_qwen/results.jsonl"), jsonl("work_tiel/results.jsonl")

def med(runs, key):
    return st.median(r[key] for r in runs)

rows = []
rows.append(("Decode, single stream (tok/s)",
             med(q["single_stream_decode"], "gen_tok_s"),
             med(t["single_stream_decode"], "gen_tok_s"), "higher"))
rows.append(("TTFT, short prompt (s)",
             med(q["single_stream_decode"], "ttft_s"),
             med(t["single_stream_decode"], "ttft_s"), "lower"))
rows.append(("Prefill, ~6.8k uncached (tok/s)",
             med(q["prefill"], "prefill_tok_s"),
             med(t["prefill"], "prefill_tok_s"), "higher"))
rows.append(("Aggregate, 4 concurrent (tok/s)",
             q["concurrent_4"]["aggregate_gen_tok_s"],
             t["concurrent_4"]["aggregate_gen_tok_s"], "higher"))

qp = sum(r["passed"] for r in qr) / len(qr) * 100
tp = sum(r["passed"] for r in tr) / len(tr) * 100
rows.append(("HumanEval pass@1 (%)", qp, tp, "higher"))

print(f"{'metric':<34} {'qwen3.8-27b':>13} {'Tiel-35B-A3B':>13}  {'winner':>8}  ratio")
print("-" * 84)
for name, qv, tv, better in rows:
    if better == "higher":
        win, ratio = ("Tiel", tv / qv) if tv > qv else ("qwen", qv / tv)
    else:
        win, ratio = ("Tiel", qv / tv) if tv < qv else ("qwen", tv / qv)
    print(f"{name:<34} {qv:>13.2f} {tv:>13.2f}  {win:>8}  {ratio:.2f}x")

qf = {r["task_id"] for r in qr if not r["passed"]}
tf = {r["task_id"] for r in tr if not r["passed"]}
print(f"\nfailed by both: {len(qf & tf)}")
print(f"only qwen failed ({len(qf - tf)}): {sorted(qf - tf, key=lambda x: int(x.split('/')[1]))[:12]}")
print(f"only Tiel failed ({len(tf - qf)}): {sorted(tf - qf, key=lambda x: int(x.split('/')[1]))[:12]}")

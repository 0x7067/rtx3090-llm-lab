# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Compare the 6144-token runs against the 12288-token re-runs, both models.

The original comparison could not separate code quality from replies that
arrived with no code in them. This prints both, so the split is visible rather
than assumed, and attributes every no-code reply to a finish_reason instead of
leaving it unexplained.

Usage: uv run compare_v2.py
"""
import json
import os


def rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def by_id(path):
    return {r["task_id"]: r for r in rows(path)}


def humaneval(label, cand_path, results_path):
    if not (os.path.exists(cand_path) and os.path.exists(results_path)):
        return None
    cands, results = by_id(cand_path), by_id(results_path)
    n = len(results)
    passed = sum(1 for t in results if results[t]["passed"])
    empty = [t for t in cands if not cands[t]["code"].strip()]
    reasons = {}
    for t in empty:
        key = cands[t].get("finish_reason") or "unrecorded"
        reasons[key] = reasons.get(key, 0) + 1
    return {"label": label, "n": n, "passed": passed,
            "pct": 100 * passed / n, "empty": len(empty), "reasons": reasons}


def multiturn(label, path):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    detail = d.get("no_code_detail") or {}
    reasons = {}
    for turns in detail.values():
        for t in turns:
            key = t.get("finish_reason") or "unrecorded"
            reasons[key] = reasons.get(key, 0) + 1
    return {"label": label, "solve_rate": d["solve_rate"],
            "turn1": d["cumulative_solve_rate_per_turn"]["1"],
            "mean_turns": d["mean_turns_to_solve"],
            "no_code": d["no_code_replies"], "reasons": reasons,
            "unsolved": len(d["unsolved"])}


def coded_only(label, mut_path, work_dir, max_turns):
    """Pass rate among replies that actually contained code.

    This is the split that showed the original multi-turn rows measured reply
    formatting: qwen was correct on every submission it made.
    """
    if not os.path.exists(work_dir):
        return None
    total = fails = 0
    for turn in range(1, max_turns + 1):
        p = os.path.join(work_dir, f"turn{turn}", "results.jsonl")
        if not os.path.exists(p):
            continue
        rs = rows(p)
        total += len(rs)
        fails += sum(1 for r in rs if not r["passed"])
    if not total:
        return None
    return {"label": label, "submissions": total, "passed": total - fails,
            "pct": 100 * (total - fails) / total}


def main():
    print("=" * 72)
    print("HumanEval pass@1")
    print("=" * 72)
    for spec in [
        ("qwen  6144", "candidates_qwen.jsonl", "work_qwen/results.jsonl"),
        ("qwen 12288", "cand_qwen_v2.jsonl", "work_qwen_v2/results.jsonl"),
        ("Tiel  6144", "candidates_tiel.jsonl", "work_tiel/results.jsonl"),
        ("Tiel 12288", "cand_tiel_v2.jsonl", "work_tiel_v2/results.jsonl"),
    ]:
        r = humaneval(*spec)
        if r is None:
            print(f"{spec[0]:<11} (not run)")
            continue
        print(f"{r['label']:<11} {r['passed']:>3}/{r['n']} = {r['pct']:5.1f}%"
              f"   no code: {r['empty']:>2}  {r['reasons'] or ''}")

    print()
    print("=" * 72)
    print("Multi-turn repair")
    print("=" * 72)
    for spec in [
        ("qwen  6144", "results_multiturn_qwen.json"),
        ("qwen 12288", "results_multiturn_qwen_v2.json"),
        ("Tiel  6144", "results_multiturn_tiel.json"),
        ("Tiel 12288", "results_multiturn_tiel_v2.json"),
    ]:
        r = multiturn(*spec)
        if r is None:
            print(f"{spec[0]:<11} (not run)")
            continue
        print(f"{r['label']:<11} 3-turn {r['solve_rate']:5.1f}%  "
              f"turn1 {r['turn1']:5.1f}%  mean turns {r['mean_turns']:.2f}  "
              f"unsolved {r['unsolved']:>2}  no-code {r['no_code']:>2}  "
              f"{r['reasons'] or ''}")

    print()
    print("=" * 72)
    print("Multi-turn, counting only replies that contained code")
    print("=" * 72)
    for spec in [
        ("qwen  6144", "mutants.jsonl", "work_mt_qwen", 3),
        ("qwen 12288", "mutants.jsonl", "work_mt_qwen_v2", 3),
        ("Tiel  6144", "mutants.jsonl", "work_mt_tiel", 3),
        ("Tiel 12288", "mutants.jsonl", "work_mt_tiel_v2", 3),
    ]:
        r = coded_only(*spec)
        if r is None:
            print(f"{spec[0]:<11} (not run)")
            continue
        print(f"{r['label']:<11} {r['passed']:>3}/{r['submissions']} "
              f"submissions correct = {r['pct']:5.1f}%")


main()

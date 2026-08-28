# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx"]
# ///
"""Multi-turn repair benchmark.

Each task starts from a verified-broken function plus its real failure output.
The model gets up to MAX_TURNS attempts; after every attempt the candidate is
re-executed in a network-less container and the *actual* stderr is appended to
the same conversation. Context accumulates across turns, so this measures
whether a model uses feedback or thrashes.

Usage: bench_multiturn.py BASE_URL MODEL mutants.jsonl out.json workdir [conc]
"""
import asyncio, json, os, re, subprocess, sys
import httpx

BASE, MODEL, MUTANTS, OUT, WORK = (sys.argv[1].rstrip("/"), sys.argv[2],
                                   sys.argv[3], sys.argv[4], sys.argv[5])
CONC = int(sys.argv[6]) if len(sys.argv) > 6 else 6
MAX_TURNS = 3

SYSTEM = ("You are fixing broken Python. Reply with the complete corrected "
          "function in a single ```python block. Keep the original function "
          "name and signature. Do not include tests.")

def extract_code(text: str) -> str | None:
    """Last fenced block, not the longest: repair replies often quote the buggy
    code first, and that quote can be longer than the fix."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    return blocks[-1] if blocks else None

def run_batch(candidates, tag):
    """Execute a turn's candidates in one container call."""
    d = os.path.join(WORK, tag)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "candidates.jsonl"), "w") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")
    subprocess.run(["cp", "batch_exec.py", d], check=True)
    subprocess.run(
        ["docker", "run", "--rm", "--network", "none", "--memory", "4g",
         "--cpus", "4", "-v", f"{os.path.abspath(d)}:/work",
         "python:3.12-slim", "python", "/work/batch_exec.py"],
        check=True, capture_output=True)
    return {json.loads(l)["task_id"]: json.loads(l)
            for l in open(os.path.join(d, "results.jsonl"))}

async def ask(client, sem, convo):
    async with sem:
        for attempt in range(3):
            try:
                r = await client.post(f"{BASE}/v1/chat/completions",
                                      # 12288, raised from 6144 on 2026-08-26
                                      # for the same reason as bench_quality.py:
                                      # a reply that runs out of budget arrives
                                      # with no code block and costs a turn.
                                      json={"model": MODEL, "temperature": 0,
                                            "max_tokens": 12288, "messages": convo},
                                      timeout=900)
                r.raise_for_status()
                choice = r.json()["choices"][0]
                msg = choice["message"]
                # See bench_quality.py: with vLLM's qwen3 reasoning parser a
                # reply that never leaves the reasoning block arrives with
                # content empty. Reported alongside finish_reason so a no-code
                # turn can be attributed rather than guessed at.
                if not (msg.get("content") or "") and (msg.get("reasoning_content") or ""):
                    return ("", f"{choice.get('finish_reason')}/reasoning-only")
                return (msg.get("content") or "", choice.get("finish_reason"))
            except Exception as e:
                if attempt == 2:
                    return (f"__ERROR__ {e}", "exception")
                await asyncio.sleep(5)

async def main():
    tasks = [json.loads(l) for l in open(MUTANTS)]
    state = {}
    for t in tasks:
        state[t["task_id"]] = {
            "task": t, "solved_turn": None, "no_code_turns": [], "turns": [],
            "convo": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content":
                 f"This function fails its test suite.\n\n```python\n{t['code']}\n```\n\n"
                 f"Test output:\n```\n{t['initial_error']}\n```\n\nFix it."}],
        }

    sem = asyncio.Semaphore(CONC)
    async with httpx.AsyncClient() as client:
        for turn in range(1, MAX_TURNS + 1):
            active = [tid for tid, s in state.items() if s["solved_turn"] is None]
            if not active:
                break
            print(f"--- turn {turn}: {len(active)} unsolved", flush=True)

            replies = await asyncio.gather(
                *[ask(client, sem, state[tid]["convo"]) for tid in active])

            candidates, coded = [], []
            for tid, (reply, finish) in zip(active, replies):
                s = state[tid]
                s["convo"].append({"role": "assistant", "content": reply})
                code = extract_code(reply)
                if code is None:
                    # A clarifying question or prose-only reply is a distinct
                    # outcome from a wrong fix; count it separately. Keep
                    # finish_reason and length: a no-code turn costs the task a
                    # turn, so "ran out of budget" and "chose to write prose"
                    # need to be tellable apart afterwards.
                    s["no_code_turns"].append(
                        {"turn": turn, "finish_reason": finish,
                         "raw_len": len(reply), "reply": reply})
                    s["convo"].append({"role": "user", "content":
                        "Reply with only the corrected function in a ```python block."})
                    continue
                coded.append(tid)
                candidates.append({"task_id": tid, "code": code,
                                   "test": s["task"]["test"],
                                   "entry_point": s["task"]["entry_point"]})

            if not candidates:
                continue
            results = run_batch(candidates, f"turn{turn}")
            for tid in coded:
                s, r = state[tid], results[tid]
                s["turns"].append({"turn": turn, "passed": r["passed"]})
                if r["passed"]:
                    s["solved_turn"] = turn
                else:
                    s["convo"].append({"role": "user", "content":
                        f"Still failing:\n```\n{r['error']}\n```\nFix it."})
            solved = sum(1 for s in state.values() if s["solved_turn"] is not None)
            print(f"    cumulative solved: {solved}/{len(tasks)}", flush=True)

    n = len(tasks)
    by_turn = {k: sum(1 for s in state.values() if s["solved_turn"] == k)
               for k in range(1, MAX_TURNS + 1)}
    solved = [s for s in state.values() if s["solved_turn"] is not None]
    summary = {
        "model": MODEL, "n_tasks": n,
        "solved_total": len(solved),
        "solve_rate": round(len(solved) / n * 100, 2),
        "newly_solved_per_turn": by_turn,
        "cumulative_solve_rate_per_turn": {
            k: round(sum(v for j, v in by_turn.items() if j <= k) / n * 100, 2)
            for k in range(1, MAX_TURNS + 1)},
        "mean_turns_to_solve": round(
            sum(s["solved_turn"] for s in solved) / max(len(solved), 1), 2),
        "solved_first_try": by_turn[1],
        "no_code_replies": sum(len(s["no_code_turns"]) for s in state.values()),
        "tasks_with_no_code": sum(1 for s in state.values() if s["no_code_turns"]),
        "unsolved": sorted([tid for tid, s in state.items() if s["solved_turn"] is None],
                           key=lambda x: int(x.split("/")[1])),
        "no_code_detail": {tid: s["no_code_turns"] for tid, s in state.items()
                           if s["no_code_turns"]},
    }
    json.dump(summary, open(OUT, "w"), indent=2)
    print(json.dumps(summary, indent=2))

asyncio.run(main())

# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx"]
# ///
"""HumanEval pass@1 generation against an OpenAI-compatible endpoint.

Greedy (temperature 0), one sample per problem. Writes candidates to a JSONL
that sandbox_runner.py executes inside a network-less docker container.

Usage: bench_quality.py BASE_URL MODEL HumanEval.jsonl candidates_out.jsonl [concurrency]
"""
import asyncio, json, re, sys
import httpx

BASE, MODEL, DATA, OUT = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3], sys.argv[4]
CONC = int(sys.argv[5]) if len(sys.argv) > 5 else 6

PROMPT_TMPL = (
    "Complete the following Python function.\n\n"
    "```python\n{prompt}\n```\n\n"
    "Return the complete, working function (including the signature and any needed "
    "imports) in a single ```python code block. No tests, no example usage."
)

def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return max(blocks, key=len)
    return text  # last resort: treat whole reply as code

async def gen(client, sem, problem):
    async with sem:
        for attempt in range(3):
            try:
                r = await client.post(
                    f"{BASE}/v1/chat/completions",
                    # 12288, matching bench_mmlu.py. The original runs used
                    # 6144 and produced replies with no code at all on 22 of
                    # qwen's 164 problems against 4 of Tiel's, which is what
                    # invalidated the first pass@1 comparison. bench_mmlu.py hit
                    # the same wall and doubled its limit; this is that fix,
                    # applied here on 2026-08-26.
                    json={"model": MODEL, "temperature": 0, "max_tokens": 12288,
                          "messages": [{"role": "user",
                                        "content": PROMPT_TMPL.format(prompt=problem["prompt"])}]},
                    timeout=900)
                r.raise_for_status()
                choice = r.json()["choices"][0]
                msg = choice["message"]
                content = msg.get("content") or ""
                # vLLM's qwen3 reasoning parser splits the reply: reasoning goes
                # to reasoning_content and only the answer lands in content. A
                # reply that never leaves the reasoning block therefore arrives
                # with content empty and scores zero. Captured for diagnosis
                # only - scoring still reads content, as the committed runs did.
                reasoning = msg.get("reasoning_content") or ""
                return {"task_id": problem["task_id"],
                        "code": extract_code(content),
                        "reply": content,
                        "raw_len": len(content),
                        "reasoning_len": len(reasoning),
                        "finish_reason": choice.get("finish_reason")}
            except Exception as e:
                if attempt == 2:
                    return {"task_id": problem["task_id"], "code": "", "error": str(e)}
                await asyncio.sleep(5)

async def main():
    problems = [json.loads(l) for l in open(DATA)]
    sem = asyncio.Semaphore(CONC)
    done = 0
    async with httpx.AsyncClient() as client:
        tasks = [asyncio.create_task(gen(client, sem, p)) for p in problems]
        results = []
        for t in asyncio.as_completed(tasks):
            results.append(await t)
            done += 1
            if done % 20 == 0:
                print(f"{done}/{len(problems)} generated", flush=True)
    by_id = {r["task_id"]: r for r in results}
    with open(OUT, "w") as f:
        for p in problems:
            r = by_id[p["task_id"]]
            # The reply is kept whole, not just its length. An empty "code"
            # scores as a failure, and extract_code collapses three different
            # causes into it: no fenced block, an empty fenced block, and an
            # empty reply. Only the text tells them apart afterwards.
            f.write(json.dumps({"task_id": p["task_id"], "code": r["code"],
                                "test": p["test"], "entry_point": p["entry_point"],
                                "error": r.get("error"),
                                "reply": r.get("reply"),
                                "raw_len": r.get("raw_len"),
                                "reasoning_len": r.get("reasoning_len"),
                                "finish_reason": r.get("finish_reason")}) + "\n")
    print(f"wrote {len(problems)} candidates to {OUT}")

asyncio.run(main())

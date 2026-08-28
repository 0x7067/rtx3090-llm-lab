# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "pandas", "pyarrow"]
# ///
"""MMLU-Pro accuracy against an OpenAI-compatible endpoint.

MMLU-Pro is 10-option multiple choice (vs MMLU's 4) and is scored with
chain-of-thought, so the model reasons then states a final letter. The sample is
stratified by category and drawn with a fixed seed, so every model sees the
identical question set.

Usage: bench_mmlu.py BASE_URL MODEL parquet out.json [per_category] [conc]
"""
import asyncio, json, re, sys
import httpx
import pandas as pd

BASE, MODEL, PARQUET, OUT = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3], sys.argv[4]
PER_CAT = int(sys.argv[5]) if len(sys.argv) > 5 else 25
CONC = int(sys.argv[6]) if len(sys.argv) > 6 else 4
LETTERS = "ABCDEFGHIJ"

PROMPT = """{question}

Options:
{options}

Think step by step, then end your reply with exactly this line:
Answer: X

where X is the letter of the correct option."""


def build(row):
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(row["options"]))
    return PROMPT.format(question=row["question"], options=opts)


def extract(text: str, n_opts: int) -> str | None:
    """Last 'Answer: X' wins; models often restate the format mid-reasoning."""
    valid = LETTERS[:n_opts]
    m = re.findall(rf"[Aa]nswer:\s*\(?([{valid}])\)?", text)
    if m:
        return m[-1]
    # Fallbacks for models that bold or parenthesise instead.
    m = re.findall(rf"\*\*\(?([{valid}])\)?\*\*", text)
    if m:
        return m[-1]
    m = re.findall(rf"\b([{valid}])\b", text[-200:])
    return m[-1] if m else None


async def ask(client, sem, row):
    async with sem:
        for attempt in range(3):
            try:
                r = await client.post(
                    f"{BASE}/v1/chat/completions",
                    json={"model": MODEL, "temperature": 0, "max_tokens": 12288,
                          "messages": [{"role": "user", "content": build(row)}]},
                    timeout=900)
                r.raise_for_status()
                choice = r.json()["choices"][0]
                content = choice["message"].get("content") or ""
                pred = extract(content, len(row["options"]))
                return {"question_id": int(row["question_id"]),
                        "category": row["category"], "gold": row["answer"],
                        "pred": pred, "correct": pred == row["answer"],
                        "no_answer": pred is None,
                        # A truncated reply scores wrong, but it is a different
                        # failure from a wrong answer -- track it separately so
                        # a token limit is not misread as a knowledge gap.
                        "truncated": choice.get("finish_reason") == "length"}
            except Exception as e:
                if attempt == 2:
                    return {"question_id": int(row["question_id"]),
                            "category": row["category"], "gold": row["answer"],
                            "pred": None, "correct": False, "no_answer": True,
                            "truncated": False, "error": str(e)[:200]}
                await asyncio.sleep(5)


async def main():
    df = pd.read_parquet(PARQUET)
    sample = pd.concat(
        [g.sample(n=min(PER_CAT, len(g)), random_state=1234)
         for _, g in df.groupby("category", sort=True)]
    ).reset_index(drop=True)
    rows = [r for _, r in sample.iterrows()]
    print(f"{len(rows)} questions across {sample['category'].nunique()} categories", flush=True)

    sem = asyncio.Semaphore(CONC)
    results, done = [], 0
    async with httpx.AsyncClient() as client:
        tasks = [asyncio.create_task(ask(client, sem, r)) for r in rows]
        for t in asyncio.as_completed(tasks):
            results.append(await t)
            done += 1
            if done % 25 == 0:
                acc = sum(x["correct"] for x in results) / len(results) * 100
                print(f"{done}/{len(rows)} running acc {acc:.1f}%", flush=True)

    n = len(results)
    correct = sum(r["correct"] for r in results)
    by_cat = {}
    for r in results:
        c = by_cat.setdefault(r["category"], [0, 0])
        c[1] += 1
        c[0] += r["correct"]
    summary = {
        "model": MODEL, "n": n, "correct": correct,
        "accuracy": round(correct / n * 100, 2),
        "unparseable": sum(r["no_answer"] for r in results),
        "truncated": sum(r.get("truncated", False) for r in results),
        "per_category": {k: {"correct": v[0], "n": v[1],
                             "acc": round(v[0] / v[1] * 100, 1)}
                         for k, v in sorted(by_cat.items())},
    }
    json.dump({"summary": summary, "results": results}, open(OUT, "w"), indent=2)
    print(json.dumps(summary, indent=2))

asyncio.run(main())

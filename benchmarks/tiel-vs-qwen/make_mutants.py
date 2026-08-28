# /// script
# requires-python = ">=3.11"
# ///
"""Seeds N deterministic bugs into each HumanEval canonical solution.

Single-operator bugs get one-shot fixed by strong models, which never exercises
the repair loop. Multiple simultaneous bugs force iteration: a model typically
finds one, re-runs, and meets the next failure.
"""
import json, random, re, sys

DATA, OUT = sys.argv[1], sys.argv[2]
N_MUT = int(sys.argv[3]) if len(sys.argv) > 3 else 3

MUTATIONS = [
    (r"(?<![<>=!])<=", "<"), (r"(?<![<>=!])>=", ">"),
    (r"(?<![<>=!+\-*/])<(?!=)", "<="), (r"(?<![<>=!+\-*/])>(?!=)", ">="),
    (r"==", "!="), (r"!=", "=="),
    (r"\band\b", "or"), (r"\bor\b", "and"),
    (r"\+ 1\b", "- 1"), (r"- 1\b", "+ 1"),
    (r"\bmin\(", "max("), (r"\bmax\(", "min("),
    (r"\bTrue\b", "False"), (r"\bFalse\b", "True"),
    (r"range\(len\(", "range(1, len("),
    (r"\[0\]", "[1]"), (r"\[1\]", "[0]"),
    (r"\bsorted\(", "reversed("),
    (r"\bappend\(", "insert(0, "),
    (r"\blen\(", "abs(len("),
]

def mutate(body: str, seed: int, n: int):
    rng = random.Random(seed)
    applied = []
    for _ in range(n):
        order = list(range(len(MUTATIONS)))
        rng.shuffle(order)
        for i in order:
            pat, rep = MUTATIONS[i]
            hits = list(re.finditer(pat, body))
            if not hits:
                continue
            h = rng.choice(hits)
            new = body[:h.start()] + rep + body[h.end():]
            if new != body:
                body, _ = new, applied.append(f"{pat} -> {rep}")
                break
    return (body, applied) if applied else (None, None)

out = []
for line in open(DATA):
    p = json.loads(line)
    seed = int(p["task_id"].split("/")[1])
    body, desc = mutate(p["canonical_solution"], seed, N_MUT)
    if body is None:
        continue
    out.append({"task_id": p["task_id"], "code": p["prompt"] + body,
                "test": p["test"], "entry_point": p["entry_point"],
                "mutation": desc, "n_mutations": len(desc)})

with open(OUT, "w") as f:
    for m in out:
        f.write(json.dumps(m) + "\n")
print(f"generated {len(out)} mutants, mean bugs/task="
      f"{sum(m['n_mutations'] for m in out)/len(out):.2f}")

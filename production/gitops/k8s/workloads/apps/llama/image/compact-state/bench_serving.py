"""ABBA latency and content checks; exact fixtures retain their meaningful text."""

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "optimization/ordered-kernels")
)
from chain_bench import cases, request


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--url", default="http://127.0.0.1:18892/v1")
    args = p.parse_args()
    fixtures = cases()
    with args.output.open("x") as out:

        def save(row, phase, repeat, concurrency):
            if row["name"].startswith("copy-"):
                expected = next(
                    c["expected"] for c in fixtures if c["name"] == row["name"]
                )
                cleaned = (
                    row["response"]
                    .strip()
                    .removeprefix("<EXTRACT>")
                    .removesuffix("</EXTRACT>")
                    .strip()
                )
                row["content_passed"] = cleaned == expected.strip()
            else:
                row["content_passed"] = row["passed"]
            row.update(
                arm=args.arm, phase=phase, repeat=repeat, concurrency=concurrency
            )
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(
                json.dumps(
                    {
                        k: row[k]
                        for k in [
                            "arm",
                            "name",
                            "phase",
                            "concurrency",
                            "content_passed",
                            "elapsed_seconds",
                        ]
                    }
                ),
                flush=True,
            )

        for c in fixtures:
            save(request(args.url, c), "warmup", 0, 1)
        for repeat in range(3):
            for c in fixtures:
                save(request(args.url, c), "measure", repeat, 1)
            for c in fixtures[:2]:
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                    jobs = [
                        dict(c, prompt=f"Batch identity {i}.\n" + c["prompt"])
                        for i in range(4)
                    ]
                    for row in pool.map(lambda c: request(args.url, c), jobs):
                        save(row, "measure", repeat, 4)
        for c in fixtures[:3]:
            save(request(args.url, c, 0.7), "sampled", 0, 1)


if __name__ == "__main__":
    main()

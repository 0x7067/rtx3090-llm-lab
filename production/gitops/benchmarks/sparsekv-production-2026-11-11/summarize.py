"""Summarize saved paired results, retaining incomplete phases and mismatches."""
import argparse
import json
from pathlib import Path
from statistics import median


def load_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(root):
    report = {"arms": {}, "request_mismatches": [], "restoration_verified": (root / "restored.json").exists()}
    all_rows = {}
    for arm in ("baseline", "candidate"):
        phases = {}
        all_rows[arm] = {}
        for phase in ("speed", "smoke", "quality", "long-context"):
            rows = load_rows(root / arm / f"{phase}.jsonl")
            all_rows[arm][phase] = rows
            summary_path = root / arm / f"{phase}-summary.json"
            phases[phase] = {
                "summary_written": summary_path.exists(), "saved_rows": len(rows),
                "passed": sum(row.get("passed") is True for row in rows),
                "transport_errors": sum(row.get("result", {}).get("status") != "ok" for row in rows),
                "summary": json.loads(summary_path.read_text()) if summary_path.exists() else None,
            }
        speed = {}
        for size in ("4k", "32k"):
            rows = [r for r in all_rows[arm]["speed"] if r["name"].startswith(size + "-")]
            values = [r["result"] for r in rows if r.get("passed")]
            speed[size] = {"samples": len(rows), "successful": len(values)}
            for key in ("ttft_s", "wall_s", "server_decode_tps", "e2e_output_tps"):
                samples = [r[key] for r in values if r.get(key) is not None]
                speed[size][key] = {"samples": samples, "median": median(samples) if len(samples) == 3 else None}
        report["arms"][arm] = {"phases": phases, "speed": speed}
    for phase in ("speed", "smoke", "quality", "long-context"):
        def mapping(arm):
            return {(r["name"], r.get("seed")): r for r in all_rows[arm][phase]}
        a, b = mapping("baseline"), mapping("candidate")
        for key in a.keys() & b.keys():
            ah, bh = a[key].get("request_sha256"), b[key].get("request_sha256")
            if ah and bh and ah != bh:
                report["request_mismatches"].append({"phase": phase, "name": key[0], "seed": key[1]})
    memory = root / "memory-kib.log"
    if memory.exists():
        readings = [int(line.split()[-1]) for line in memory.read_text().splitlines() if line.strip()]
        report["minimum_host_available_GiB"] = min(readings) / 1024**2 if readings else None
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory), indent=2))

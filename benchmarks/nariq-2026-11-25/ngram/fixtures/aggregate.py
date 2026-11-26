#!/usr/bin/env python3
"""Collect per-arm results into results/aggregate.json and results/aggregate.md.
Reports observations only; no ranking or significance claims."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
ARMS = [a["name"] for a in json.loads((ROOT / "arms.json").read_text())]


def mib(s):
    s = s.split("/")[0].strip()
    for unit, f in (("GiB", 1024), ("MiB", 1), ("KiB", 1 / 1024), ("B", 1 / 1048576)):
        if s.endswith(unit):
            return float(s[: -len(unit)]) * f
    return None


def telemetry(d):
    out = {}
    host = d / "host-telemetry.tsv"
    if host.exists():
        rows = [l.split("\t") for l in host.read_text().splitlines() if l and not l.startswith("abort")]
        if rows:
            out["host_min_available_gib"] = round(min(int(r[1]) for r in rows) / 1048576, 1)
            out["gpu_peak_used_mib"] = max(int(r[3]) for r in rows)
            mems = [mib(r[4]) for r in rows if len(r) > 4 and mib(r[4]) is not None]
            out["container_peak_mem_gib"] = round(max(mems) / 1024, 1) if mems else None
    gpu = d / "gpu-telemetry.tsv"
    if gpu.exists():
        vals = [l.split("\t")[2].split(", ") for l in gpu.read_text().splitlines() if l.count("\t") >= 2]
        vals = [v for v in vals if len(v) >= 3 and v[0].strip().isdigit()]
        if vals:
            out["gpu_peak_temp_c"] = max(int(v[0]) for v in vals)
            out["gpu_peak_power_w"] = max(float(v[1]) for v in vals)
    return out


def main():
    rows = []
    for arm in ARMS:
        d = RES / arm
        row = {"arm": arm, "status": (d / "STATUS").read_text().strip() if (d / "STATUS").exists() else "not-run"}
        s = d / "summary.json"
        if s.exists():
            q = json.loads(s.read_text())["quality"]
            row["quality_full"] = sum(t.get("full_score", 0) for t in q.values())
            row["quality_score"] = round(sum(t.get("mean_score", 0) * t.get("attempted", 0) for t in q.values()), 2)
            row["quality_attempted"] = sum(t.get("attempted", 0) for t in q.values())
            row["by_tier"] = {k: f"{t.get('full_score', 0)}/{t.get('attempted', 0)}" for k, t in q.items()}
        sp = d / "speed.json"
        if sp.exists():
            for r in json.loads(sp.read_text())["results"]:
                keep = {k: r.get(k) for k in ("prompt_tokens", "completion_tokens", "ttft_s", "prefill_tok_s",
                                              "decode_tok_s", "aggregate_tok_s", "error", "skipped", "errors")
                        if r.get(k) not in (None, [], "")}
                row[f"speed:{r['probe']}"] = keep
        row.update(telemetry(d))
        rows.append(row)
    (RES / "aggregate.json").write_text(json.dumps(rows, indent=2) + "\n")

    def g(r, probe, k):
        v = r.get(f"speed:{probe}", {}).get(k)
        return "-" if v is None else str(v)

    md = ["| Arm | Status | Quality (score / full / n) | Short prose tok/s | Short code tok/s | 64K decode | 64K prefill | 120K decode | 120K prefill | 4-caller aggregate | Peak VRAM MiB | Min host avail GiB | Container peak GiB |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        q = f"{r.get('quality_score', '-')} / {r.get('quality_full', '-')} / {r.get('quality_attempted', '-')}"
        md.append("| " + " | ".join([r["arm"], r["status"], q,
                                      g(r, "short-prose", "decode_tok_s"), g(r, "short-code", "decode_tok_s"),
                                      g(r, "ctx-64k", "decode_tok_s"), g(r, "ctx-64k", "prefill_tok_s"),
                                      g(r, "ctx-120k", "decode_tok_s"), g(r, "ctx-120k", "prefill_tok_s"),
                                      g(r, "concurrent-4", "aggregate_tok_s"),
                                      str(r.get("gpu_peak_used_mib", "-")), str(r.get("host_min_available_gib", "-")),
                                      str(r.get("container_peak_mem_gib", "-"))]) + " |")
    (RES / "aggregate.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

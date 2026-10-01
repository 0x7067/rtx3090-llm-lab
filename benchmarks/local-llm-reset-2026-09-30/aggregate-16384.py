import collections
import datetime
import json
import os
import pathlib


root = pathlib.Path(__file__).parent
arms = {
    "thinkingcap_llama": "thinkingcap",
    "gsq_llama": "gsq_llama",
    "w4_vllm": "w4_vllm",
    "orca_vllm": "orca_vllm",
    "strata_flash_next": "strata_flash_next",
}


def read_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def memory_gib(value):
    number, unit = value.split()[0][:-3], value.split()[0][-3:]
    scale = {"GiB": 1, "MiB": 1 / 1024, "KiB": 1 / 1024 / 1024}
    return float(number) * scale[unit]


report = {
    "plan_sha256": json.load((root / "plan-16384.json").open())["sha256"],
    "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
    "arms": {},
}

for arm, prefix in arms.items():
    run = root / f"{prefix}-16384-run"
    summary_path = root / f"{prefix}-16384-summary.json"
    summary = json.load(summary_path.open())
    rows = read_jsonl(run / "results.jsonl")
    quality_rows = [row for row in rows if row["score"] is not None]
    telemetry = []
    with (root / f"{prefix}-16384-telemetry.tsv").open() as handle:
        for line in handle:
            fields = line.rstrip().split("\t")
            if len(fields) == 5:
                telemetry.append(fields)
    thermal = read_jsonl(root / f"{prefix}-16384-thermal.jsonl")
    temperatures = [sample["gpus"][0] for sample in thermal]
    manifest = json.load((run / "manifest.json").open())
    performance = {}
    for name, item in summary["performance"].items():
        performance[name] = {
            metric: values["median"]
            for metric, values in item["metrics"].items()
        }
    report["arms"][arm] = {
        "quality_score": sum(row["score"] for row in quality_rows),
        "quality_max": len(quality_rows),
        "full_passes": sum(row["score"] == 1 for row in quality_rows),
        "partial_scores": sum(0 < row["score"] < 1 for row in quality_rows),
        "zero_scores": sum(row["score"] == 0 for row in quality_rows),
        "statuses": dict(collections.Counter(row["result"]["status"] for row in rows)),
        "finish_reasons": dict(collections.Counter(str(row["result"].get("finish_reason")) for row in rows)),
        "failures": summary["failures"],
        "performance": performance,
        "harness_elapsed_s": summary_path.stat().st_mtime - manifest["started_unix"],
        "resources": {
            "samples": len(telemetry),
            "min_host_available_gib": min(int(row[1]) for row in telemetry) / 1024 / 1024,
            "max_memory_psi_some_avg10": max(float(row[2]) for row in telemetry),
            "max_gpu_memory_mib": max(int(row[3]) for row in telemetry),
            "max_container_memory_gib": max(memory_gib(row[4]) for row in telemetry),
            "thermal_samples": len(thermal),
            "max_core_c": max(row["core"] for row in temperatures),
            "max_junction_c": max(row["junction"] for row in temperatures),
            "max_vram_c": max(row["vram"] for row in temperatures),
        },
    }

with (root / "comparison-16384.json").open("x") as handle:
    json.dump(report, handle, indent=2)
    handle.write("\n")

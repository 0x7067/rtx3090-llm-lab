#!/usr/bin/env bash
# Benchmarks the llama-swap model served by the in-cluster llama Service.
# Usage: scripts/benchmark-llama.sh [--vision]
#   default: RUNS chat completions of one prompt, reporting latency and completion tok/s.
#   --vision: synthetic PNG fixtures; MODE=benchmark|qualify|all (default all).
# Env: NAMESPACE SERVICE DEPLOYMENT CONTAINER MODEL RUNS MAX_TOKENS PORT TIMEOUT_SECONDS
#   READY_TIMEOUT_SECONDS OUTPUT_DIR API_BASE PROMPT
#   --vision only: MODE QUALIFY_MAX_TOKENS
#   API_BASE, when set, skips the port-forward.
set -euo pipefail

VISION=0
if [ "${1:-}" = "--vision" ]; then
  VISION=1
  shift
fi
if [ "$#" -gt 0 ]; then
  echo "usage: $0 [--vision]" >&2
  exit 2
fi

NAMESPACE="${NAMESPACE:-apps}"
SERVICE="${SERVICE:-llama}"
DEPLOYMENT="${DEPLOYMENT:-llama}"
CONTAINER="${CONTAINER:-llama-swap}"
MODEL="${MODEL:-qwen3.8-27b-hyperqwen}"
RUNS="${RUNS:-3}"
PORT="${PORT:-18080}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-900}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-300}"
PROMPT="${PROMPT:-Write a concise Python function that computes Fibonacci numbers iteratively, then explain its complexity in two sentences.}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/llama-benchmarks}"
API_BASE="${API_BASE:-}"
if [ "${VISION}" = 1 ]; then
  MAX_TOKENS="${MAX_TOKENS:-192}"
  MODE="${MODE:-all}" # benchmark, qualify, all
  QUALIFY_MAX_TOKENS="${QUALIFY_MAX_TOKENS:-1024}"
  label="-vision"
else
  MAX_TOKENS="${MAX_TOKENS:-256}"
  label=""
fi

die() {
  echo "error: $*" >&2
  exit 2
}

require_positive_int() {
  case "$2" in
    ''|*[!0-9]*|0) die "$1 must be a positive integer, got '$2'" ;;
  esac
}

require_positive_int PORT "${PORT}"
require_positive_int RUNS "${RUNS}"
require_positive_int MAX_TOKENS "${MAX_TOKENS}"
require_positive_int TIMEOUT_SECONDS "${TIMEOUT_SECONDS}"
require_positive_int READY_TIMEOUT_SECONDS "${READY_TIMEOUT_SECONDS}"
if [ "${VISION}" = 1 ]; then
  require_positive_int QUALIFY_MAX_TOKENS "${QUALIFY_MAX_TOKENS}"
  case "${MODE}" in
    benchmark|qualify|all) ;;
    *) die "MODE must be benchmark, qualify, or all, got '${MODE}'" ;;
  esac
fi

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "error: missing required command: $1" >&2
    exit 1
  }
}

need curl
need kubectl
need python3

mkdir -p "${OUTPUT_DIR}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
results_file="${OUTPUT_DIR}/llama-${MODEL}${label}-${timestamp}.jsonl"
pf_log="${OUTPUT_DIR}/llama${label}-port-forward-${timestamp}.log"

port_forward_pid=""
cleanup() {
  if [ -n "${port_forward_pid}" ]; then
    kill "${port_forward_pid}" >/dev/null 2>&1 || true
    wait "${port_forward_pid}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if [ -z "${API_BASE}" ]; then
  API_BASE="http://127.0.0.1:${PORT}"
  kubectl -n "${NAMESPACE}" port-forward --address 127.0.0.1 "svc/${SERVICE}" "${PORT}:8080" >"${pf_log}" 2>&1 &
  port_forward_pid="$!"
fi

# Default covers the ~5 min cold boot of the resident model.
ready_deadline=$(( $(date +%s) + READY_TIMEOUT_SECONDS ))
until curl -fsS "${API_BASE}/v1/models" >/dev/null 2>&1; do
  if [ -n "${port_forward_pid}" ] && ! kill -0 "${port_forward_pid}" >/dev/null 2>&1; then
    echo "error: kubectl port-forward exited before the API became reachable" >&2
    sed -n '1,120p' "${pf_log}" >&2
    exit 1
  fi
  if [ "$(date +%s)" -ge "${ready_deadline}" ]; then
    echo "error: ${API_BASE}/v1/models did not answer within ${READY_TIMEOUT_SECONDS}s" >&2
    echo "hint: a cold boot takes ~5 min; inspect: kubectl -n ${NAMESPACE} logs deploy/${DEPLOYMENT} -c ${CONTAINER}" >&2
    exit 1
  fi
  sleep 2
done

# Prints the error instead of an empty field when the exec fails, e.g. a wrong CONTAINER.
gpu_snapshot() {
  local out
  if out=$(kubectl -n "${NAMESPACE}" exec "deploy/${DEPLOYMENT}" -c "${CONTAINER}" -- \
    nvidia-smi --query-gpu=name,persistence_mode,power.draw,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits 2>&1); then
    printf '%s' "${out}"
  else
    printf 'unavailable (container=%s): %s' "${CONTAINER}" "${out}"
  fi
}

banner() {
  if [ "${VISION}" = 1 ]; then
    echo "Vision benchmark for ${MODEL} via ${API_BASE}"
    echo "mode=${MODE} runs=${RUNS} max_tokens=${MAX_TOKENS} qualify_max_tokens=${QUALIFY_MAX_TOKENS} timeout=${TIMEOUT_SECONDS}s"
  else
    echo "Benchmarking ${MODEL} via ${API_BASE}"
    echo "runs=${RUNS} max_tokens=${MAX_TOKENS} timeout=${TIMEOUT_SECONDS}s"
  fi
  echo "results=${results_file}"
  echo "note: run 1 includes model load if the backend was not already resident"
}

if [ "${VISION}" = 1 ]; then
  kind=vision
else
  kind=text
fi

banner
echo
echo "GPU before: $(gpu_snapshot)"
echo
python3 - "${kind}" "${API_BASE}" "${MODEL}" "${MAX_TOKENS}" "${QUALIFY_MAX_TOKENS:-0}" \
  "${TIMEOUT_SECONDS}" "${RUNS}" "${MODE:-}" "${OUTPUT_DIR}" "${timestamp}" "${results_file}" "${PROMPT}" <<'PY'
import base64
import json
import statistics
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

kind, api_base, model, max_tokens, qualify_max_tokens, timeout_seconds, runs, mode, output_dir, timestamp, results_file, prompt = sys.argv[1:]
VISION = kind == "vision"
max_tokens = int(max_tokens)
qualify_max_tokens = int(qualify_max_tokens)
timeout_seconds = float(timeout_seconds)
runs = int(runs)
results_path = Path(results_file)


def write_png(path, pixels):
    height = len(pixels)
    width = len(pixels[0])
    raw = bytearray()
    for row in pixels:
        raw.append(0)
        for r, g, b in row:
            raw.extend((r, g, b))

    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def canvas(width, height, color=(255, 255, 255)):
    return [[color for _ in range(width)] for _ in range(height)]


def rect(img, x0, y0, x1, y1, color):
    for y in range(max(0, y0), min(len(img), y1)):
        for x in range(max(0, x0), min(len(img[0]), x1)):
            img[y][x] = color


def circle(img, cx, cy, radius, color):
    r2 = radius * radius
    for y in range(max(0, cy - radius), min(len(img), cy + radius + 1)):
        for x in range(max(0, cx - radius), min(len(img[0]), cx + radius + 1)):
            if (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r2:
                img[y][x] = color


def triangle(img, top, left, right, color):
    x_top, y_top = top
    x_left, y_left = left
    x_right, y_right = right
    for y in range(max(0, y_top), min(len(img), max(y_left, y_right) + 1)):
        t = (y - y_top) / max(1, max(y_left, y_right) - y_top)
        xl = round(x_top + (x_left - x_top) * t)
        xr = round(x_top + (x_right - x_top) * t)
        for x in range(max(0, min(xl, xr)), min(len(img[0]), max(xl, xr) + 1)):
            img[y][x] = color


def make_fixtures(fixtures_dir):
    shapes = canvas(360, 180)
    circle(shapes, 70, 90, 36, (220, 30, 35))
    rect(shapes, 145, 55, 220, 130, (35, 80, 220))
    triangle(shapes, (290, 45), (245, 135), (335, 135), (35, 150, 75))
    shapes_path = fixtures_dir / "shape-order.png"
    write_png(shapes_path, shapes)

    chart = canvas(320, 200)
    rect(chart, 35, 35, 38, 165, (20, 20, 20))
    rect(chart, 35, 162, 285, 165, (20, 20, 20))
    rect(chart, 65, 105, 105, 162, (35, 80, 220))
    rect(chart, 140, 38, 180, 162, (230, 125, 25))
    rect(chart, 215, 72, 255, 162, (35, 150, 75))
    chart_path = fixtures_dir / "bar-chart.png"
    write_png(chart_path, chart)

    dots = canvas(280, 180)
    for cx, cy in [(45, 55), (95, 55), (145, 55), (70, 115), (125, 115)]:
        circle(dots, cx, cy, 16, (15, 15, 15))
    for cx, cy in [(205, 70), (230, 120)]:
        circle(dots, cx, cy, 16, (220, 30, 35))
    dots_path = fixtures_dir / "dot-count.png"
    write_png(dots_path, dots)

    return {
        "shape_order": shapes_path,
        "bar_chart": chart_path,
        "dot_count": dots_path,
    }


def build_cases(fixtures):
    final_only_system_prompt = "Return only the final answer. Do not explain. Do not reason step by step."
    benchmark_case = {
        "id": "shape_order",
        "image_path": fixtures["shape_order"],
        "prompt": "Describe the image in detail. Identify each colored shape and its left-to-right order.",
        "expected": "Left to right: red circle, blue square, green triangle.",
    }
    qual_cases = [
        {
            "id": "shape_order",
            "image_path": fixtures["shape_order"],
            "prompt": "Answer with only the colored shapes from left to right.",
            "image_first": True,
            "system_prompt": final_only_system_prompt,
            "expected": "Left to right: red circle, blue square, green triangle.",
            "criteria": [
                {"label": "red circle", "all": ["red", "circle"]},
                {"label": "blue square", "all": ["blue", "square"]},
                {"label": "green triangle", "all": ["green", "triangle"]},
                {"label": "left-to-right order", "order": ["red", "blue", "green"]},
            ],
        },
        {
            "id": "bar_chart",
            "image_path": fixtures["bar_chart"],
            "prompt": "Answer with only the color order of the bars from tallest to shortest.",
            "image_first": True,
            "system_prompt": final_only_system_prompt,
            "expected": "Orange is tallest; order is orange, green, blue.",
            "criteria": [
                {"label": "orange present", "all": ["orange"]},
                {"label": "green present", "all": ["green"]},
                {"label": "blue present", "all": ["blue"]},
                {"label": "height order", "order": ["orange", "green", "blue"]},
            ],
        },
        {
            "id": "dot_count",
            "image_path": fixtures["dot_count"],
            "prompt": "Count dots by color. Answer only: black=<number> red=<number>.",
            "image_first": True,
            "system_prompt": final_only_system_prompt,
            "expected": "5 black dots and 2 red dots.",
            "criteria": [
                {"label": "five black", "all": ["black"], "any": ["5", "five"]},
                {"label": "two red", "all": ["red"], "any": ["2", "two"]},
            ],
        },
    ]
    return benchmark_case, qual_cases


def data_url(path):
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def chat(messages, request_max_tokens, temperature):
    """POST one chat completion and return the parsed body with its wall-clock seconds."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": request_max_tokens,
        "temperature": temperature,
    }
    req = urllib.request.Request(
        f"{api_base}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code}: {detail}") from exc
    return body, time.perf_counter() - started


def text_of(raw):
    if isinstance(raw, str):
        return raw.strip()
    return json.dumps(raw, separators=(",", ":")) if raw else ""


def response_fields(body, elapsed):
    usage = body.get("usage") or {}
    choices = body.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    completion_tokens = usage.get("completion_tokens")
    tokens_per_second = None
    if isinstance(completion_tokens, (int, float)) and completion_tokens > 0:
        tokens_per_second = completion_tokens / elapsed
    return {
        "latency_seconds": elapsed,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion_tokens,
        "total_tokens": usage.get("total_tokens"),
        "tokens_per_second": tokens_per_second,
        "finish_reason": choice.get("finish_reason"),
        "reasoning_content": text_of(message.get("reasoning_content") or message.get("reasoning")),
        "content": text_of(message.get("content")),
    }


def text_request(run):
    body, elapsed = chat([{"role": "user", "content": prompt}], max_tokens, 0.2)
    row = {"run": run, "model": model, **response_fields(body, elapsed)}
    del row["finish_reason"], row["reasoning_content"]
    row["content_excerpt"] = row.pop("content")[:240]
    return row


def vision_request(case, phase, run, request_max_tokens):
    image_part = {"type": "image_url", "image_url": {"url": data_url(case["image_path"])}}
    text_part = {"type": "text", "text": case["prompt"]}
    content = [image_part, text_part] if case.get("image_first") else [text_part, image_part]
    messages = []
    if case.get("system_prompt"):
        messages.append({"role": "system", "content": case["system_prompt"]})
    messages.append({"role": "user", "content": content})
    body, elapsed = chat(messages, request_max_tokens, 0.1)
    return {
        "phase": phase,
        "run": run,
        "case": case["id"],
        "model": model,
        "image": str(case["image_path"]),
        **response_fields(body, elapsed),
        "max_tokens": request_max_tokens,
        "expected": case["expected"],
    }


def criterion_passed(text, criterion):
    lower = text.lower()
    ok = True
    if "all" in criterion:
        ok = ok and all(term in lower for term in criterion["all"])
    if "any" in criterion:
        ok = ok and any(term in lower for term in criterion["any"])
    if "order" in criterion:
        positions = [lower.find(term) for term in criterion["order"]]
        ok = ok and all(pos >= 0 for pos in positions) and positions == sorted(positions)
    return ok


def print_bench_header():
    print(f"{'run':<4} {'latency_s':<10} {'completion_tokens':<18} {'tok_s':<10} {'total_tokens':<12}")


def print_bench_row(row):
    tok_s = row["tokens_per_second"]
    tok_text = f"{tok_s:.2f}" if tok_s is not None else "n/a"
    print(
        f"{row['run']:<4} "
        f"{row['latency_seconds']:<10.3f} "
        f"{str(row['completion_tokens']):<18} "
        f"{tok_text:<10} "
        f"{str(row['total_tokens']):<12}"
    )


def emit(fh, row):
    fh.write(json.dumps(row, separators=(",", ":")) + "\n")
    fh.flush()


def run_text(fh):
    print_bench_header()
    bench_rows = []
    for run in range(1, runs + 1):
        row = text_request(run)
        emit(fh, row)
        bench_rows.append(row)
        print_bench_row(row)
    return bench_rows, []


def run_vision(fixtures_dir, fh):
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    benchmark_case, qual_cases = build_cases(make_fixtures(fixtures_dir))
    bench_rows, qual_rows = [], []
    if mode in {"benchmark", "all"}:
        print_bench_header()
        for run in range(1, runs + 1):
            row = vision_request(benchmark_case, "benchmark", run, max_tokens)
            emit(fh, row)
            bench_rows.append(row)
            print_bench_row(row)
        print()
    if mode in {"qualify", "all"}:
        print(f"{'case':<14} {'latency_s':<10} {'score':<7} answer")
        for case in qual_cases:
            row = vision_request(case, "qualify", 1, qualify_max_tokens)
            passed = [criterion["label"] for criterion in case["criteria"] if criterion_passed(row["content"], criterion)]
            row["criteria_passed"] = passed
            row["criteria_total"] = len(case["criteria"])
            emit(fh, row)
            qual_rows.append(row)
            compact_answer = " ".join(row["content"].split())[:140]
            print(f"{case['id']:<14} {row['latency_seconds']:<10.3f} {len(passed)}/{len(case['criteria']):<5} {compact_answer}")
        print()
    return bench_rows, qual_rows


def print_summary(bench_rows, qual_rows):
    if bench_rows:
        latencies = [row["latency_seconds"] for row in bench_rows]
        speeds = [row["tokens_per_second"] for row in bench_rows if row["tokens_per_second"] is not None]
        print(f"avg_latency_s={statistics.mean(latencies):.3f}")
        if len(latencies) > 1:
            print(f"warm_avg_latency_s={statistics.mean(latencies[1:]):.3f}")
        if speeds:
            print(f"avg_completion_tok_s={statistics.mean(speeds):.2f}")
            if len(speeds) > 1:
                print(f"warm_avg_completion_tok_s={statistics.mean(speeds[1:]):.2f}")
    if qual_rows:
        passed = sum(len(row["criteria_passed"]) for row in qual_rows)
        total = sum(row["criteria_total"] for row in qual_rows)
        print(f"qualitative_score={passed}/{total}")


fixtures_dir = Path(output_dir) / f"llama-vision-fixtures-{timestamp}"
# Rows are appended as each request finishes so a later failure keeps earlier results.
with results_path.open("w", encoding="utf-8") as results_fh:
    if VISION:
        bench_rows, qual_rows = run_vision(fixtures_dir, results_fh)
    else:
        bench_rows, qual_rows = run_text(results_fh)

print_summary(bench_rows, qual_rows)
if VISION:
    print(f"fixtures_dir={fixtures_dir}")
print(f"results_jsonl={results_path}")
PY
echo
echo "GPU after: $(gpu_snapshot)"

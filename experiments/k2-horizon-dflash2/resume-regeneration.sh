#!/usr/bin/env bash
# Exclusive use of the 3090; restore the local API on success or failure.
set -euo pipefail
if (( $# > 1 )) || [[ -n "${1:-}" && "$1" != --prepare-only ]]; then
  echo 'Usage: resume-regeneration.sh [--prepare-only]' >&2
  exit 2
fi
W=/data/buttercup_6tb/specforge-work
E=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUN="$W/resume-2026-09-05"
mkdir -p "$RUN"
exec 9>"$W/.k2-training.lock"
flock -n 9 || { echo 'Another K2 training job owns the lock' >&2; exit 1; }
export PATH="$HOME/.local/bin:$PATH"
# shellcheck source=/dev/null
source "$W/venv/bin/activate"
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
# User services do not inherit the interactive shell's kubeconfig export.
export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"

# Keep the original outputs intact: their normalization already erased the
# delimiters needed to recover swallowed answers without generating them again.
python - "$W" <<'PY'
import json, os, pathlib, sys, tempfile
w = pathlib.Path(sys.argv[1])
dst = w / 'cache/dataset/regen-output-v2.jsonl'
if not dst.exists():
    kept = truncated = rejected = 0
    with tempfile.NamedTemporaryFile(mode='w', dir=dst.parent, delete=False) as out:
        for line in (w / 'cache/dataset/regen-output.jsonl').open():
            row = json.loads(line)
            answer = row['conversations'][-1]
            if row['finish_reason'] == 'length':
                truncated += 1
            elif row['finish_reason'] in ('stop', 'tool_calls') and (
                (answer.get('content') or '').strip() or answer.get('tool_calls')
            ):
                kept += 1
            else:
                rejected += 1
                continue
            out.write(line)
    os.replace(out.name, dst)
    print(f'Seeded {kept} usable and {truncated} truncated rows; regenerate {rejected} malformed responses')
ids = [json.loads(line)['id'] for line in dst.open()]
inputs = {json.loads(line)['id'] for line in (w / 'cache/dataset/regen-input.jsonl').open()}
assert len(ids) == len(set(ids)), 'Duplicate output IDs'
assert set(ids) <= inputs, 'Output has IDs outside the frozen input'
print(f'{len(inputs) - len(ids)} rows remain')
PY

python -m unittest discover -s "$E" -p test_resume.py > "$RUN/preflight.log" 2>&1
if [[ "${1:-}" == --prepare-only ]]; then
  echo 'Dataset and parser preflight ready; GPU and local API untouched'
  exit 0
fi
for command in flux kubectl nvidia-smi curl setsid; do
  command -v "$command" > /dev/null || { echo "Missing command: $command" >&2; exit 1; }
done
replicas=$(kubectl -n apps get deploy llama -o jsonpath='{.spec.replicas}')
suspended=$(kubectl -n flux-system get kustomization apps -o jsonpath='{.spec.suspend}')
[[ "$replicas" == 1 && "$suspended" != true ]] || {
  echo 'Expected the normal one-replica API with Flux apps active; another operation may own it' >&2
  exit 1
}
server_pid=''
restore=0
cleanup() {
  rc=$?
  trap - EXIT INT TERM
  set +e
  if [[ -n "$server_pid" ]]; then
    kill -TERM -- "-$server_pid" 2>/dev/null
    for ((attempt=0; attempt<30; attempt++)); do
      kill -0 -- "-$server_pid" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$server_pid" 2>/dev/null
    wait "$server_pid" 2>/dev/null
  fi
  if [[ "$restore" == 1 ]]; then
    kubectl -n apps scale deploy/llama --replicas="$replicas" || rc=1
    flux resume kustomization apps || rc=1
    kubectl -n apps rollout status deploy/llama --timeout=15m || rc=1
  fi
  echo "Regeneration finished with exit=$rc; local API restoration attempted"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
restore=1
flux suspend kustomization apps
kubectl -n apps scale deploy/llama --replicas=0
kubectl -n apps wait --for=delete pod -l app.kubernetes.io/name=llama --timeout=5m
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
[[ "$used" -lt 500 ]] || { echo "GPU remains occupied: ${used} MiB" >&2; exit 1; }

setsid bash "$E/serve-k2-sglang.sh" > "$RUN/server.log" 2>&1 &
server_pid=$!
ready=0
for ((attempt=0; attempt<90; attempt++)); do
  kill -0 "$server_pid" || { echo 'SGLang exited during startup' >&2; exit 1; }
  if curl -fsS --max-time 5 -A 'OpenAI File Downloader, XaiImageApiFetch/1.0' \
      http://127.0.0.1:30000/health > /dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 10
done
[[ "$ready" == 1 ]] || { echo 'SGLang readiness timed out' >&2; exit 1; }

PYTHONPATH="$E${PYTHONPATH:+:$PYTHONPATH}" python - "$W" <<'PY'
import sys
from regenerate_sessions import call
reply = call('http://127.0.0.1:30000/v1', '', {
    'model': sys.argv[1] + '/models/IFM/K2-Horizon-7B',
    'messages': [{'role': 'user', 'content': 'What is 2+2? Answer with the number only.'}],
    'reasoning_effort': 'medium', 'max_tokens': 512, 'temperature': 0,
}, 180)
choice = reply['choices'][0]
assert choice['finish_reason'] == 'stop', 'Smoke answer did not finish'
assert (choice['message'].get('content') or '').strip() == '4', 'Smoke answer missing or incorrect'
print('Live medium-effort answer parsing passed')
PY

python -u "$E/regenerate_sessions.py" \
  "$W/cache/dataset/regen-input.jsonl" "$W/cache/dataset/regen-output-v2.jsonl" \
  --base-url http://127.0.0.1:30000/v1 --model "$W/models/IFM/K2-Horizon-7B" \
  --effort medium --concurrency 8 --max-tokens 4096
python "$E/build_capture_set.py" "$W/cache/dataset/regen-output-v2.jsonl" \
  "$W/cache/dataset/k2-eagle3-regenerated.jsonl"
echo 'REGENERATION-DONE: regenerated capture input is ready'

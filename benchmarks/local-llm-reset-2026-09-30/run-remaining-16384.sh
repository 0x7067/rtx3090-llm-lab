#!/usr/bin/env bash
set -euo pipefail

task_dir=/data/docker-services/rtx3090-llm-lab/benchmarks/local-llm-reset-2026-09-30
harness=$task_dir/../paired-harness/harness.py
plan=$task_dir/plan-16384.json
llama_image=ghcr.io/ggml-org/llama.cpp@sha256:7149a45c80596644320de1d565db1b3b8e70b22c66fed612e69b73f6b1e0d746
vllm_image=local/llm-baseline-vllm:0.30.0
strata_image=local/strata-baseline:30ec18e

cd "$task_dir"
trap 'touch /tmp/local-llm-reset-16384-complete' EXIT

wait_ready() {
  local container=$1
  local url=$2
  local attempts=$3
  local i
  for i in $(seq 1 "$attempts"); do
    if curl -fsS "$url" >/dev/null; then
      return 0
    fi
    if [[ $(docker inspect "$container" --format '{{.State.Running}}') != true ]]; then
      docker logs "$container"
      return 1
    fi
    sleep 5
  done
  docker logs --tail 200 "$container"
  return 1
}

summarize() {
  local arm=$1
  local run_dir=$2
  python3 "$harness" summarize --plan "$plan" --run "$run_dir" --out "$arm-16384-summary.json" --markdown "$arm-16384-summary.md"
}

run_arm() {
  local arm=$1
  local container=$2
  local url=$3
  bash monitor.sh "$container" "$arm-16384-telemetry.tsv" > "$arm-16384-monitor.log" 2>&1 &
  local monitor_pid=$!
  local ready_status=0
  wait_ready "$container" "$url" 180 || ready_status=$?
  if [[ $ready_status -ne 0 ]]; then
    docker logs "$container" > "$arm-16384-server.log" 2>&1 || true
    docker stop -t 30 "$container" || true
    wait "$monitor_pid" || true
    return 1
  fi
  local run_status=0
  python3 "$harness" run --plan "$plan" --arm "$arm" --out "$arm-16384-run" --timeout 1800 > "$arm-16384-harness.log" 2>&1 || run_status=$?
  docker logs "$container" > "$arm-16384-server.log" 2>&1 || true
  docker stop -t 30 "$container" || true
  local monitor_status=0
  wait "$monitor_pid" || monitor_status=$?
  if grep -q '^abort' "$arm-16384-telemetry.tsv"; then
    return 1
  fi
  if [[ $monitor_status -ne 0 || $run_status -ne 0 ]]; then
    return 1
  fi
  summarize "$arm" "$arm-16384-run"
}

while systemctl --user is-active --quiet local-llm-thinkingcap-16384.service; do
  sleep 10
done

[[ $(systemctl --user show local-llm-thinkingcap-16384.service --property Result --value) == success ]]
[[ $(wc -l < thinkingcap-16384-run/results.jsonl) -eq 58 ]]
docker logs thinkingcap-16384-final > thinkingcap-16384-server.log 2>&1 || true
docker stop -t 30 thinkingcap-16384-final || true
systemctl --user stop local-llm-thinkingcap-monitor.service || true
summarize thinkingcap thinkingcap-16384-run

[[ $(docker image inspect "$llama_image" --format '{{.Id}}') == sha256:20b3edfeb4086a504829db8feb63b00734b5ebe16d55f47de2ee00225575db70 ]]
docker run -d --name gsq-16384 --gpus all --network host --memory 32g --memory-swap 32g --cpus 8 -v /data/models/qwen3.8/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF:/models:ro "$llama_image" --model /models/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf --ctx-size 150000 --parallel 1 --n-gpu-layers 999 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --jinja --host 127.0.0.1 --port 18181 --alias gsq-qwen3.8-27b --threads 10
run_arm gsq_llama gsq-16384 http://127.0.0.1:18181/health

[[ $(docker image inspect "$vllm_image" --format '{{.Id}}') == sha256:1d60e0a2fc9eb37042f94b8d5fb23457df6a27ca365305854b1bbcea8b4d9308 ]]
docker run -d --name w4-16384 --gpus all --network host --ipc host --memory 44g --memory-swap 44g --cpus 8 -v /data/models/qwen3.8/dbirks/Qwen3.8-27B-W4A16-AutoRound:/models/w4:ro "$vllm_image" /models/w4 --served-model-name w4-qwen3.8-27b --port 18180 --host 127.0.0.1 --max-model-len 98304 --gpu-memory-utilization 0.972 --max-num-seqs 8 --no-enable-log-requests --enable-prefix-caching --kv-cache-dtype fp8 --mamba-ssm-cache-dtype float16 --language-model-only --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
run_arm w4_vllm w4-16384 http://127.0.0.1:18180/health

docker run -d --name orca-16384 --gpus all --network host --ipc host --memory 44g --memory-swap 44g --cpus 8 -v /data/models/qwen3.8/orcarouter/OrcaSAQ2-27B:/models/orca:ro -e PRESET=16gb-mtp -e MODEL=/models/orca -e SERVED_NAME=orca-qwen3.8-27b -e PORT=18180 -e BIND=127.0.0.1 -e MAXLEN=150000 -e MAXSEQS=8 --entrypoint /opt/orcasaq2/presets/serve.sh "$vllm_image"
run_arm orca_vllm orca-16384 http://127.0.0.1:18180/health

[[ $(docker image inspect "$strata_image" --format '{{.Id}}') == sha256:f0aead168c875cb4035ad7e8647c015939658866e7f37089bb2a126f2f06dda4 ]]
docker run -d --name strata-16384 --gpus all --network host --memory 44g --memory-swap 44g --cpus 8 --ulimit memlock=-1:-1 -v /data/strata-baseline:/data -e VISION=no -e HOST=127.0.0.1 -e PORT=18182 -e FAMILY=qwen -e MODEL=IQ2_XS -e CONTEXT=150000 -e KV=int8 -e LOW_RAM=on -e GPU=0 "$strata_image"
run_arm strata_flash_next strata-16384 http://127.0.0.1:18182/health

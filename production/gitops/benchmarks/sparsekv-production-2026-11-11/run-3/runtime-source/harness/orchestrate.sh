#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "$0")" && pwd)
production="$here/../production"
unit=sparsekv-production-bench
reconcile=kustomize.toolkit.fluxcd.io/reconcile
owner=bench.denguinho.org/sparsekv-owner
selector=app.kubernetes.io/name=llama
k=(kubectl --request-timeout=12s)
ctr=(sudo -n k3s ctr -n k8s.io)
mode=${1:-}
out=${2:-}

die() { printf '%s\n' "$*" >&2; exit 1; }
deployment() { "${k[@]}" -n apps get deploy llama -o json; }
available_kib() { awk '$1=="MemAvailable:" {print $2}' /proc/meminfo; }
disk_ok() {
  "${k[@]}" get node denguinho-server -o json |
    jq -e 'any(.status.conditions[]; .type == "DiskPressure" and .status == "False")' >/dev/null
}
gpu_empty() {
  local owners used
  owners=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader) || return 1
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) || return 1
  [[ -z $owners && $used =~ ^[0-9]+$ && $used -lt 256 ]]
}

preflight() {
  [[ $(hostname) == denguinho-server ]] || die "Wrong host"
  [[ -x $here/../launch-arm.sh && -s $(cat "$here/../.scratch")/overlay.tar ]] ||
    die "Candidate launcher or overlay missing"
  disk_ok || die "Node disk pressure is present or unknown"
  local active scope pods pod_uid pid processes tasks containers listeners template_cm template_data
  for scope in system user; do
    local systemctl=(systemctl)
    [[ $scope != user ]] || systemctl+=(--user)
    active=$("${systemctl[@]}" list-units --type=service --state=active,activating,deactivating,reloading \
      --no-legend --plain 'qwen-deletion-*') || die "Cannot check $scope qwen-deletion services"
    [[ -z $active ]] || die "Another Qwen job is active in $scope scope: $active"
  done

  live=$(deployment)
  jq -e --arg reconcile "$reconcile" --arg owner "$owner" '
    .spec.replicas == 1 and .status.availableReplicas == 1 and
    .status.readyReplicas == 1 and .status.updatedReplicas == 1 and
    .status.observedGeneration == .metadata.generation and
    (.metadata.annotations[$owner] == null) and (.metadata.annotations[$reconcile] != "disabled")
  ' <<<"$live" >/dev/null || die "Production is unavailable, rolling, or already claimed"
  diff -q <(jq -Sc '.spec.template.spec.containers[] | select(.name == "llama-swap").image' <<<"$live") \
    <(jq -Sc '.spec.template.spec.containers[] | select(.name == "llama-swap").image' "$production/deployment.json") ||
    die "Production image differs from the benchmark snapshot"
  diff -q <("${k[@]}" -n apps get cm llama-swap-config -o json | jq -Sc .data) \
    <(jq -Sc .data "$production/swap-configmap.json") || die "Production model config differs from the benchmark snapshot"
  template_cm=$(jq -er '.spec.template.spec.volumes[] | select(.name == "chat-template").configMap.name' <<<"$live")
  template_data=$("${k[@]}" -n apps get cm "$template_cm" -o json)
  local file
  for file in chat_template.jinja start_qwen_cache.py; do
    diff -q <(jq -erj --arg file "$file" '.data[$file]' <<<"$template_data") "$production/$file" ||
      die "Production $file differs from the benchmark snapshot"
  done

  pods=$("${k[@]}" -n apps get pods -l "$selector" -o json)
  pod_uid=$(jq -er '
    [.items[] | select(.status.phase != "Failed" and .status.phase != "Succeeded")] | if length == 1 and .[0].metadata.deletionTimestamp == null and
      any(.[0].status.conditions[]; .type == "Ready" and .status == "True")
    then .[0].metadata.uid else error("expected one Ready production pod") end
  ' <<<"$pods")
  processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
  [[ -n $processes ]] || die "Production model is not resident on the GPU"
  for pid in $processes; do
    [[ $pid =~ ^[0-9]+$ ]] || die "Unexpected GPU process listing"
    if ! grep -Eq "pod(${pod_uid}|${pod_uid//-/_})" "/proc/$pid/cgroup"; then
      die "GPU owner PID $pid is outside the production pod"
    fi
  done
  tasks=$("${ctr[@]}" tasks list -q)
  containers=$("${ctr[@]}" containers list -q)
  if grep -Eq '^sparsekv-bench-(baseline|candidate)$' <<<"$tasks"$'\n'"$containers"; then
    die "Previous benchmark container or task is present"
  fi
  listeners=$(ss -Hltn '( sport = :5890 or sport = :18091 )')
  [[ -z $listeners ]] || die "A benchmark listener is already present"
  # This replaces the resident model, rather than adding another model.
  # Count only resident anon+shmem released by the verified production pod;
  # ordinary file cache may already be included in MemAvailable.
  local pod_stats released_kib projected_kib
  pod_stats="/sys/fs/cgroup/kubepods.slice/kubepods-burstable.slice/kubepods-burstable-pod${pod_uid//-/_}.slice/memory.stat"
  released_kib=$(awk '$1=="anon" || $1=="shmem" {n+=$2} END {printf "%.0f", n/1024}' "$pod_stats")
  projected_kib=$(( $(available_kib) + released_kib ))
  # 17.8 GB candidate pool + native 8 GB reserve + ~6 GiB runtime budget.
  # Native allocation checks and the 36 GiB container cap remain unchanged.
  (( projected_kib >= 30 * 1024 * 1024 )) || die "Projected available RAM after releasing production is below 30 GiB"
  awk -v n="$projected_kib" 'BEGIN {printf "Projected available RAM after handoff: %.2f GiB\n", n/1024/1024}'
  echo "Preflight passed: production Ready, only production owns the GPU, no disk pressure or active deletion trial"
}

stop_arm() {
  local id="sparsekv-bench-$1" tasks containers deadline=$((SECONDS+15))
  tasks=$("${ctr[@]}" tasks list -q) || return 1
  if grep -Fxq "$id" <<<"$tasks"; then
    "${ctr[@]}" tasks kill --signal SIGTERM --all "$id" || true
    while (( SECONDS < deadline )); do
      tasks=$("${ctr[@]}" tasks list -q) || return 1
      grep -Fxq "$id" <<<"$tasks" || break
      sleep 1
    done
    if grep -Fxq "$id" <<<"$tasks"; then
      "${ctr[@]}" tasks delete --force "$id" || true
    fi
  fi
  tasks=$("${ctr[@]}" tasks list -q) || return 1
  grep -Fxq "$id" <<<"$tasks" && return 1
  containers=$("${ctr[@]}" containers list -q) || return 1
  if grep -Fxq "$id" <<<"$containers"; then
    if ! "${ctr[@]}" containers delete "$id"; then
      containers=$("${ctr[@]}" containers list -q) || return 1
      grep -Fxq "$id" <<<"$containers" && return 1
    fi
  fi
  return 0
}

check_inference() {
  # Use the pod's direct loopback backend and require an actual answer.
  "${k[@]}" -n apps exec deploy/llama -c llama-swap -- /app/venv/bin/python -I -S -c '
import json, urllib.request
body = {
    "model": "qwen3.8-27b", "messages": [{"role": "user", "content": "What is 17 plus 25? Reply with only the number."}],
    "max_tokens": 16, "temperature": 0, "reasoning_effort": "none",
}
req = urllib.request.Request("http://127.0.0.1:5890/v1/chat/completions",
    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=8) as response:
    data = json.load(response)
assert data["choices"][0]["message"]["content"].strip() == "42", data
print(json.dumps({"model": data["model"], "content": "42", "usage": data.get("usage")}))
'
}

restore() {
  if [[ ! -f $out/claim.json || -f $out/restored.json ]]; then
    uv run --no-project python "$here/image_retention.py" restore "$out"
    return 0
  fi
  local live owner_id expected_uid rc=0 patch deadline=$((SECONDS+540))
  owner_id=$(jq -er .owner "$out/claim.json")
  expected_uid=$(jq -er .uid "$out/claim.json")
  live=$(deployment)
  [[ $(jq -er '.metadata.uid' <<<"$live") == "$expected_uid" ]] || die "Production Deployment UID changed"
  local observed_owner
  observed_owner=$(jq -r --arg owner "$owner" '.metadata.annotations[$owner] // ""' <<<"$live")
  if [[ $observed_owner == "$owner_id" || -f $out/.claimed ]]; then
    stop_arm candidate || rc=1
    stop_arm baseline || rc=1
    (( rc == 0 )) || die "Benchmark task cleanup failed. Leaving the Deployment paused until the GPU is free."
  fi
  if [[ $observed_owner == "$owner_id" ]]; then
    patch=$(jq -c --arg reconcile "$reconcile" --arg owner "$owner" '
      def ap($key): "/metadata/annotations/" + ($key | gsub("~"; "~0") | gsub("/"; "~1"));
      [
        {op:"test",path:"/metadata/uid",value:.uid},
        {op:"test",path:ap($owner),value:.owner},
        {op:"test",path:ap($reconcile),value:"disabled"},
        {op:"replace",path:"/spec/replicas",value:.replicas},
        (if .reconcile == null then {op:"remove",path:ap($reconcile)}
         else {op:"replace",path:ap($reconcile),value:.reconcile} end),
        {op:"remove",path:ap($owner)}
      ]
    ' "$out/claim.json")
    "${k[@]}" -n apps patch deploy llama --type=json -p "$patch"
  elif [[ -n $observed_owner ]]; then
    die "Deployment is owned by another trial"
  else
    # A timed-out claim could have failed before changing production, or a
    # previous restore could already have released the owner annotation.
    jq -e --arg reconcile "$reconcile" --slurpfile claim "$out/claim.json" '
      .spec.replicas == $claim[0].replicas and .metadata.annotations[$reconcile] == $claim[0].reconcile
    ' <<<"$live" >/dev/null || die "Deployment ownership was lost and the original state does not match"
  fi
  rm -f -- "$out/.trial-api-key" "$out/.claimed"
  echo "Waiting for the original Deployment and direct inference"
  while (( SECONDS < deadline )); do
    if live=$(deployment) && jq -e --arg owner "$owner" --arg reconcile "$reconcile" \
        --slurpfile claim "$out/claim.json" '
      .metadata.uid == $claim[0].uid and .spec.replicas == $claim[0].replicas and
      .metadata.annotations[$owner] == null and .metadata.annotations[$reconcile] == $claim[0].reconcile and
      .status.availableReplicas == $claim[0].replicas and
      .status.observedGeneration == .metadata.generation
    ' <<<"$live" >/dev/null && check_inference > "$out/.restored.json.tmp" 2>>"$out/restore-check.log"; then
      mv "$out/.restored.json.tmp" "$out/restored.json"
      uv run --no-project python "$here/image_retention.py" restore "$out"
      cat "$out/restored.json"
      return 0
    fi
    sleep 10
  done
  die "Deployment restore could not be verified by the deadline"
}

claim() {
  local live=$1 patch owner_id=$INVOCATION_ID
  jq -c --arg owner "$owner_id" --arg reconcile "$reconcile" '{
    owner:$owner, uid:.metadata.uid, resourceVersion:.metadata.resourceVersion,
    replicas:.spec.replicas, reconcile:.metadata.annotations[$reconcile]
  }' <<<"$live" > "$out/.claim.json.tmp"
  mv "$out/.claim.json.tmp" "$out/claim.json"
  patch=$(jq -c --arg reconcile "$reconcile" --arg owner "$owner" '
    def ap($key): "/metadata/annotations/" + ($key | gsub("~"; "~0") | gsub("/"; "~1"));
    [
      {op:"test",path:"/metadata/resourceVersion",value:.resourceVersion},
      {op:"add",path:ap($reconcile),value:"disabled"},
      {op:"add",path:ap($owner),value:.owner},
      {op:"replace",path:"/spec/replicas",value:0}
    ]
  ' "$out/claim.json")
  "${k[@]}" -n apps patch deploy llama --type=json -p "$patch"
  touch "$out/.claimed"
  local deadline=$((SECONDS+90)) pods
  until (( SECONDS >= deadline )); do
    pods=$("${k[@]}" -n apps get pods -l "$selector" -o json)
    if jq -e 'all(.items[]; .status.phase == "Failed" or .status.phase == "Succeeded")' <<<"$pods" >/dev/null && gpu_empty; then
      return 0
    fi
    sleep 3
  done
  die "Production did not release the GPU within 90 seconds"
}

benchmark_arm() {
  local arm=$1 port=$2 deadline=$((SECONDS+420)) launcher rc=0
  echo "Starting $arm on 127.0.0.1:$port"
  "$here/../launch-arm.sh" "$arm" >"$out/$arm-server.log" 2>&1 &
  launcher=$!
  while ! curl -fsS --max-time 3 -H 'Authorization: Bearer sparsekv-bench-local' \
      "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
    kill -0 "$launcher" 2>/dev/null || die "$arm exited during startup. See $out/$arm-server.log"
    (( SECONDS < deadline )) || die "$arm readiness timed out. See $out/$arm-server.log"
    sleep 5
  done
  curl -fsS --max-time 5 -H 'Authorization: Bearer sparsekv-bench-local' \
    "http://127.0.0.1:$port/v1/models" |
    jq -e 'any(.data[]; .id == "qwen3.8-27b")' >/dev/null || die "$arm did not advertise the benchmark model"
  "$here/run.sh" "http://127.0.0.1:$port/v1" qwen3.8-27b "$out/$arm" "$trial_id" "$out/.trial-api-key" ||
    rc=$?
  printf '%s\n' "$rc" > "$out/$arm-exit"
  stop_arm "$arm" || die "Could not stop $arm"
  wait "$launcher" || true
  deadline=$((SECONDS+15))
  until gpu_empty; do
    (( SECONDS < deadline )) || die "$arm did not release the GPU"
    sleep 1
  done
  disk_ok || die "Disk pressure appeared during $arm"
}

case "$mode" in
  --preflight) preflight ;;
  --execute)
    [[ ${SPARSEKV_BENCH_UNIT:-} == "$unit" && -n ${INVOCATION_ID:-} ]] || die "Use start.sh --execute"
    [[ -d $out && ! -e $out/claim.json ]] || die "Expected a new benchmark output directory"
    umask 077
    preflight
    trial_id=$INVOCATION_ID
    if [[ -n ${SPARSEKV_REUSE_BASELINE:-} ]]; then
      [[ $(cat "$SPARSEKV_REUSE_BASELINE/baseline-exit") == 0 ]] || die "Previous baseline failed"
      cp -a "$SPARSEKV_REUSE_BASELINE/baseline" "$out/baseline"
      cp "$SPARSEKV_REUSE_BASELINE/baseline-exit" "$out/baseline-exit"
      printf '%s\n' "$SPARSEKV_REUSE_BASELINE" > "$out/baseline-source.txt"
      trial_id=$(jq -er .owner "$SPARSEKV_REUSE_BASELINE/claim.json")
    fi
    uv run --no-project python "$here/image_retention.py" pin "$out"
    claim "$live"
    (( $(available_kib) >= 30 * 1024 * 1024 )) || die "Less than 30 GiB host RAM available after handoff"
    printf '%s\n' sparsekv-bench-local > "$out/.trial-api-key"
    nvidia-smi --query-gpu=timestamp,temperature.gpu,utilization.gpu,power.draw,clocks.gr,clocks.mem,memory.used \
      --format=csv -l 5 >"$out/gpu.csv" &
    telemetry=$!
    owner_pid=$$
    (
      while sleep 1; do
        available=$(available_kib)
        printf '%s %s\n' "$(date -Iseconds)" "$available" >> "$out/memory-kib.log"
        if (( available < 3 * 1024 * 1024 )); then
          echo "Host RAM below 3 GiB; aborting benchmark for restoration" >&2
          kill -TERM "$owner_pid"
          exit 1
        fi
      done
    ) &
    memory_monitor=$!
    trap 'kill "$telemetry" "$memory_monitor" 2>/dev/null || true' EXIT
    if [[ -z ${SPARSEKV_REUSE_BASELINE:-} ]]; then
      benchmark_arm baseline 5890
    fi
    benchmark_arm candidate 18091
    baseline_status=$(cat "$out/baseline-exit")
    candidate_status=$(cat "$out/candidate-exit")
    exit "$(( baseline_status > candidate_status ? baseline_status : candidate_status ))"
    ;;
  --restore)
    [[ -d $out ]] || die "Expected the benchmark output directory"
    umask 077
    restore
    ;;
  *) die "Usage: $0 --preflight | --execute OUTPUT_DIR | --restore OUTPUT_DIR" ;;
esac

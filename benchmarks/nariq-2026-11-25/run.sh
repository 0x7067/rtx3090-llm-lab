#!/usr/bin/env bash
# Run Qwen3.8-27B NariQ on an NVIDIA GPU with one command (Linux x86-64): downloads the nari-llama.cpp binaries
# and CUDA runtime, the GGUF and the vision projector once, checks each file's sha256, and starts llama-server
# with an OpenAI-compatible API and a chat page at http://127.0.0.1:8080.
#
#   bash run.sh                 # the 3.0 bpw file, with vision, the longest context the GPU's free memory holds
#   bash run.sh 2.5             # the 2.5 bpw file (2.0: the 2.0 bpw one); 3, 3.0bpw, 3bit and the like work too
#   bash run.sh 3.0 --docker    # the same through the Docker image (needs the NVIDIA Container Toolkit)
#   bash run.sh 3.0 --no-vision -- -np 4 -c 131072    # arguments after -- go to llama-server unchanged
#
# Options: --port N (8080), --host ADDR (127.0.0.1), --no-vision, --docker, --download-only.
# Files go to ~/.cache/nari-llama.cpp (about 13 GB with the 3.0 file); delete that folder to remove them.
# Environment: TRELLIS_HOME (another download folder); TRELLIS_RELEASE_URL, TRELLIS_MODEL_URL, TRELLIS_IMAGE and
# TRELLIS_TAG (mirrors and tests); HF_TOKEN (sent to https://huggingface.co only, for gated or private model
# repositories). Needs curl, tar, sha256sum, an NVIDIA driver of 570 or newer, glibc 2.34 or newer, libstdc++ with
# GLIBCXX_3.4.30 and libgomp (Ubuntu 22.04 or newer, Debian 12 or newer; on RHEL 9 or Amazon Linux 2023 use --docker).
# Without a GPU it runs on the CPU, slowly.
set -euo pipefail

if [ "${BASH_VERSINFO[0]}" -lt 4 ] || [ "$(uname -s)" != Linux ] || [ "$(uname -m)" != x86_64 ]; then
    echo "run.sh requires Linux x86-64 and Bash 4+. Other platforms: build from source," \
         "https://github.com/nari-labs/nari-llama.cpp#build-from-source" >&2
    exit 2
fi

TAG="${TRELLIS_TAG:-b12431-nariv0}"   # the release tag (nari-labs/nari-llama.cpp)
RELEASE_URL="${TRELLIS_RELEASE_URL:-https://github.com/nari-labs/nari-llama.cpp/releases/download/$TAG}"
MODEL_URL="${TRELLIS_MODEL_URL:-https://huggingface.co/nari-labs/Qwen3.8-27B-NariQ-GGUF/resolve/main}"
IMAGE="${TRELLIS_IMAGE:-ghcr.io/nari-labs/nari-llama.cpp:$TAG}"
HOME_DIR="${TRELLIS_HOME:-$HOME/.cache/nari-llama.cpp}"
CUDA=12.8

# file, size in bytes and sha256 per budget; the layout of these files is frozen
declare -A FILE=([3.0]=Qwen3.8-27B-NariQ-3.0bpw.gguf [2.5]=Qwen3.8-27B-NariQ-2.5bpw.gguf [2.0]=Qwen3.8-27B-NariQ-2.0bpw.gguf)
declare -A SIZE=([3.0]=10123626592 [2.5]=8443119712 [2.0]=6753954304)
declare -A SHA=([3.0]=1eaad934aa05579b8eba47ed9b26450392f9875fc2354efe8cfa8b29be8f4419
                [2.5]=2445372e6de20620b940fbfab53aa718af9ba741f202da30cd2f02b5df78d05c
                [2.0]=3a9e4955329dba8ba0be5509a9552220e5ccdae2f66bcc72c21c4580cd49bbb2)
MMPROJ=Qwen3.8-27B-NariQ-mmproj-BF16.gguf
MMPROJ_SHA=8832425f593068d3278e08eda4c1591072c7a2d8dc90f21e4f86394e4b2eb002
MMPROJ_SIZE=931145760

budget=3.0 port=8080 host=127.0.0.1 vision=1 docker=0 download_only=0
extra=()
while [ $# -gt 0 ]; do
    case "$1" in
        3.0|2.5|2.0) budget="$1" ;;
        3|2) budget="$1.0" ;;
        3.0bpw|2.5bpw|2.0bpw) budget="${1%bpw}" ;;
        3bit|2.5bit|2bit) budget="${1%bit}"; [ "$budget" = 3 ] && budget=3.0; [ "$budget" = 2 ] && budget=2.0 ;;
        --port|--host)
            [ $# -ge 2 ] && [ -n "$2" ] || { echo "run.sh: $1 requires a value" >&2; exit 2; }
            if [ "$1" = --port ]; then port="$2"; else host="$2"; fi
            shift ;;
        --port=?*) port="${1#*=}" ;;
        --host=?*) host="${1#*=}" ;;
        --no-vision) vision=0 ;;
        --docker) docker=1 ;;
        --download-only) download_only=1 ;;
        --) shift; extra=("$@"); break ;;
        -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
        *) echo "run.sh: unknown argument '$1' (see --help)" >&2; exit 2 ;;
    esac
    shift
done
case "$port" in ''|*[!0-9]*) echo "run.sh: --port needs a number, not '$port'" >&2; exit 2 ;; esac
for tool in curl tar sha256sum; do
    command -v "$tool" > /dev/null || { echo "run.sh: needs $tool" >&2; exit 2; }
done

if command -v nvidia-smi > /dev/null && gpu="$(nvidia-smi --query-gpu=name,driver_version,memory.total \
        --format=csv,noheader 2> /dev/null | head -n 1)" && [ -n "$gpu" ]; then
    driver="${gpu#*, }"; driver="${driver%%.*}"
    echo "GPU: $gpu"
    if [ "$driver" -lt 570 ]; then
        echo "run.sh: NVIDIA driver $driver is too old for CUDA $CUDA; install 570 or newer" >&2
        exit 2
    fi
    # the weights plus a short context; below this llama-server keeps some layers on the CPU, which is much slower
    mem="${gpu##*, }"; mem="${mem%% *}"
    want=$(( ${SIZE[$budget]} / 1048576 + 2560 ))
    case "$mem" in *[!0-9]*|'') ;; *)
        [ "$mem" -lt "$want" ] && echo "run.sh: the $budget file wants about $(( want / 1024 + 1 )) GB of GPU memory;" \
            "this GPU has $(( mem / 1024 )) GB, so part of the model will run on the CPU (try 2.5 or 2.0)" >&2 ;;
    esac
else
    echo "run.sh: no NVIDIA GPU found; llama-server will run on the CPU, which is slow for a 27B model" >&2
fi

hf_auth() {  # url: the Authorization header for a Hugging Face URL when HF_TOKEN is set (curl reads it from a pipe,
    # so the token is not on a command line; curl does not send it on to the hosts a download redirects to)
    case "$1" in https://huggingface.co/*|https://hf.co/*)
        [ -n "${HF_TOKEN:-}" ] && printf 'Authorization: Bearer %s\n' "$HF_TOKEN" ;; esac
    return 0
}

fetch() {  # url, path, sha256[, size]: download once (resuming a partial file), then check
    local url="$1" path="$2" sha="$3" size="${4:-}" have=0
    if [ -f "$path" ] && [ -f "$path.ok" ]; then return 0; fi
    mkdir -p "$(dirname "$path")"
    [ -f "$path.part" ] && have="$(stat -c %s "$path.part")"
    if [ -n "$size" ] && [ "$have" -gt "$size" ]; then rm -f "$path.part"; have=0; fi
    if [ -z "$size" ] || [ "$have" -lt "$size" ]; then
        echo "downloading $(basename "$path")"
        if ! curl -fL -# --retry 5 --retry-delay 3 -C - -H @<(hf_auth "$url") -o "$path.part" "$url"; then
            # resuming a part that is already complete fails (HTTP 416): keep it if it checks out
            if [ ! -f "$path.part" ] || [ "$(sha256sum "$path.part" | cut -d' ' -f1)" != "$sha" ]; then
                echo "run.sh: could not download $(basename "$path") (see the curl error above); run again to resume" >&2
                exit 1
            fi
        fi
    fi
    echo "checking $(basename "$path")"
    if [ "$(sha256sum "$path.part" | cut -d' ' -f1)" != "$sha" ]; then
        rm -f "$path.part"
        echo "run.sh: $(basename "$path") does not match its sha256; deleted, run again to download it again" >&2
        exit 1
    fi
    mv "$path.part" "$path" && touch "$path.ok"
}

# free space for what is still to download (the binaries: about 0.9 GB of archives, kept, and 1.2 GB extracted)
mkdir -p "$HOME_DIR"
need=0
[ -f "$HOME_DIR/models/${FILE[$budget]}.ok" ] || need=$(( need + ${SIZE[$budget]} ))
[ "$vision" = 1 ] && [ ! -f "$HOME_DIR/models/$MMPROJ.ok" ] && need=$(( need + MMPROJ_SIZE ))
[ "$docker" = 0 ] && [ ! -x "$HOME_DIR/bin/$TAG/llama-server" ] && need=$(( need + 2200000000 ))
free="$(df -P -B1 "$HOME_DIR" 2> /dev/null | awk 'NR == 2 { print $4 }')"
if [ -n "$free" ] && [ "$need" -gt "$free" ]; then
    echo "run.sh: needs about $(( need / 1000000000 + 1 )) GB free in $HOME_DIR but has $(( free / 1000000000 )) GB;" \
         "free some space or set TRELLIS_HOME to another folder" >&2
    exit 1
fi
echo "files: $HOME_DIR"

fetch_models() {
    fetch "$MODEL_URL/${FILE[$budget]}" "$HOME_DIR/models/${FILE[$budget]}" "${SHA[$budget]}" "${SIZE[$budget]}"
    [ "$vision" = 0 ] || fetch "$MODEL_URL/$MMPROJ" "$HOME_DIR/models/$MMPROJ" "$MMPROJ_SHA" "$MMPROJ_SIZE"
}

# llama.cpp starts a thread per core of the whole machine, even where this process may use fewer CPUs (a cpuset or a
# CPU quota, as in containers); the coded embedding's CPU ops then wait on descheduled threads at every token, which
# slowed the 2-bit file's generation 4x on some hosts. Pass the CPUs this process has when they are fewer.
cpus="$(nproc)"
if [ -r /sys/fs/cgroup/cpu.max ] && read -r quota period < /sys/fs/cgroup/cpu.max && [ "$quota" != max ]; then
    quota=$(( (quota + period - 1) / period ))
    [ "$quota" -lt "$cpus" ] && cpus="$quota"
fi
threads=()
case " ${extra[*]} " in
    *" -t "*|*" --threads "*) ;;
    *) [ "$cpus" -lt "$(getconf _NPROCESSORS_ONLN)" ] && threads=(-t "$cpus") ;;
esac

# the GGUFs carry the model's sampling settings except min_p, which llama-server would otherwise set to 0.05
sampling=(--min-p 0)

if [ "$docker" = 1 ]; then
    fetch_models
    [ "$download_only" = 1 ] && { docker pull "$IMAGE"; exit 0; }
    args=(-m "/models/${FILE[$budget]}" "${threads[@]}" "${sampling[@]}")
    [ "$vision" = 1 ] && args+=(--mmproj "/models/$MMPROJ" --fit-target 2560)
    echo "starting $IMAGE on http://$host:$port"
    exec docker run --rm --gpus all -p "$host:$port:8080" -v "$HOME_DIR/models:/models:ro" "$IMAGE" \
        "${args[@]}" --host 0.0.0.0 --port 8080 "${extra[@]}"
fi

bin="$HOME_DIR/bin/$TAG"
if [ ! -x "$bin/llama-server" ]; then
    name="nari-llama.cpp-$TAG-bin-linux-cuda-$CUDA-x64"
    cudart="cudart-nari-llama.cpp-$TAG-linux-cuda-$CUDA-x64"
    mkdir -p "$HOME_DIR/downloads"
    curl -fsSL --retry 5 -o "$HOME_DIR/downloads/SHA256SUMS-$TAG" "$RELEASE_URL/SHA256SUMS" ||
        { echo "run.sh: could not download $RELEASE_URL/SHA256SUMS" >&2; exit 1; }
    for archive in "$name" "$cudart"; do
        sha="$(sed -n "s/^\([0-9a-f]\{64\}\)  $archive.tar.gz$/\1/p" "$HOME_DIR/downloads/SHA256SUMS-$TAG")"
        [ -n "$sha" ] || { echo "run.sh: $archive.tar.gz is not in the release's SHA256SUMS" >&2; exit 1; }
        fetch "$RELEASE_URL/$archive.tar.gz" "$HOME_DIR/downloads/$archive.tar.gz" "$sha"
    done
    rm -rf "$bin.part" && mkdir -p "$bin.part"
    tar -xzf "$HOME_DIR/downloads/$name.tar.gz" -C "$bin.part" --strip-components=1
    tar -xzf "$HOME_DIR/downloads/$cudart.tar.gz" -C "$bin.part" --strip-components=1
    rm -rf "$bin" && mv "$bin.part" "$bin"
    # check that the binaries run here before the model download (llama-server --version needs no GPU)
    if ! "$bin/llama-server" --version > /dev/null 2>&1; then
        echo "run.sh: the release binaries do not run on this system; they need glibc 2.34 or newer, libstdc++ with" \
             "GLIBCXX_3.4.30 and libgomp. Try --docker. The error:" >&2
        "$bin/llama-server" --version >&2 || true
        rm -rf "$bin"
        exit 1
    fi
fi
fetch_models
[ "$download_only" = 1 ] && exit 0

args=(-m "$HOME_DIR/models/${FILE[$budget]}" "${threads[@]}" "${sampling[@]}")
# the projector is loaded after the context is fitted to free memory: keep room for it and its image batch
[ "$vision" = 1 ] && args+=(--mmproj "$HOME_DIR/models/$MMPROJ" --fit-target 2560)
echo "starting llama-server ($budget) on http://$host:$port"
exec "$bin/llama-server" "${args[@]}" --host "$host" --port "$port" "${extra[@]}"

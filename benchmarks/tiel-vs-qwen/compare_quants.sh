#!/usr/bin/env bash
# Does IQ4_XS give up quality against Q4_K_S on this model?
#
# The model card ranks tiers by size and prose ("4-bit quality with the most
# context headroom" against "tight 4-bit") but never measures them against each
# other, and neither is the tier it benchmarked. This measures it.
#
# Perplexity on a fixed corpus, identical settings, one model after the other.
# Lower is closer to the distribution the unquantized model would produce.
#
# What this is NOT: KL divergence against the BF16 reference, which is the
# rigorous way to score quantization damage. That needs the 70 GB original,
# which does not fit here. Comparing two quants' perplexity on the same text is
# a weaker but real signal -- it is directional, and a tier gap large enough to
# matter shows up in it.
#
# The corpus leans on code because Tiel is a coding model and that is the
# distribution we care about; a wikitext number would be more comparable to
# other people's results and less relevant to this decision.
#
# Usage: compare_quants.sh MODEL_A [MODEL_B ...]
set -uo pipefail
cd "$(dirname "$0")"
MODELS=/data/buttercup_6tb/k3s/vllm-trial/models
IMG=ghcr.io/ggml-org/llama.cpp@sha256:851b3b87f89bda98f2ad416e71ab91b6e88be1807502a963937f1d21f3b8555d
CORPUS=/tmp/ppl_corpus.txt
CTX=${CTX:-512}
OUT=${OUT:-results_quant_ppl.json}

restore() {
  docker rm -f pplrun >/dev/null 2>&1 || true
  echo "== restoring deployment"
  ./restore_qwen.sh || echo "RESTORE FAILED - check kubectl -n apps get pods" >&2
}
trap restore EXIT

echo "== building the corpus"
python3 - "$CORPUS" <<'PY'
import json, glob, sys, os
out = open(sys.argv[1], "w")
n = 0
# HumanEval prompt + canonical solution: on-domain, and already in this repo.
for line in open("HumanEval.jsonl"):
    d = json.loads(line)
    out.write(d["prompt"] + d["canonical_solution"] + "\n\n")
    n += 1
# This repo's own Python and shell, for a second flavour of real code.
for pat in ("*.py", "*.sh"):
    for f in sorted(glob.glob(pat)):
        try:
            out.write(open(f).read() + "\n\n")
        except Exception:
            pass
# Prose, so the score is not purely code.
for f in ("REPORT.md", "PARALLELIZATION.md"):
    if os.path.exists(f):
        out.write(open(f).read() + "\n\n")
out.close()
print(f"   {n} HumanEval problems plus repo sources -> {os.path.getsize(sys.argv[1])} bytes")
PY

echo "== suspending flux (must precede the scale-down)"
flux suspend kustomization apps
kubectl -n apps scale deploy llama --replicas=0
kubectl -n apps wait --for=delete pod -l app.kubernetes.io/name=llama --timeout=300s || true
for _ in $(seq 1 60); do
  [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 500 ] && break
  sleep 3
done

rows=""
for m in "$@"; do
  echo
  echo "== perplexity: $m"
  docker rm -f pplrun >/dev/null 2>&1 || true
  # No --mmproj and no server: this is the base model's own distribution.
  docker run --rm --name pplrun --gpus all --user 1000:1000 \
    -v "$MODELS:/models:ro" -v /tmp:/host_tmp:ro \
    --entrypoint /app/llama-perplexity "$IMG" \
    -m "/models/$m" -f /host_tmp/$(basename "$CORPUS") \
    -ngl 99 -c "$CTX" --no-mmap 2>&1 | tee "/tmp/ppl_$m.log" | tail -3
  ppl=$(grep -oE 'Final estimate: PPL = [0-9.]+' "/tmp/ppl_$m.log" | tail -1 | grep -oE '[0-9.]+$')
  size=$(stat -c%s "$MODELS/$m")
  echo "   -> PPL ${ppl:-unknown}"
  rows="$rows{\"model\":\"$m\",\"bytes\":$size,\"ppl\":${ppl:-null}},"
done

printf '{"ctx":%s,"corpus_bytes":%s,"results":[%s]}\n' \
  "$CTX" "$(stat -c%s "$CORPUS")" "${rows%,}" > "$OUT"
echo
echo "wrote $OUT"
python3 -c "
import json
d=json.load(open('$OUT'))
rs=[r for r in d['results'] if r['ppl']]
if len(rs)>1:
    best=min(rs,key=lambda r:r['ppl'])
    print()
    for r in sorted(rs,key=lambda r:r['ppl']):
        delta=(r['ppl']/best['ppl']-1)*100
        print(f\"  {r['model']:<44} {r['bytes']/1e9:5.2f} GB  PPL {r['ppl']:.4f}  {delta:+.2f}%\")
"
echo "QUANT_COMPARE_DONE"

#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
cd "$(git rev-parse --show-toplevel)"
scratch=$(mktemp -d)
finish() {
  result=$?
  printf '%s\n' "$result" > "$here/qualification/checks.exit"
  rm -r -- "$scratch"
}
trap finish EXIT
scripts/test.sh
scripts/check-supply-chain.sh
kubectl kustomize k8s/workloads > "$scratch/workloads.yaml"
kubectl kustomize k8s/infra > "$scratch/infra.yaml"
scripts/check-supply-chain.sh --rendered-workloads "$scratch/workloads.yaml"
bash -n "$here"/*.sh
shellcheck "$here"/*.sh
uv run --no-project python - "$here" <<'PY'
import ast
from pathlib import Path
import sys
for path in Path(sys.argv[1]).glob('*.py'):
    ast.parse(path.read_text(), filename=str(path))
print('Python sources parse successfully')
PY
uv run --locked --project tools/supply-chain python - <<'PY'
from pathlib import Path
import subprocess
import yaml
path = 'k8s/workloads/apps/llama/swap-configmap.yaml'
def config(data):
    return yaml.safe_load(yaml.safe_load(data)['data']['config.yaml'])
old = config(subprocess.check_output(['git', 'show', 'HEAD:' + path], text=True))
new = config(Path(path).read_text())
resident = 'qwen3.8-27b-hyperqwen'
assert old['models'].keys() == new['models'].keys()
for model in old['models']:
    if model != resident:
        assert old['models'][model] == new['models'][model], model
for key in old:
    if key != 'models':
        assert old[key] == new[key], key
for key in ('name', 'ttl', 'proxy', 'useModelName', 'metadata'):
    assert old['models'][resident][key] == new['models'][resident][key], key
print('All other model entries and routing settings are unchanged')
PY

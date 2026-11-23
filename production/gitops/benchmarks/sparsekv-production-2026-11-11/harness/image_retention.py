"""Temporarily pin only this deployment's images against kubelet image GC."""
import json
from pathlib import Path
import subprocess
import sys
import time
import re

KEY = 'io.cri-containerd.pinned'
CTR = ['sudo', '-n', 'k3s', 'ctr', '-n', 'k8s.io', 'images']

def call(args):
    return subprocess.check_output(args, text=True, timeout=30)

def main(mode, output):
    receipt = output / 'image-pins.json'
    if mode == 'restore':
        if not receipt.exists():
            return
        for row in json.loads(receipt.read_text()):
            call([*CTR, 'label', row['name'], KEY + '=' + row['previous']])
        (output / 'image-pins-restored.txt').write_text('Original pin labels restored.\n')
        return
    assert mode == 'pin' and not receipt.exists()
    deployment = json.loads((Path(__file__).resolve().parents[1] / 'production/deployment.json').read_text())
    spec = deployment['spec']['template']['spec']
    digests = {c['image'].split('@')[1] for c in spec['containers'] + spec['initContainers']}
    listing = call([*CTR, 'list']).splitlines()[1:]
    lines = {line.split()[0]: line for line in listing if line.strip()}
    names = list(lines)
    selected = [name for name in names if any(d in lines[name] for d in digests)]
    assert all(any(name.endswith('@' + d) for name in selected) for d in digests), 'A restoration image is absent'
    rows = []
    for name in selected:
        label = re.search(re.escape(KEY) + r'=([^,\s]+)', lines[name])
        previous = label.group(1) if label else ''
        rows.append({'name': name, 'previous': previous})
        receipt.write_text(json.dumps(rows, indent=2) + '\n')
        call([*CTR, 'label', name, KEY + '=pinned'])
    for _ in range(10):
        images = json.loads(call(['sudo', '-n', 'k3s', 'crictl', 'images', '-o', 'json']))['images']
        if all(any(image.get('pinned') and any(ref.endswith('@' + d) for ref in image.get('repoDigests', [])) for image in images) for d in digests):
            (output / 'image-pins-verified.txt').write_text('CRI reports all three deployment image digests pinned.\n')
            return
        time.sleep(1)
    raise RuntimeError('CRI did not confirm image retention')

if __name__ == '__main__':
    main(sys.argv[1], Path(sys.argv[2]))

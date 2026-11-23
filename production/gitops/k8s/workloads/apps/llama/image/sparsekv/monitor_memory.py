"""Bounded host/process/cgroup telemetry for an approved SparseKV rollout."""
import argparse
import json
from pathlib import Path
import time


def sample():
    mem = {line.split(':', 1)[0]: int(line.split()[1]) * 1024
           for line in Path('/proc/meminfo').read_text().splitlines()}
    groups = {}
    cgroups = {}
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            name = (proc / 'comm').read_text().strip()
            if name not in ('VLLM::EngineCor', 'vllm', 'workerd'):
                continue
            rss = next(int(line.split()[1]) * 1024
                       for line in (proc / 'status').read_text().splitlines() if line.startswith('VmRSS:'))
            group = groups.setdefault(name, {'processes': 0, 'rss_bytes': 0})
            group['processes'] += 1
            group['rss_bytes'] += rss
            relative = next(line.split(':', 2)[2] for line in
                            (proc / 'cgroup').read_text().splitlines() if line.startswith('0::'))
            if name == 'workerd' or relative in cgroups:
                continue
            folder = Path('/sys/fs/cgroup') / relative.lstrip('/')
            stats = dict(line.split() for line in (folder / 'memory.stat').read_text().splitlines())
            cgroups[relative] = {
                'current_bytes': int((folder / 'memory.current').read_text()),
                'anon_bytes': int(stats.get('anon', 0)), 'file_bytes': int(stats.get('file', 0)),
                'shmem_bytes': int(stats.get('shmem', 0)),
                'events': dict(line.split() for line in (folder / 'memory.events').read_text().splitlines()),
            }
        except (OSError, StopIteration, ValueError):
            continue  # Processes may exit between /proc reads.
    return {'time': time.time(), 'host_available_bytes': mem['MemAvailable'],
            'swap_free_bytes': mem['SwapFree'], 'process_groups': groups, 'model_cgroups': cgroups}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=1800)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 3600:
        parser.error('--seconds must be 1..3600')
    deadline = time.monotonic() + args.seconds
    with args.output.open('x') as output:
        while time.monotonic() < deadline:
            output.write(json.dumps(sample()) + '\n')
            output.flush()
            time.sleep(2)

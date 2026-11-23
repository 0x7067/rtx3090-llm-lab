"""Launch-time CUDA flight recorder for the resident SparseKV process.

Only the target-process path forwards termination to the model. Collection
failures preserve evidence and leave inference running.
"""
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.request

NSYS = '/opt/nvidia/nsight-systems-cli/2026.4.1/target-linux-x64/nsys'
ROOT = Path('/cache/sparsekv-cuda-traces')
WINDOW_SECONDS = 30
KEEP_WINDOWS = 12
BUDGET_BYTES = 2 * 1024**3
STALL_SECONDS = 240


def write_json(path, value):
    temporary = path.with_suffix('.new')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def size(path):
    total = 0
    for item in path.rglob('*'):
        try:
            if item.is_file():
                total += item.stat().st_size
        except FileNotFoundError:
            pass  # Nsight moves temporary data while finalizing reports.
    return total


def prune(run):
    reports = sorted(run.glob('window-*.nsys-rep'))
    for report in reports[:-KEEP_WINDOWS]:
        report.unlink()
    # This budget includes preserved earlier runs and current raw data. Frozen
    # runs and raw data are evidence; a full budget stops new collection.
    reports = sorted(run.glob('window-*.nsys-rep'))
    used = size(ROOT)
    while used > BUDGET_BYTES and len(reports) > 2:
        old = reports.pop(0)
        used -= old.stat().st_size
        old.unlink()
    return used


def progress():
    with urllib.request.urlopen('http://127.0.0.1:5890/metrics', timeout=2) as response:
        lines = response.read().decode().splitlines()
    values = {}
    for line in lines:
        match = re.match(r'(vllm:(?:num_requests_running|prompt_tokens_total|generation_tokens_total))(?:\{.*\})?\s+(\S+)', line)
        if match:
            name, value = match.groups()
            values[name] = values.get(name, 0) + float(value)
    required = ('vllm:num_requests_running', 'vllm:prompt_tokens_total', 'vllm:generation_tokens_total')
    if not all(key in values for key in required):
        raise ValueError('progress metrics unavailable')
    return values[required[0]], values[required[1]] + values[required[2]]


def target_alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1][0] != 'Z'
    except FileNotFoundError:
        return False


def target(run, controller, command):
    # The model owns a process group so llama-swap shutdown reaches its workers.
    if os.getpgrp() != os.getpid():
        os.setsid()
    # Keep normal pod logging independent of Nsight's report stdout capture.
    for fd in (1, 2):
        source = os.open(f'/proc/{controller}/fd/{fd}', os.O_WRONLY)
        os.dup2(source, fd)
        os.close(source)
    write_json(run / 'target.json', {'pid': os.getpid(), 'pgid': os.getpgrp()})
    os.execv(command[0], command)


def main(command):
    os.umask(0o077)
    ROOT.mkdir(parents=True, exist_ok=True)
    run = ROOT / (time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + f'-{os.getpid()}')
    run.mkdir()
    (run / 'raw').mkdir()
    link = ROOT / 'current.new'
    link.unlink(missing_ok=True)
    link.symlink_to(run.name)
    link.replace(ROOT / 'current')
    session = 'sparsekv-' + run.name
    env = dict(os.environ, TMPDIR=str(run / 'raw'))
    state = {'session': session, 'state': 'starting', 'window_seconds': WINDOW_SECONDS,
             'keep_windows': KEEP_WINDOWS, 'budget_bytes': BUDGET_BYTES,
             'stall_seconds': STALL_SECONDS, 'pid': os.getpid()}
    stopping = False
    frozen = False
    recording = False
    model_pid = None

    def save(**fields):
        state.update(fields, updated_at=time.time())
        write_json(run / 'status.json', state)

    def control(*args):
        # Kill only the control CLI on timeout, never the target or daemon.
        result = subprocess.run([NSYS, *args], env=env, capture_output=True,
                                text=True, timeout=20)
        if result.returncode:
            raise RuntimeError(f'nsys {args[0]}: {result.stderr[-2000:]} {result.stdout[-2000:]}')
        return result.stdout

    def freeze(reason, stop=True):
        nonlocal frozen, recording
        frozen = True
        (run / 'frozen').write_text(reason + '\n')
        save(state='frozen', reason=reason)
        if recording and stop:
            try:
                control('stop', '--session=' + session)
                recording = False
            except (subprocess.TimeoutExpired, RuntimeError) as error:
                save(finalization_error=str(error))
                return
        # Detach after a successful stop. Model stdout and lifetime remain owned
        # by this wrapper; shutdown never signals the model.
        if stop:
            try:
                control('shutdown', '--session=' + session, '--kill=none')
            except (subprocess.TimeoutExpired, RuntimeError) as error:
                save(detach_error=str(error))

    def terminate(signum, _frame):
        nonlocal stopping
        stopping = True
        if model_pid and target_alive(model_pid):
            os.killpg(model_pid, signum)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    save()
    launcher = subprocess.Popen([
        NSYS, 'launch', '--session-new=' + session, '--trace=cuda',
        '--cuda-event-trace=false', '--cuda-graph-trace=node',
        '--cuda-flush-interval=1000', '--trace-fork-before-exec=false',
        '--env-var=TMPDIR=/tmp', sys.executable, __file__, '--target',
        str(run), str(os.getpid()), *command,
    ], env=env)
    last_tokens = None
    last_progress = time.monotonic()
    next_window = 0.0
    window = 0
    try:
        while True:
            pidfile = run / 'target.json'
            if model_pid is None and pidfile.exists():
                model_pid = json.loads(pidfile.read_text())['pid']
                save(target_pid=model_pid)
                if stopping:
                    os.killpg(model_pid, signal.SIGTERM)
            if model_pid and not target_alive(model_pid):
                break
            if model_pid is None:
                if launcher.poll() is not None:
                    raise RuntimeError('Nsight launcher exited before starting model')
                time.sleep(0.2)
                continue
            if stopping or frozen:
                time.sleep(1)
                continue
            now = time.monotonic()
            try:
                running, tokens = progress()
                if running == 0 or tokens != last_tokens:
                    last_progress = now
                last_tokens = tokens
                save(running_requests=running, progress_tokens=tokens,
                     stalled_seconds=round(now - last_progress, 1))
                if running > 0 and now - last_progress >= STALL_SECONDS:
                    freeze('active requests without token progress for 240 seconds')
                    continue
            except (OSError, ValueError) as error:
                save(metrics_error=str(error))
            if (run / 'freeze').exists():
                freeze('manual capture requested')
                continue
            try:
                used = prune(run)
                if used >= BUDGET_BYTES:
                    freeze('trace storage budget reached')
                    continue
                if now >= next_window:
                    if recording:
                        control('stop', '--session=' + session)
                        recording = False
                    window += 1
                    output = run / f'window-{window:06d}'
                    control('start', '--session=' + session, '--sample=none',
                            '--cpuctxsw=none', '--discard-environment=true',
                            '--export=none', '--output=' + str(output))
                    recording = True
                    next_window = time.monotonic() + WINDOW_SECONDS
                    save(state='recording', window=window, report=str(output), bytes_used=used)
            except (subprocess.TimeoutExpired, RuntimeError, OSError) as error:
                # An unresponsive stop can be the hang itself. Preserve raw data
                # and issue no further profiler commands in this state.
                freeze(str(error), stop=False)
            time.sleep(2)
    finally:
        if not frozen:
            freeze('model exited')
        save(target_exited=True)
        try:
            launcher.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    return launcher.returncode or 0


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--target':
        target(Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4:])
    else:
        raise SystemExit(main(sys.argv[1:] or ['/bin/bash', '/opt/sparsekv/serve-sparse.sh']))

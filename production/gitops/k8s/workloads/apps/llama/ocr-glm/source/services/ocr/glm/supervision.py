"""Linux process adoption/reaping for the existing GLM service launcher.

Direct Popen children keep their normal wait/poll owner. Only adopted children
are reaped here, including CPU grandchildren whose worker was forcibly killed.
No model, package, queue, process-group or resource setting is changed.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import signal
import sys
import threading
import time


class AdoptedChildReaper:
    def __init__(self, lock, direct_children):
        self.lock=lock
        self.direct_children=direct_children
        self.closed=threading.Event()
        self.thread=None

    def start(self):
        if sys.platform!='linux':raise RuntimeError('GLM descendant supervision requires Linux')
        libc=ctypes.CDLL(None,use_errno=True)
        libc.prctl.argtypes=[ctypes.c_int,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_ulong]
        libc.prctl.restype=ctypes.c_int
        if libc.prctl(36,1,0,0,0)!=0:raise OSError(ctypes.get_errno(),'GLM subreaper setup failed')
        self.thread=threading.Thread(target=self._run,name='glm-child-reaper',daemon=True)
        self.thread.start()

    def _adopted(self):
        direct={p.pid for p in self.direct_children() if p is not None and p.returncode is None}
        children=set()
        for path in Path('/proc/self/task').glob('*/children'):
            try:children.update(int(pid) for pid in path.read_text().split())
            except (FileNotFoundError,ProcessLookupError):continue
        return children-direct

    def reap(self,kill=False):
        # The launcher uses this same lock around Popen plus attribute binding,
        # so a newly spawned direct child cannot be mistaken for an orphan.
        with self.lock:
            children=self._adopted()
            for pid in children:
                if kill:
                    try:
                        group=os.getpgid(pid)
                        if group!=os.getpgrp():os.killpg(group,signal.SIGKILL)
                        else:os.kill(pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                try:os.waitpid(pid,os.WNOHANG)
                except ChildProcessError:pass
            return self._adopted()

    def _run(self):
        while not self.closed.wait(.05):self.reap()

    def close(self,timeout=10):
        # API/native direct children have already stopped. Their remaining
        # descendants belong to this service and must not outlive the launcher.
        end=time.monotonic()+timeout
        while self.reap(kill=True):
            if time.monotonic()>=end:raise RuntimeError('GLM descendants did not drain')
            time.sleep(.05)
        self.closed.set()
        if self.thread is not None:self.thread.join(timeout=1)

"""Separate managed GLM API using the immutable image's existing credential bridge.

Native GPU inference, CPU SDK worker, namespace and database remain distinct
from the preserved Paddle service. The model manager owns this process tree.
"""
from __future__ import annotations
import importlib.util
import os
import signal
import subprocess
import threading
import time
import sys

sys.path.insert(0,'/opt/finance-ocr-glm')
from services.ocr.glm.supervision import AdoptedChildReaper

ROOT='/opt/finance-ocr-glm'
PYTHON='/opt/ocr-glm/runtime/venv/bin/python'
API=('127.0.0.1',9419)
spec=importlib.util.spec_from_file_location('pinned_ocr_front','/opt/ocr/launch.py')
front=importlib.util.module_from_spec(spec)
spec.loader.exec_module(front)
front.ROOT=ROOT;front.API=API;front.VLLM_HEALTH='http://127.0.0.1:8001/health'


def child_env(extra):
    env={
        'PATH':'/opt/ocr/node/bin:/usr/bin:/bin','LANG':'C.UTF-8',
        'HOME':'/dev/shm/ocr-glm-home','TMPDIR':'/dev/shm/ocr-glm-tmp',
        'LD_LIBRARY_PATH':'/opt/py312/lib:/opt/llama:/opt/llama/cuda:/usr/local/nvidia/lib64:/usr/local/nvidia/lib',
        'NVIDIA_VISIBLE_DEVICES':'all','NVIDIA_DRIVER_CAPABILITIES':'compute,utility',
        'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1','PYTHONUNBUFFERED':'1',
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','DO_NOT_TRACK':'1',
        'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
        'OCR_ENABLED':'real','OCR_PROVIDER':'glm','OCR_PYTHON':PYTHON,
        'OCR_RUNTIME_MANIFEST':'/etc/ocr-glm/manifest.json','OCR_PORT':str(API[1]),'OCR_LISTEN_HOST':API[0],
        'OCR_WORK_DIR':'/dev/shm/ocr-glm-work',
        'OCR_TOKEN_FILE':os.environ['OCR_TOKEN_FILE'],'OCR_NAMESPACE':os.environ['OCR_NAMESPACE'],
        'OCR_DATA_DIR':os.environ['OCR_DATA_DIR'],
    }
    env.update(extra)
    return env


class GlmService(front.Service):
    def __init__(self):
        super().__init__();self.lock=threading.Lock();self.stopping=False
        self.reaper=AdoptedChildReaper(self.lock,lambda:[self.api,self.vllm])

    def spawn(self,command,slot):
        with self.lock:
            if self.stopping:raise SystemExit('GLM startup was stopped')
            try:
                process=subprocess.Popen(command,cwd=ROOT,env=child_env({}),start_new_session=True,
                                        stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            except OSError:raise SystemExit('GLM child process could not start') from None
            setattr(self,slot,process)
            return process

    def start(self):
        data_dir=os.environ.get('OCR_DATA_DIR','')
        if not os.path.isabs(data_dir):raise SystemExit('GLM data directory must be explicitly configured')
        for directory in ['/dev/shm/ocr-glm-home','/dev/shm/ocr-glm-tmp','/dev/shm/ocr-glm-work']:
            os.makedirs(directory,mode=0o700,exist_ok=True)
        # The operator deep-checks immutable artifacts once. Both launcher and
        # Node recheck pins, source and the full metadata record before startup.
        try:
            checked=subprocess.run([PYTHON,'services/ocr/deploy/glm_preflight.py','--manifest','/etc/ocr-glm/manifest.json',
                '--verified-cache',os.path.join(data_dir,'glm-verified-artifacts-v1.json')],
                cwd=ROOT,env=child_env({}),stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,timeout=120)
        except (OSError,subprocess.TimeoutExpired):raise SystemExit('GLM offline preflight failed') from None
        if checked.returncode!=0:raise SystemExit('GLM offline preflight failed')
        # Preflight's subprocess.run owns its own child. Start adoption only
        # after it has waited, before either long-lived process is created.
        try:self.reaper.start()
        except (OSError,RuntimeError):raise SystemExit('GLM descendant supervision could not start') from None
        self.vllm=self.spawn(['/opt/llama/llama-server','--model','/models/glm-ocr-baseline/GLM-OCR-f16.gguf',
            '--mmproj','/models/glm-ocr-baseline/mmproj-GLM-OCR-Q8_0.gguf','--alias','glm-ocr',
            '--ctx-size','24576','--parallel','1','--n-gpu-layers','999','--threads','2','--threads-batch','2',
            '--jinja','--host','127.0.0.1','--port','8001','--temp','0','--top-p','1',
            '--n-predict','4096','--image-max-tokens','4096','--no-context-shift','--metrics'],'vllm')
        deadline=time.monotonic()+120
        while not front.vllm_healthy():
            self.check_alive()
            if self.stopping or time.monotonic()>deadline:raise SystemExit('GLM native inference did not become healthy')
            time.sleep(1)
        self.api=self.spawn(['/opt/ocr/node/bin/node','--import','tsx','services/ocr/main.ts'],'api')
        deadline=time.monotonic()+120
        while not front.api_listening():
            self.check_alive()
            if self.stopping or time.monotonic()>deadline:raise SystemExit('GLM API did not become healthy')
            time.sleep(1)
        self.ready=True

    def check_alive(self):
        if self.reaper.thread is None or not self.reaper.thread.is_alive():
            raise SystemExit('GLM descendant supervision exited')
        for name,process in [('GLM native inference',self.vllm),('GLM OCR API',self.api)]:
            if process is not None and process.poll() is not None:raise SystemExit(name+' exited')

    def stop(self):
        with self.lock:self.stopping=True;self.ready=False
        # Let API cancellation/drain finish while native /metrics and /slots live.
        for process in [self.api,self.vllm]:
            if process is None:continue
            if process.poll() is None:
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                try:process.wait(timeout=3)
                except subprocess.TimeoutExpired:pass
        self.reaper.close()


front.Service=GlmService
front.child_env=child_env

if __name__=='__main__':raise SystemExit(front.main())

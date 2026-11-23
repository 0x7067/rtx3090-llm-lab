#!/usr/bin/env python3
"""Run the whole finance OCR service as one llama-swap model.

llama-swap starts this, waits for GET /health on --port, proxies requests, and
sends SIGTERM when it swaps the card to another model. The service is three
processes:

  1. vLLM (PaddleOCR-VL-1.6), the reviewed run_vllm.py, on 127.0.0.1:8000
  2. the durable OCR API (services/ocr/main.ts), on 127.0.0.1:9417
  3. this front, on --port: answers /health, forwards every other connection to 2

The OCR API has no health route and its manifest preflight rejects any change to
the reviewed profile, so readiness is decided here rather than in the API.
Hostname `ocr-vllm` must resolve to 127.0.0.1 (pod hostAliases).
"""
import argparse
import asyncio
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = "/opt/finance-ocr"
VLLM_HEALTH = "http://127.0.0.1:8000/health"
API = ("127.0.0.1", 9417)
VLLM_START_SECONDS = 900
API_START_SECONDS = 120
CHUNK = 64 * 1024


def child_env(extra: dict[str, str]) -> dict[str, str]:
    env = {
        "PATH": "/opt/ocr/node/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME": os.environ.get("HOME", "/tmp"),
        "LD_LIBRARY_PATH": "/opt/py312/lib:/usr/local/nvidia/lib64:/usr/local/nvidia/lib",
        "NVIDIA_VISIBLE_DEVICES": "all",
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
        "PADDLE_PDX_MODEL_SOURCE": "HuggingFace",
        "VLLM_NO_USAGE_STATS": "1",
        "DO_NOT_TRACK": "1",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "TMPDIR": os.environ.get("OCR_TMPDIR", "/tmp"),
        "OCR_ENABLED": "real",
        "OCR_RUNTIME_MANIFEST": os.environ["OCR_RUNTIME_MANIFEST"],
        "OCR_TOKEN_FILE": os.environ["OCR_TOKEN_FILE"],
        "OCR_NAMESPACE": os.environ["OCR_NAMESPACE"],
        "OCR_DATA_DIR": os.environ["OCR_DATA_DIR"],
        "OCR_PORT": str(API[1]),
        "OCR_LISTEN_HOST": API[0],
        "OCR_PYTHON": "/opt/ocr/venv-layout/bin/python3",
    }
    env.update(extra)
    return env


def vllm_healthy() -> bool:
    try:
        with urllib.request.urlopen(VLLM_HEALTH, timeout=2) as r:
            return r.status == 200
    except OSError:
        return False


def api_listening() -> bool:
    try:
        with socket.create_connection(API, timeout=1):
            return True
    except OSError:
        return False


class Service:
    def __init__(self) -> None:
        self.vllm: subprocess.Popen | None = None
        self.api: subprocess.Popen | None = None
        self.ready = False

    def start(self) -> None:
        self.vllm = subprocess.Popen(
            ["/opt/ocr/venv-vllm/bin/python", "services/ocr/deploy/run_vllm.py"],
            cwd=ROOT, env=child_env({}), start_new_session=True)
        deadline = time.monotonic() + VLLM_START_SECONDS
        while not vllm_healthy():
            self.check_alive()
            if time.monotonic() > deadline:
                raise SystemExit("vLLM did not become healthy")
            time.sleep(3)
        self.api = subprocess.Popen(
            ["node", "--import", "tsx", "services/ocr/main.ts"],
            cwd=ROOT, env=child_env({}), start_new_session=True)
        deadline = time.monotonic() + API_START_SECONDS
        while not api_listening():
            self.check_alive()
            if time.monotonic() > deadline:
                raise SystemExit("OCR API did not start listening")
            time.sleep(1)
        self.ready = True

    def check_alive(self) -> None:
        for name, proc in (("vLLM", self.vllm), ("OCR API", self.api)):
            if proc is not None and proc.poll() is not None:
                raise SystemExit(f"{name} exited with {proc.returncode}")

    def stop(self) -> None:
        # API first: it drains and records state before the model goes away.
        for proc in (self.api, self.vllm):
            if proc is not None and proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 30
        for proc in (self.api, self.vllm):
            if proc is None:
                continue
            try:
                proc.wait(max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while chunk := await reader.read(CHUNK):
            writer.write(chunk)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        if writer.can_write_eof():
            try:
                writer.write_eof()
            except OSError:
                pass


MAX_HEAD = 64 * 1024
# llama-swap authenticates with `Authorization: Bearer <llama token>` and rejects any
# other value there, while the OCR API authenticates with its own bearer in the same
# header. Clients therefore send the OCR token as X-OCR-Token, and this front moves
# it into Authorization. The API still does its own authentication; nothing here
# decides who is allowed in. One request per connection, no pipelining.
DROPPED = {b"authorization", b"x-api-key", b"x-ocr-token", b"connection"}


def rewrite_head(head: bytes) -> bytes:
    lines = head.split(b"\r\n")
    request_line, headers = lines[0], [l for l in lines[1:] if l]
    ocr_token = None
    kept = []
    for line in headers:
        name, _, value = line.partition(b":")
        lname = name.strip().lower()
        if lname == b"x-ocr-token":
            ocr_token = value.strip()
        if lname not in DROPPED:
            kept.append(line)
    if ocr_token:
        kept.append(b"Authorization: Bearer " + ocr_token)
    kept.append(b"Connection: close")
    return b"\r\n".join([request_line, *kept]) + b"\r\n\r\n"


async def handle(service: Service, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    upstream = None
    try:
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 10)
        if len(head) > MAX_HEAD:
            return
        if head.startswith(b"GET /health "):
            ok = service.ready and vllm_healthy()
            body = b"ok" if ok else b"starting"
            status = b"200 OK" if ok else b"503 Service Unavailable"
            writer.write(b"HTTP/1.1 " + status + b"\r\nContent-Length: " + str(len(body)).encode()
                         + b"\r\nConnection: close\r\n\r\n" + body)
            await writer.drain()
            return
        up_reader, upstream = await asyncio.open_connection(*API, limit=CHUNK)
        upstream.write(rewrite_head(head))
        await asyncio.gather(pipe(reader, upstream), pipe(up_reader, writer))
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, asyncio.TimeoutError, OSError):
        pass
    finally:
        for w in (writer, upstream):
            if w is not None:
                w.close()


async def serve(service: Service, port: int) -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    server = await asyncio.start_server(lambda r, w: handle(service, r, w), "127.0.0.1", port, limit=MAX_HEAD)

    async def watchdog() -> None:
        while not stop.is_set():
            await asyncio.sleep(5)
            if service.ready:
                try:
                    service.check_alive()
                except SystemExit as exit_:
                    print(exit_, file=sys.stderr, flush=True)
                    stop.set()

    async with server:
        watch = asyncio.create_task(watchdog())
        await stop.wait()
        watch.cancel()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    service = Service()
    # Bind first so llama-swap's /health poll gets 503 while vLLM loads.
    starter = None
    try:
        import threading
        starter = threading.Thread(target=lambda: _start(service), daemon=True)
        starter.start()
        asyncio.run(serve(service, args.port))
    finally:
        service.stop()
    return 0


def _start(service: Service) -> None:
    try:
        service.start()
    except SystemExit as exit_:
        print(exit_, file=sys.stderr, flush=True)
        os.kill(os.getpid(), signal.SIGTERM)


if __name__ == "__main__":
    sys.exit(main())

"""One bounded CPU layout/decoder process; GPU inference is owned by the parent."""
from __future__ import annotations

import os
from pathlib import Path
import signal
import socket
import sys
import json

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "services.ocr.glm"

from ..paddle.worker import (
    MAX_PROTOCOL_BYTES, MAX_OUTPUT_BYTES, SafeParser, _strict_json, apply_process_limits,
)
from ..paddle.decode import PixelBudget, decode_pages, read_original
from ..paddle.errors import WorkerError
from .adapter import GlmAdapter
from .config import parse_config


class WorkerCancelled(BaseException):
    """Unwind optional readers too; their ordinary failure handling must not catch it."""


def cancel_worker(*_):
    raise WorkerCancelled()


def read_config(filename):
    try:
        if not os.path.isabs(filename):raise WorkerError("provider_unavailable")
        descriptor=os.open(filename,os.O_RDONLY | getattr(os,"O_NOFOLLOW",0))
        with os.fdopen(descriptor,"rb") as stream:data=stream.read(MAX_PROTOCOL_BYTES+1)
        if len(data)>MAX_PROTOCOL_BYTES:raise WorkerError("provider_unavailable")
        return parse_config(_strict_json(data))
    except WorkerError:raise
    except Exception:raise WorkerError("provider_unavailable") from None


def restrict_network():
    """Defense in depth; deployment egress isolation remains the network boundary."""
    original=socket.socket.connect
    def connect(sock,address):
        if sock.family==socket.AF_INET and isinstance(address,tuple) and address==("127.0.0.1",8001):
            return original(sock,address)
        raise PermissionError("OCR worker network destination is not allowed")
    def connect_ex(sock,address):
        connect(sock,address)
        return 0
    socket.socket.connect=connect
    socket.socket.connect_ex=connect_ex


def process(original_path,config_path,request,adapter_factory=GlmAdapter):
    if (not isinstance(request,dict) or set(request)!={"protocolVersion","source"}
            or type(request["protocolVersion"]) is not int or request["protocolVersion"]!=1):
        raise WorkerError("invalid_document")
    config=read_config(config_path)
    data=read_original(original_path,request["source"])
    budget=PixelBudget();pages=[];adapter=None
    try:
        for reference in decode_pages(data,request["source"]["mime"],budget,config.pdf_scale):
            try:
                if adapter is None:adapter=adapter_factory(config,budget,source_sha256=request["source"]["originalSha256"])
                pages.append(adapter.recognize(reference))
            finally:reference.close()
        if not pages:raise WorkerError("invalid_document")
        adapter.finish(pages,data,request["source"],config.pdf_scale)
        if not any(page["status"]=="read" for page in pages):
            raise getattr(adapter,"first_failure",None) or WorkerError("invalid_document")
        warnings=[item for page in pages for item in page["warnings"]]
        if len(warnings)>100:raise WorkerError("resource_limit",resource_limit="warning_count")
        return {"protocolVersion":1,"methodsUsed":["ocr"],"pages":pages,"warnings":warnings}
    finally:
        if adapter is not None:adapter.close()


def main():
    protocol=os.dup(1)
    with open(os.devnull,"wb") as sink:
        os.dup2(sink.fileno(),1);os.dup2(sink.fileno(),2)
    code=0
    try:
        parser=SafeParser(add_help=False)
        parser.add_argument("--input",required=True);parser.add_argument("--config",required=True)
        args=parser.parse_args()
        if sys.version_info[:2]!=(3,12):raise WorkerError("provider_unavailable")
        apply_process_limits();restrict_network()
        # The CPU reader owns a subprocess. Let its finally block kill/wait it
        # before this parent exits; default SIGTERM would orphan a zombie under
        # a container PID 1 which does not reap arbitrary descendants.
        signal.signal(signal.SIGTERM,cancel_worker)
        payload=sys.stdin.buffer.read(MAX_PROTOCOL_BYTES+1)
        if len(payload)>MAX_PROTOCOL_BYTES:raise WorkerError("resource_limit",resource_limit="input_protocol")
        output=process(args.input,args.config,_strict_json(payload))
        encoded=json.dumps(output,ensure_ascii=False,allow_nan=False,separators=(",",":")).encode("utf-8","strict")
        if len(encoded)>MAX_OUTPUT_BYTES:raise WorkerError("resource_limit",resource_limit="output_protocol")
    except WorkerError as error:
        code=2;encoded=json.dumps(error.envelope(),separators=(",",":")).encode()
    except MemoryError:
        code=2;encoded=json.dumps(WorkerError("resource_limit",resource_limit="memory").envelope(),separators=(",",":")).encode()
    except BaseException:
        code=2;encoded=json.dumps(WorkerError("provider_failure").envelope(),separators=(",",":")).encode()
    try:
        remaining=memoryview(encoded+b"\n")
        while remaining:remaining=remaining[os.write(protocol,remaining):]
    finally:signal.alarm(0);os.close(protocol)
    return code


if __name__=="__main__":raise SystemExit(main())

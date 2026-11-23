"""One killable job process. Its parent owns paths, process group and cleanup."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import signal
import sys

# Support the fixed direct-file invocation inside the combined API/layout image.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "services.ocr.paddle"

from .adapter import PaddleAdapter, parse_config
from .decode import PixelBudget, decode_pages, read_original
from .errors import WorkerError

MAX_PROTOCOL_BYTES = 65_536
MAX_OUTPUT_BYTES = 10_485_760


def _strict_json(data: bytes):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise WorkerError("invalid_document")
            result[key] = value
        return result
    try:
        return json.loads(data.decode("utf-8", "strict"), object_pairs_hook=pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(WorkerError("invalid_document")))
    except WorkerError:
        raise
    except Exception:
        raise WorkerError("invalid_document") from None


def _read_config(filename: str):
    try:
        if not os.path.isabs(filename):
            raise WorkerError("provider_unavailable")
        descriptor = os.open(filename, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as config:
            data = config.read(MAX_PROTOCOL_BYTES + 1)
        if len(data) > MAX_PROTOCOL_BYTES:
            raise WorkerError("provider_unavailable")
        return parse_config(_strict_json(data))
    except WorkerError:
        raise
    except Exception:
        raise WorkerError("provider_unavailable") from None


def apply_process_limits():
    os.umask(0o077)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024 * 1024, 32 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
    resource.setrlimit(resource.RLIMIT_CPU, (300, 300))
    # RLIMIT_AS is usable in the Linux deployment; Darwin ignores/varies this limit.
    if sys.platform.startswith("linux"):
        resource.setrlimit(resource.RLIMIT_AS, (6 * 1024**3, 6 * 1024**3))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(WorkerError("timeout", True)))
    signal.alarm(300)


def process(original_path: str, config_path: str, request: dict, adapter_factory=PaddleAdapter):
    if not isinstance(request, dict) or set(request) != {"protocolVersion", "source"} or type(request["protocolVersion"]) is not int or request["protocolVersion"] != 1:
        raise WorkerError("invalid_document")
    config = _read_config(config_path)
    data = read_original(original_path, request["source"])
    budget = PixelBudget()
    pages = []
    adapter = None
    try:
        # Decoder preflight runs before constructing/loading any model.
        for reference in decode_pages(data, request["source"]["mime"], budget, config.pdf_scale):
            try:
                if adapter is None:
                    adapter = adapter_factory(config, budget)
                pages.append(adapter.recognize(reference))
            finally:
                reference.close()
        if not pages or not any(page["status"] == "read" for page in pages):
            raise WorkerError("invalid_document")
        warnings = [item for page in pages for item in page["warnings"]]
        if len(warnings) > 100:
            raise WorkerError("resource_limit", resource_limit="warning_count")
        return {"protocolVersion": 1, "methodsUsed": ["ocr"], "pages": pages, "warnings": warnings}
    finally:
        if adapter is not None:
            adapter.close()


class SafeParser(argparse.ArgumentParser):
    def error(self, _message):
        raise WorkerError("invalid_document")


def main() -> int:
    # Native/Python dependency logs can contain paths or recognized text. Only
    # this preserved protocol descriptor can write to the parent's stdout pipe.
    protocol = os.dup(1)
    with open(os.devnull, "wb") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
    code = 0
    try:
        parser = SafeParser(add_help=False)
        parser.add_argument("--input", required=True)
        parser.add_argument("--config", required=True)
        args = parser.parse_args()
        if sys.version_info[:2] != (3, 12):
            raise WorkerError("provider_unavailable")
        apply_process_limits()
        payload = sys.stdin.buffer.read(MAX_PROTOCOL_BYTES + 1)
        if len(payload) > MAX_PROTOCOL_BYTES:
            raise WorkerError("resource_limit", resource_limit="input_protocol")
        output = process(args.input, args.config, _strict_json(payload))
        encoded = json.dumps(output, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8", "strict")
        if len(encoded) > MAX_OUTPUT_BYTES:
            raise WorkerError("resource_limit", resource_limit="output_protocol")
    except WorkerError as error:
        code = 2
        encoded = json.dumps(error.envelope(), separators=(",", ":")).encode()
    except MemoryError:
        code = 2
        encoded = json.dumps(WorkerError("resource_limit", resource_limit="memory").envelope(), separators=(",", ":")).encode()
    except BaseException:
        code = 2
        encoded = json.dumps(WorkerError("provider_failure").envelope(), separators=(",", ":")).encode()
    try:
        remaining = memoryview(encoded + b"\n")
        while remaining:
            remaining = remaining[os.write(protocol, remaining):]
    finally:
        signal.alarm(0)
        os.close(protocol)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

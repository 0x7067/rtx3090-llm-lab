"""Expose the active vLLM worker's /metrics on :9106 for Prometheus.

Direct loopback endpoints come from llama-swap's shared configuration.
llama-swap's /upstream/<model>/metrics route would also reach them, but a request
there makes llama-swap load the model, so every scrape after an image/OCR swap
would evict that job. This proxy only reads worker ports: when no vLLM worker is
loaded it answers 200 with vllm_metrics_proxy_upstream_up 0 instead of failing
the scrape.
"""

import http.server
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

CONFIG = Path(os.environ.get("LLAMA_SWAP_CONFIG", "/config/config.yaml"))
UP = b"# HELP vllm_metrics_proxy_upstream_up 1 if an active vLLM worker answered /metrics.\n# TYPE vllm_metrics_proxy_upstream_up gauge\nvllm_metrics_proxy_upstream_up %d\n"


def upstream_urls(config=CONFIG):
    models = yaml.safe_load(config.read_text())["models"]
    urls = []
    for model in models.values():
        proxy = model.get("proxy")
        if not isinstance(proxy, str) or "$" in proxy:
            continue
        parsed = urllib.parse.urlsplit(proxy)
        if parsed.scheme not in ("http", "https") or parsed.hostname not in (
            "127.0.0.1", "localhost", "::1"
        ):
            continue
        url = parsed._replace(path="/metrics", query="", fragment="").geturl()
        if url not in urls:
            urls.append(url)
    return urls


def worker_metrics():
    for url in upstream_urls():
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                body = response.read()
            if any(line.startswith(b"vllm:") for line in body.splitlines()):
                return body + UP % 1
        except OSError:
            continue
    return UP % 0


def observer_metrics():
    try:
        state = json.loads(
            Path("/cache/sparsekv-observations/current/status.json").read_text()
        )
        age = max(0, time.time() - state["updated_at"])
        up = age <= 30 and not state.get("target_exited", False)
    except (OSError, ValueError, KeyError, TypeError):
        state, age, up = {}, 0, False
    values = {
        "up": int(up),
        "sample_age_seconds": age,
        "suspected_stall": int(up and bool(state.get("reason"))),
        "stalled_seconds": state.get("stalled_seconds", 0),
        "captures_total": state.get("capture_count", 0),
        "storage_budget_reached": int(bool(state.get("storage_budget_reached"))),
    }
    return "".join(
        f"# TYPE sparsekv_observer_{name} {'counter' if name == 'captures_total' else 'gauge'}\n"
        f"sparsekv_observer_{name} {value}\n"
        for name, value in values.items()
    ).encode()


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/healthz":
            body = b"ok\n"
        elif self.path == "/metrics":
            body = worker_metrics() + observer_metrics()
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    upstream_urls()  # Fail startup if the shared configuration cannot be read.
    http.server.ThreadingHTTPServer(("0.0.0.0", 9106), Handler).serve_forever()

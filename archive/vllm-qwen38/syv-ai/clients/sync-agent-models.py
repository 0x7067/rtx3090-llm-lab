#!/usr/bin/env python3
"""Sync pi and prime-agent model configs from the live vLLM endpoint.

Discovers, from the server itself:
  - model id and context window   (GET /v1/models: id, max_model_len)
  - reasoning-effort levels       (probe with an invalid effort; the chat
                                   template's 400 message enumerates them,
                                   e.g. "Supported types are xhigh (default),
                                   medium, and low.")
  - vision support                (1x1-pixel image probe: 200 = multimodal,
                                   400 = text-only)

Then rewrites the target provider entry in ~/.pi/agent/models.json and
~/.prime/agent/models.json, preserving every other provider and each file's
local conventions (headers, apiKey, authHeader). A timestamped .bak of each
file is written first; writes are atomic. Re-run after any deployment change
(MAX_LEN, model swap, template change) instead of editing configs by hand.

Usage:
  clients/sync-agent-models.py [--base http://llama.apps.svc.cluster.local:8080/v1]
                               [--provider home-vllm] [--dry-run]
  BEARER=<token> for the Caddy-gated public path.
"""
import argparse, base64, json, os, re, shutil, sys, time, urllib.request

def _png_1px():
    import struct, zlib
    def chunk(t, d):
        c = t + d
        return struct.pack(">I", len(d)) + c + struct.pack(">I", zlib.crc32(c))
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\xff\x00\x00")
    return base64.b64encode(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
                            + chunk(b"IDAT", idat) + chunk(b"IEND", b"")).decode()

PNG_1PX = _png_1px()

def call(base, path, payload=None, bearer=None, timeout=60):
    req = urllib.request.Request(base + path,
        data=json.dumps(payload).encode() if payload else None,
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {bearer}"} if bearer else {})},
        method="POST" if payload else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        try: return e.code, json.load(e)
        except Exception: return e.code, {"raw": e.read().decode(errors="replace")[:400]}

def discover(base, bearer):
    st, j = call(base, "/models", bearer=bearer)
    if st != 200 or not j.get("data"):
        sys.exit(f"cannot list models at {base}/models: HTTP {st} {j}")
    m = j["data"][0]
    model_id, ctx = m["id"], m.get("max_model_len")
    if not ctx:
        sys.exit(f"/models did not report max_model_len: {m}")
    # effort levels: the template's raise_exception enumerates them on a bad value
    st, j = call(base, "/chat/completions", {
        "model": model_id, "max_tokens": 1,
        "messages": [{"role": "user", "content": "hi"}],
        "chat_template_kwargs": {"reasoning_effort": "___invalid___"}}, bearer)
    msg = json.dumps(j)
    levels = None
    if st == 400:
        found = re.findall(r"\b(xhigh|high|medium|low|minimal)\b", msg)
        levels = sorted(set(found), key=found.index) or None
    # vision probe
    st, _ = call(base, "/chat/completions", {
        "model": model_id, "max_tokens": 1, "messages": [{"role": "user", "content": [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG_1PX}"}}]}]},
        bearer, timeout=300)
    vision = st == 200
    return model_id, ctx, levels, vision

def thinking_map(levels, style):
    """pi fills aliases; prime-agent nulls non-native levels."""
    if not levels:
        return None
    have = set(levels)
    alias = {"off": "off",
             "minimal": "minimal" if "minimal" in have else ("low" if "low" in have else None),
             "low": "low" if "low" in have else None,
             "medium": "medium" if "medium" in have else None,
             "high": "high" if "high" in have else ("xhigh" if "xhigh" in have else None),
             "xhigh": "xhigh" if "xhigh" in have else None,
             "max": "xhigh" if "xhigh" in have else ("high" if "high" in have else None)}
    if style == "prime":  # exact levels only, keep off
        return {k: (v if (v in have or (k == v == "off")) and k in (*have, "off") else None)
                for k, v in alias.items()}
    return alias

def update_file(path, provider, model_id, ctx, levels, vision, max_tokens, dry):
    if not os.path.exists(path):
        print(f"skip {path}: not found"); return
    cfg = json.load(open(path))
    prov = cfg.setdefault("providers", {}).setdefault(provider, {})
    models = prov.get("models") or [{}]
    entry = models[0]
    style = "prime" if "/.prime/" in path else "pi"
    old = json.dumps(entry, sort_keys=True)
    entry.update({
        "id": model_id,
        "name": entry.get("name") or f"{model_id} (home vLLM)",
        "reasoning": bool(levels),
        "input": ["text", "image"] if vision else ["text"],
        "contextWindow": ctx,
        "maxTokens": min(max_tokens, ctx // 2),
        "cost": entry.get("cost") or {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
    })
    tm = thinking_map(levels, style)
    if tm:
        entry["thinkingLevelMap"] = tm
        entry.setdefault("compat", {})["supportsReasoningEffort"] = True
    prov["models"] = [entry]
    changed = old != json.dumps(entry, sort_keys=True)
    if dry:
        print(f"{path}: {'WOULD update' if changed else 'up to date'}"); return
    if changed:
        shutil.copy2(path, f"{path}.bak-{time.strftime('%Y%m%dT%H%M%S')}")
        tmp = path + ".tmp"
        json.dump(cfg, open(tmp, "w"), indent=2)
        os.replace(tmp, path)
    print(f"{path}: {'updated' if changed else 'up to date'} "
          f"(id={model_id} ctx={ctx} levels={levels} vision={vision})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://llama.apps.svc.cluster.local:8080/v1")
    ap.add_argument("--provider", default="home-vllm")
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    model_id, ctx, levels, vision = discover(a.base.rstrip("/"), os.environ.get("BEARER"))
    print(f"discovered: id={model_id} max_model_len={ctx} effort_levels={levels} vision={vision}")
    for p in ("~/.pi/agent/models.json", "~/.prime/agent/models.json"):
        update_file(os.path.expanduser(p), a.provider, model_id, ctx, levels, vision,
                    a.max_tokens, a.dry_run)

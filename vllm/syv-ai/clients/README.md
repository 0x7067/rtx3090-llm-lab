# Client configs with dynamically discovered parameters

`sync-agent-models.py` configures the pi and prime-agent CLIs against this
stack's OpenAI-compatible endpoint without hand-maintained parameters. It
interrogates the running server and rewrites the `home-vllm` provider entry
in `~/.pi/agent/models.json` and `~/.prime/agent/models.json`, preserving
every other provider and each file's local conventions.

## What is discovered, and from where

| parameter | source |
|---|---|
| model id, `contextWindow` | `GET /v1/models` (vLLM reports `max_model_len`) |
| `reasoning_effort` levels | probe with an invalid effort; the chat template's `raise_exception` message enumerates the valid levels in its 400 body |
| vision (`input: ["text","image"]`) | a 1x1-pixel PNG probe: HTTP 200 = multimodal, 400 = text-only |
| `maxTokens` | policy: `min(--max-tokens, contextWindow // 2)`, default cap 32768 |

Neither CLI discovers custom-provider parameters natively (their catalog
refresh covers SaaS providers only), so this script is the discovery layer.

## Usage

```bash
clients/sync-agent-models.py                 # discover + write both configs
clients/sync-agent-models.py --dry-run       # show what would change
clients/sync-agent-models.py --base http://host:port/v1 --provider home-vllm
BEARER=<token> clients/sync-agent-models.py  # via the Caddy-gated public path
```

Re-run after any deployment change — a different `MAX_LEN`, a model swap, or
a chat-template change — instead of editing the JSON by hand. Timestamped
`.bak-*` backups are written next to each config before any change; writes
are atomic.

Total probe cost: three requests, 3 output tokens.

## Notes

- `thinkingLevelMap` styles differ by agent and are preserved: pi fills
  aliases (`high -> xhigh`, `minimal -> low`), prime-agent maps only exact
  levels and nulls the rest.
- The default endpoint is the in-cluster service (`llama.apps.svc.cluster.local:8080`). The
  public path is bearer-gated by Caddy; pass `BEARER`.
- vLLM 0.27.1 returns the CoT in `message.reasoning` (not
  `reasoning_content`); both CLIs already read the right field.

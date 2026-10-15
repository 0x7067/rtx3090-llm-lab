---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
library_name: vllm
language:
  - en
  - zh
tags:
  - qwen
  - qwen3.8
  - qwen3_5
  - orcasaq2
  - quantization
  - mixed-precision
  - 3-bit
  - vllm
  - reasoning
  - function-calling
  - mtp
  - agentic
  - long-horizon
  - long-context
---

<p align="center">
  <a href="https://www.orcarouter.ai">
    <img src="https://www.orcarouter.ai/orca-logo-classic.png" width="96" alt="OrcaRouter">
  </a>
</p>

<h1 align="center">OrcaSAQ2 27B</h1>


<!-- orcacyber-promo:start -->
<table>
<tr>
<td width="150" style="vertical-align: middle;">
<div align="center"><a href="https://www.orcarouter.ai/models/orca/orcacyber-zero-1.0"><img src="https://www.orcarouter.ai/orca-logo-classic.png" width="90" alt="OrcaCyber Zero 1.0"></a></div>
</td>
<td>
<p>🛡️ <strong>New from Orca &mdash; <a href="https://www.orcarouter.ai/models/orca/orcacyber-zero-1.0">OrcaCyber Zero 1.0</a></strong></p>
<p>Post-trained for cybersecurity: vulnerability analysis, vulnerability reproduction, red teaming &amp; offensive-security research. <strong>98.07% pass@1 on CyberGym Level&nbsp;1</strong> &middot; 1M context &middot; function calling &middot; OpenAI-compatible API.</p>
<p><a href="https://www.orcarouter.ai/models/orca/orcacyber-zero-1.0"><img src="https://img.shields.io/badge/Try%20it%20on%20OrcaRouter%20%E2%86%92-FF6A00?style=for-the-badge&amp;labelColor=0B1F3A" alt="Try it on OrcaRouter"></a></p>
</td>
</tr>
</table>
<!-- orcacyber-promo:end -->

<p align="center"><strong>High-fidelity 3-bit Qwen3.8 for long-horizon agents.</strong></p>

<p align="center"><strong>54 GB → 12.3 GB · +0.02% PPL · 93.2% Top-1 Agreement · 0.031 KLD · 262K Context</strong></p>

<p align="center">
  <a href="https://www.orcarouter.ai"><strong>OrcaRouter AI Gateway</strong></a> ·
  <a href="https://x.com/OrcaRouter"><strong>X</strong></a> ·
  <a href="https://discord.gg/yAh6Tex6kx"><strong>Discord</strong></a> ·
  <a href="https://github.com/Continuum-AI-Corp"><strong>GitHub</strong></a> ·
  <a href="https://www.orcarouter.ai/models"><strong>All Models</strong></a>
</p>

---

> ## 27B reasoning. 12.3 GB.
>
> **OrcaSAQ2 27B** compresses Qwen3.8-27B from a **54 GB BF16 checkpoint to 12.3 GB** while preserving extremely high fidelity to the original model.
>
> Built for: **long-horizon agents · coding · tool use · reasoning · stateful execution**

OrcaSAQ2 is a proprietary **sensitivity-aware mixed-precision quantization system** developed by OrcaRouter and its research team behind.

It is optimized around one goal: **Preserve as much useful model behavior as possible inside a practical GPU memory envelope.**

The resulting checkpoint provides:
- **77.2% smaller storage footprint**
- only **+0.02% perplexity** versus BF16
- **93.2% token-level Top-1 agreement**
- **0.031 mean KLD**
- **262K context**
- thinking mode
- tool calling
- MTP speculative decoding
- production serving through **vLLM**

---

## At a glance

| Metric | BF16 | OrcaSAQ2 |
|---|---:|---:|
| Checkpoint | 54 GB | **12.3 GB** |
| Relative size | 100% | **22.8%** |
| Storage reduction | — | **77.2%** |
| Decoder precision | 16-bit | **3.21 bpw avg.** |
| Perplexity | 5.6468 | **5.6482** |
| PPL delta | — | **+0.02%** |
| Top-1 agreement | 100% | **93.2%** |
| Mean KLD | — | **0.031** |
| Context | 262K | **262K** |

<p align="center"><strong>4.4× smaller. +0.02% perplexity.</strong></p>
<p align="center">The point is not 3-bit.</p>
<p align="center"><strong>The point is what survives at 3-bit.</strong></p>

---

## BF16 fidelity

All numbers below are measured using **these exact OrcaSAQ2 weights** against the BF16 reference through the same evaluation path.

### WikiText-2

**16,376 predicted tokens**

| Build | Size | Decoder Bits | Mean KLD ↓ | Top-1 Agreement ↑ | PPL ↓ |
|---|---:|---:|---:|---:|---:|
| Qwen3.8-27B BF16 | 54 GB | 16 | — | 100% | 5.6468 |
| **OrcaSAQ2 27B** | **12.3 GB** | **3.21** | **0.031** | **93.2%** | **5.6482** |

### Perplexity

```text
BF16       5.6468  ████████████████████████████████████████
OrcaSAQ2   5.6482  ████████████████████████████████████████

Delta: +0.02%
```

### Top-1 agreement

```text
OrcaSAQ2 vs BF16

███████████████████████████████████████████████░░░  93.2%
```

### Model footprint

```text
Qwen3.8-27B BF16

██████████████████████████████████████████████████  54.0 GB

OrcaSAQ2

███████████                                         12.3 GB
```

**77.2% smaller.**

---

## Long-horizon agents

### Low-bit fidelity matters more as the horizon grows.

Short benchmarks can hide small degradation.

Agents cannot.

A small model error can change a tool call.

That changes the environment state.

The changed state affects every decision that follows.

```text
Plan
  ↓
Act
  ↓
Observe
  ↓
Decide
  ↓
Recover
  ↓
Repeat
  ↓
...
  ↓
Task Success
```

Across long trajectories, small errors can compound into large behavioral differences.

That makes long-horizon execution an especially useful stress test for compressed reasoning models.

OrcaSAQ2 performs **strongly on long-horizon workloads relative to models in its deployment and parameter class**, despite operating from a **12.3 GB checkpoint**.

This makes it particularly suitable for:

- coding agents
- terminal agents
- browser agents
- computer-use agents
- security agents
- repository-scale tasks
- multi-tool workflows
- failure recovery
- long-running stateful execution

### Why this matters

Perplexity asks:

> **How similar is the next-token distribution?**

Long-horizon evaluation asks:

> **Can the model still finish the job after many decisions?**

For agent models, both matter.

---

## Long-horizon performance

Agent benchmarks depend heavily on the surrounding scaffold, tools, reasoning
budget, timeouts and execution environment. The results below are therefore
shown as **public reference points, not direct apples-to-apples comparisons**.

#### SWE-bench Verified

| Model | Reported score |
|---|---:|
| Claude Sonnet 4.6 | 79.6 |
| Claude Sonnet 4.5 | 77.2 |
| Gemini 3 | 76.2 |
| **OrcaSAQ2 27B** | **70.0** |
| Qwen3-Coder-480B-A35B | 69.6 |
| Gemini 2.5 Pro | 63.8 |
| GPT-4.1 | 54.6 |

**70.0% SWE-bench Verified from a 12.06 GB 27B checkpoint.**

#### Terminal-Bench 2.1

| Model / Agent | Reported score |
|---|---:|
| Gemini 3.1 Pro / Terminus 2 | 70.7 |
| Claude Opus 4.6 / Claude Code | 70.1 |
| Claude Opus 4.6 / Terminus 2 | 63.8 |
| Claude Sonnet 4.6 / Claude Code | 58.5 |
| **OrcaSAQ2 27B** | **58.4** |
| Gemini 3 Flash / Gemini CLI | 56.9 |
| GPT-5.4 / Terminus 2 | 54.8 |
| Claude Sonnet 4.6 / Terminus 2 | 51.5 |

**58.4% Terminal-Bench 2.1 while fitting in ~12 GB of checkpoint storage.**

> Public scores use different agent stacks and should not be interpreted as a
> strict model-only ranking.

---

## Architecture

| | |
|---|---|
| **Base model** | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) |
| **Architecture** | `Qwen3_5ForCausalLM` |
| **Layers** | 64 |
| **Hidden size** | 5120 |
| **Hybrid attention** | 48 Gated DeltaNet + 16 full-attention layers |
| **Context** | **262,144 tokens** |
| **Vocabulary** | 248,320 |
| **MTP head** | Included |
| **Thinking** | Supported |
| **Tool calling** | Supported |
| **Checkpoint** | **12.3 GB** |
| **Decoder average** | **3.21 bpw** |
| **Serving** | **vLLM** |
| **Vision** | Not included |
| **License** | Apache-2.0 |

---

## Production serving

### Up to 90.1 tok/s single-stream on a 16 GB GPU

Measured under a **15.7 GiB GPU memory cap**.

| Configuration | 1 Stream | 8 Streams | 16 Streams | KV Pool |
|---|---:|---:|---:|---:|
| vLLM · MTP off | 65.3 tok/s | **332 tok/s** | **333 tok/s** | 29,354 tok |
| **vLLM · MTP on** | **90.1 tok/s** | 220 tok/s | 219 tok/s | 14,563 tok |

### MTP speculative decoding

```text
Single-stream decode

MTP off    █████████████████████████████       65.3 tok/s

MTP on     ████████████████████████████████████████
                                                90.1 tok/s
```

**+38% single-stream decode throughput**

MTP trades additional compute and KV capacity for stronger interactive decode performance.

It is particularly useful for:

- coding assistants
- interactive agents
- terminal agents
- tool-heavy applications
- low-concurrency inference

For highly batched workloads, benchmark both configurations.

---

## 27B on a 16 GB GPU

OrcaSAQ2's checkpoint is **12.3 GB**.

That makes deployment possible on hardware that cannot hold the original 54 GB BF16 checkpoint.

```text
16 GB GPU
┌───────────────────────────────────────────┐
│                                           │
│   OrcaSAQ2 weights          12.3 GB       │
│   ███████████████████████████████████     │
│                                           │
│   Remaining               ~3.7 GB         │
│   ██████████                              │
│                                           │
└───────────────────────────────────────────┘
```

Actual usable memory depends on:

- vLLM overhead
- KV-cache configuration
- MTP
- batch size
- context length
- CUDA graph configuration

A practical starting point for a 16 GB GPU is approximately **32K interactive context**, then tune based on the workload.

The model architecture supports up to **262K context**.

---

## Built for agents

### Long-horizon execution

```text
plan → act → observe → recover → repeat
```

### Coding

Repository-scale generation, editing, testing and debugging.

### Tool use

Structured workflows where action-selection quality matters.

### Reasoning

Preserving the capabilities of the 27B base model under an aggressive deployment constraint.

### Single-GPU deployment

A **12.3 GB** checkpoint designed around practical inference hardware.

### Production serving

**vLLM + MTP + OpenAI-compatible APIs.**

---

## Generation samples

One prompt each, first attempt.

### Pelican on a bicycle

The standard SVG test, asked for as an animation.

Chain over the chainring, cranks 180° out of phase, parallax background.
Pure SMIL, no JavaScript. Used as generated.

<p align="center">
  <img src="https://huggingface.co/orcarouter/OrcaSAQ-2-27B/resolve/main/assets/pelican-bicycle.svg" width="760" alt="Animated SVG of a pelican riding a bicycle">
</p>

### Low-poly Statue of Liberty

> Create a html low-poly 3D models of the Statue of Liberty

A single self-contained HTML file: Three.js scene, orbit controls, procedural geometry.

<p align="center">
  <video src="https://huggingface.co/orcarouter/OrcaSAQ-2-27B/resolve/main/assets/liberty.mp4" width="760" controls loop muted playsinline></video>
</p>

---

## Quickstart

### Install

```bash
pip install -U vllm huggingface_hub

pip install git+https://github.com/Continuum-AI-Corp/OrcaSAQ2-kernel
```

---

### Download

```bash
hf download orcarouter/OrcaSAQ2-27B \
  --local-dir ./OrcaSAQ2-27B
```

---

### Serve with vLLM

```bash
vllm serve ./OrcaSAQ2-27B \
  --served-model-name OrcaSAQ2-27B \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
```

---

## OpenAI-compatible API

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="OrcaSAQ2-27B",
    messages=[
        {
            "role": "user",
            "content": "Analyze this repository and plan the next five actions."
        }
    ],
)

print(response.choices[0].message.content)
```

---

## Recommended sampling

```text
temperature = 1.0
top_p       = 0.95
top_k       = 20
```

Thinking mode is enabled by default.

For agent deployments, benchmark against the actual tool schema, context distribution and reasoning budget used in production.

---

## Evaluation philosophy

A low-bit reasoning model should not be judged by checkpoint size alone.

We look at the intersection of:

<p align="center"><strong>Footprint × BF16 Fidelity × Capability × Long-Horizon Stability × Serving Performance</strong></p>

A useful low-bit model must remain useful after compression.

---

## Why perplexity alone is not enough

Perplexity is useful and reproducible.

It is not a complete measure of agentic capability.

Quantization can affect:

```text
reasoning
   ↓
planning
   ↓
tool selection
   ↓
state tracking
   ↓
recovery
   ↓
task completion
```

That is why OrcaSAQ2 reports BF16 fidelity metrics alongside downstream and long-horizon evaluation.

---

## Method

OrcaSAQ2 uses a proprietary **sensitivity-aware mixed-precision quantization system** developed by OrcaRouter.

The implementation is optimized to preserve model quality under a strict deployment-memory target.

Detailed quantization methodology, calibration strategy, precision allocation and packing techniques are not currently disclosed.

---

## Limitations

- OrcaSAQ2 inherits the capabilities, biases and limitations of Qwen3.8-27B.
- Quantization is not mathematically lossless.
- **93.2% Top-1 agreement** means some token decisions differ from BF16.
- **+0.02% PPL** is a model-fidelity measurement and does not guarantee identical downstream performance.
- Long-horizon comparisons should use a controlled same-harness evaluation.
- This checkpoint is **text-only**.
- The vision tower is not included.
- OrcaSAQ2 requires the OrcaSAQ2 vLLM integration.
- Maximum architectural context does not imply that the full context fits into every GPU memory envelope.

---

## Open source from OrcaRouter

### [OrcaCode Review](https://github.com/Continuum-AI-Corp/Orca-Code-Review)

Open multi-model code review.

### [OrcaReplay](https://github.com/Continuum-AI-Corp/OrcaReplay)

Record, replay, fork and debug AI-agent runs.

### [OrcaRouter Lite](https://github.com/Continuum-AI-Corp/OrcaRouter-Lite)

Self-hosted multi-model AI infrastructure.

<p align="center"><strong>Open model. Open harness. Open bill.</strong></p>

---

## Citation

```bibtex
@misc{qwen38,
    title = {Qwen3.8-Max: A New Bar for Coding and Cowork},
    author = {{Qwen Team}},
    year = {2026},
    month = {August},
    url = {https://qwen.ai/blog?id=qwen3.8}
}
```

---

## License

**Apache-2.0**

Inherited from:

[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)

Quantization does not change the underlying license obligations.

---

<h2 align="center">One Gateway. Every Model.</h2>

<p align="center"><strong>Route Smarter · Ship Safer · Spend Less</strong></p>

<p align="center">
  <a href="https://www.orcarouter.ai">Website</a> ·
  <a href="https://www.orcarouter.ai/models">Models</a> ·
  <a href="https://github.com/Continuum-AI-Corp">GitHub</a> ·
  <a href="https://discord.gg/yAh6Tex6kx">Discord</a> ·
  <a href="https://x.com/OrcaRouter">X</a>
</p>

# OrcaSAQ2 versus W4A16 on one RTX 3090

## Decision

Promote OrcaSAQ2 EXL3 3.21 bpw under the stable `qwen3.8-27b` model ID. Keep
the previous dbirks W4A16 AutoRound profile as `qwen3.8-27b-w4a16` for
rollback.

Use `medium` reasoning effort by default. Use `low` for code-only generation.
The clients cannot select effort by task type automatically, so this remains a
request-level choice.

## Profiles

Both profiles ran behind llama-swap on one RTX 3090 with vLLM 0.30.0 and the
froggeric fixed Qwen chat template v22.5. The template source revision was
`855bffc49448e299789730ff92c9b8d834d6cc14`. Its source SHA-256 was
`e57684bae4156211a55473c5a63be976a405a37ab5be5ae0e5abf1df5349c4b2`.
The vendored file added a final newline and had SHA-256
`b143d1dd3a9627fe1976848dd3a533f47991967059091636196b055f7a6d3963`.

| Profile | Weights and runtime | Configured context | Concurrency |
| --- | --- | ---: | ---: |
| `qwen3.8-27b` | OrcaSAQ2 EXL3 3.21 bpw, `orcasaq2-kernel`, exllamav3, MTP | 150,000 | 8 sequences |
| `qwen3.8-27b-w4a16` | dbirks W4A16 AutoRound, stock vLLM compressed-tensors loader | 98,304 | 64 sequences |

The promoted OrcaSAQ2 boot reported 292,647 KV-cache tokens and 1.95 times
maximum concurrency at 150,000 tokens. This capacity report does not replace a
150,000-token request test.

## Results

### Template screen

Both profiles passed the 16-case smoke workload with the fixed template. The
template also passed stringified tool history, developer-message,
false-positive tool-error, and inline reasoning-tag checks.

The smoke timings did not establish a speed winner. OrcaSAQ2 decoded the short
probe at 48.39 tokens per second versus 47.42 for W4A16. Its time to first token
was 0.139 seconds versus 0.084 seconds, and its 6K prefill took 5.335 seconds
versus 4.696 seconds.

### Quality and effort

The workload had 54 quality cases and four performance probes. The quality
cases covered executable Python, extraction, reasoning, relevance, retrieval,
tool replay, and transcript questions.

| Effort policy | Coding | Other quality | Total | Result |
| --- | ---: | ---: | ---: | --- |
| medium for every case | 15/20 | 34/34 | 49/54 | Five coding cases used all 4,096 tokens as hidden reasoning and emitted no code. |
| low for every case | 20/20 | 26/34 | 46/54 | All six retrieval cases failed, plus one extraction and one relevance case. |
| low for coding, medium otherwise | 20/20 | 34/34 | **54/54** | Qualified. |

The qualified hybrid suite SHA-256 was
`5cc022ae88bdca06205458088ff0d66f7fa50722251585fd622edd215c00493b`.
The frozen plan SHA-256 was
`17cfff0e4fabccb714fbd007d19103523497bb76faf8f6e500da0b407d509f1a`.
The largest retrieval prompt contained 57,468 tokens.

Increasing the medium-effort ceiling did not provide a general fix. With an
8,192-token ceiling, three of the five failed coding cases passed, but
`code-histogram` and `code-path-normalize` again used the entire budget as
hidden reasoning and emitted no code.

### W4A16 admission failure

W4A16 passed the smoke workload, then exhausted GPU memory during the larger
workload. The failures reproduced at 57,468, 28,794, 14,460, and 1,916 prompt
tokens when each request reserved 4,096 output tokens. At 14,460 prompt tokens,
the process had about 40 MiB free and failed a 38 MiB Gated DeltaNet prefill
allocation. The fatal vLLM error stopped the engine, and later requests returned
HTTP 500 or 502.

The failure means that the configured 98,304-token limit was not an achievable
request limit for this profile. The complete matched quality comparison could
not continue after the engine exited. Preserve those failed cohorts as
admission evidence rather than scoring them as model-quality failures.

## Production verification

Flux applied the promotion at GitOps commit
`2a9d4c2b7b8ade62cb87c979723ca5c2d4882c22`. The deployment reached one ready,
updated, and available replica. Authenticated checks passed exact text output,
structured tool calling, and the model catalog. The catalog exposed
`qwen3.8-27b` and `qwen3.8-27b-w4a16`.

Pi, Prime Agent, and Jcode resolved the stable model ID and completed live
requests. Pi and Prime Agent advertise a 150,000-token context window. Their
model-specific effort remains `medium`. Jcode's OpenAI-compatible provider
effort also remains `medium`.

The second reference Mac at `100.64.0.7` was unreachable over SSH, so its
client metadata remains unverified.

## Limits

- The workload is synthetic development and regression data. It is not an
  independently sampled production benchmark.
- The effort comparison changes one request parameter on one quantization and
  runtime profile. It does not establish that low effort is generally better
  for coding models.
- The W4A16 engine failure prevents a complete large-workload quality
  comparison. The promotion rests on equal smoke quality, OrcaSAQ2's complete
  workload, and W4A16's repeated admission failure.
- The configured 150,000-token context needs a separate end-to-end request
  check before treating the full window as qualified.

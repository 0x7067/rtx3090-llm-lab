# Paddle worker

`worker.py` implements the private, single-job subprocess used by the TypeScript
provider. It decodes original bytes, runs PP-DocLayoutV3 on CPU, and recognizes
each measured region through PaddleOCR-VL-1.6-0.9B on the private vLLM server.
It returns page/block evidence; financial field extraction belongs to
`../normalize.ts`.

The worker is implementation code, not evidence of a validated GPU deployment.
The Python decoder and synthetic protocol tests have run locally. Paddle model
loading, real layout inference, vLLM generation, GPU memory, end-to-end latency,
and accuracy on real Portuguese receipts still require the staged container
and authorized representative documents. No real inference was run while
implementing this adapter.

## Process protocol

The parent creates a private job directory, writes the original and trusted
configuration, and invokes:

```text
python /opt/finance-ocr/services/ocr/paddle/worker.py --input /private/job/original --config /private/job/worker.json
```

Stdin contains exactly this JSON shape, bounded to 64 KiB:

```json
{"protocolVersion":1,"source":{"mime":"image/png","sizeBytes":123,"originalSha256":"64 lowercase hexadecimal characters"}}
```

No path, endpoint, prompt, or model selector is accepted from document metadata.
The absolute input path is a trusted process argument. It must be a regular file
with one hard link; symlinks, changed size/metadata, and hash mismatches fail.

Success exits zero and writes one JSON object to stdout:

```text
{protocolVersion:1, methodsUsed:["ocr"], pages:[contract page objects], warnings:[contract warnings]}
```

Failure exits two and writes
`{protocolVersion:1,error:{code,message,retryable}}` with a fixed redacted message.
The parent treats missing/invalid output or a signal as failure. Python and native
stdout/stderr are redirected to the null device; only the bounded protocol
descriptor is exposed. The subprocess never writes recognized text to logs.

The trusted config has these exact keys:

```json
{
  "configVersion": 1,
  "pipelineVersion": "v1.6",
  "layoutModelDir": "/models/layout",
  "vlModelDir": "/models/ocr",
  "vllmBaseUrl": "http://ocr-vllm:8000/v1",
  "servedModelName": "PaddleOCR-VL-1.6-0.9B",
  "pdfScale": 2,
  "layoutDevice": "cpu"
}
```

Model directories must already exist and be validated by deployment preflight.
The endpoint is limited to `ocr-vllm` or a loopback IP, with path `/v1`, no query,
fragment, or credentials. The runtime network and model server are private.

## Bounds and evidence

- Originals: 10 MiB, at most 50 pages, 16,000 pixels per side, 40 million pixels
  per page, and 100 million cumulative raster pixels. Every explicit full-size
  conversion/copy is charged before allocation: the EXIF copy even at orientation
  1, all four alpha-compositing rasters, and PDF bitmap/Pillow/detachment copies.
  NumPy/color conversion, crops, encoding/decoding, and bounded recognition rasters
  consume the same cumulative budget. Native layout preprocessing is charged at
  its actual inspected resize target (maximum 1,024 pixels per side); an unknown
  target fails closed. Model feature activations are governed separately by the
  process memory limit. These ceilings do not promise that every document
  at an individual ceiling fits all processing passes.
- PNG, JPEG, and WebP use pinned Pillow decoding and all eight EXIF orientations.
  Transparent pixels are flattened on white. Animated/multiframe images fail.
- PDF uses PDFium at scale 2 (144 DPI), with its inherited CropBox and `/Rotate`
  handling. Rendered dimensions and pixel colors are tested for all four rotations.
  Both password-protected and empty-password encrypted PDFs fail. Dynamic forms
  are unsupported. No PDF embedded-text API is called.
- Layout runs without deskew, document orientation classification, unwarping, or
  merging. The layout predictor's reading order is preserved. Rectangles are
  validated in the full displayed original page, cropped with integer outward
  bounds, and normalized in that same reference frame. No crop-relative boxes
  are returned. Missing render geometry fails; known unreadable pages stay explicit.
- At most 512 regions per page are accepted. Each crop is lossless PNG, and each
  request uses the fixed `OCR:` transcription prompt, at most 4,096 output tokens,
  fixed generation parameters, and bounded multimodal pixel parameters. Calls
  are serial, with SDK retries, redirects, and environment proxies disabled.
  Truncated, refused, tool, malformed, or oversized responses are rejected.
- Text is preserved without spelling/Unicode/Markdown cleanup. No confidence
  number is invented. Empty recognition on nonblank pages is unreadable; empty
  recognition on uniformly white pages remains empty. No layout regions triggers
  an explicitly warned full-page crop, still using actual OCR inference.
- Output is at most 10 MiB, with 10,000 characters per block, one million total
  text characters, and contract-shaped warnings. Overflow fails; it is not truncated.
- The worker uses a 300-second alarm/CPU limit, 6 GiB Linux address-space limit,
  bounded open files/file size, and disabled core dumps. The parent additionally
  owns the authoritative wall deadline, process group termination, vLLM occupancy
  tracking, and temporary-directory cleanup. Killing Python alone does not prove
  a remote vLLM request has stopped. Process exit and private temporary storage
  are required; Python object release is not a physical-memory-erasure guarantee.

The adapter does not identify separate documents inside a scan, infer document
type, interpret invoices, follow printed URLs, or execute printed instructions.
Those limitations must remain visible in rollout/readiness documentation.

## Versions and validation

Deployment resolves hashed dependencies separately. The worker checks PaddleOCR
3.7.0, PaddleX 3.7.2, PaddlePaddle CPU 3.2.1, Pillow 11.3.0, pypdfium2 4.30.0,
NumPy 2.2.6, and OpenAI client SDK 1.109.1 at startup. Python must be 3.12; the
deployment image pins its patch version. No model is downloaded by this worker.

The adapter intentionally uses the pinned layout predictor and recognition
`genai_client` below PaddleOCRVL. This avoids the general pipeline's parallel
region submission and text rewriting. The generic recognition predictor also
uses JPEG and discards `finish_reason`; the direct client retains the complete
completion so truncation is never published as a complete read.

The pinned construction path shallow-copies dictionaries and validates GenAI
configuration in Pydantic's Python mode; it preserves the opaque `http_client`
object. It does not call YAML export or deepcopy on this configuration. This
path was also checked by executing the upstream merge and GenAIConfig definitions
with a non-deepcopyable lock object and checking identity after validation/dump.
Full Paddle construction still belongs to container validation.

The API was checked against official versioned source distributions and these
upstream sources:

- [PaddleOCR 3.7.0 pipeline wrapper](https://github.com/PaddlePaddle/PaddleOCR/blob/v3.7.0/paddleocr/_pipelines/paddleocr_vl.py)
- [PaddleX 3.7.2 recognition predictor](https://github.com/PaddlePaddle/PaddleX/blob/v3.7.2/paddlex/inference/models/doc_vlm/predictor.py)
- [PaddleX 3.7.2 GenAI client](https://github.com/PaddlePaddle/PaddleX/blob/v3.7.2/paddlex/inference/models/common/genai.py)
- [Pillow 11.3.0 EXIF transpose](https://github.com/python-pillow/Pillow/blob/11.3.0/src/PIL/ImageOps.py)
- [pypdfium2 4.30.0 page geometry and rendering](https://github.com/pypdfium2-team/pypdfium2/blob/4.30.0/src/pypdfium2/_helpers/page.py)
- [OpenAI client 1.109.1](https://github.com/openai/openai-python/blob/v1.109.1/src/openai/_client.py)

Run the synthetic suite with a Python 3.12 environment containing only the three
decoder dependencies (Pillow, pypdfium2, NumPy); it does not require Paddle or a GPU:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest services.ocr.paddle.test_worker -v
```

The local run passed 22 tests using Python 3.12.8 and the exact decoder versions
above. Decoder tests use actual libraries and synthetic PNG/JPEG/WebP/PDF bytes.
Model-call tests explicitly substitute a fake pipeline; they verify the boundary,
not real OCR quality. A subprocess test verifies that missing Paddle dependencies
fail closed with a redacted response.

# Paddle worker

`worker.py` implements the private, single-job subprocess used by the TypeScript
provider. It decodes original bytes, runs PP-DocLayoutV3 on CPU, and recognizes
each measured region through PaddleOCR-VL-1.6-0.9B on the private vLLM server.
It returns page/block evidence; financial field extraction belongs to
`../normalize.ts`.

The worker is running in the private home-server deployment as of 23 October
2026. Model loading, CPU layout and vLLM recognition passed six generated-document
cases with full result validation and assistant-reviewed evidence overlays.
Observed case latency was approximately 9–15 seconds. These smoke cases do not
establish accuracy on representative Portuguese receipts or a completed finance-app
workflow. The [access guide](../../../docs/ocr/ACCESS.md) explains how to use the
service; the [runtime record](../../../docs/ocr/RUNTIME.md) preserves exact versions,
results and measured resource snapshots. Qwen remains explicitly paused for OCR.

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
Local operator-diagnostic addition (not deployed): resource failures may also carry
an optional fixed `error.resourceLimit` classification in this internal protocol.
The TypeScript parent validates it and emits only
`{"event":"ocr_resource_limit","reason":"pixel_budget"}` to its operator log.
Legacy errors emit `unspecified`; unknown classifications or classifications on
another error code are rejected. This field is not copied into the public API,
retained model result or finance-app error. Logging failure cannot change the
terminal job state, retry policy or cleanup.

Classifications distinguish page/region dimensions, cumulative pixel budget,
page/region count, token limit, per-block/aggregate text limits, memory allocation,
warning count, source bytes and input/output protocol size. No text, filename,
source hash, document ID, namespace, credential, geometry or exception detail is
logged. Dependency stdout/stderr remain discarded. `memory` means a caught Python
allocation failure, not proof of host OOM; `pixel_budget` is a software work
ceiling, not measured RAM consumption. A killed worker or unavailable output
cannot identify its cause. Existing failures cannot be retroactively classified.
Docker log timestamps can correlate future events with host observations without
reprocessing old originals merely for diagnostics. All limits remain unchanged.

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
Full Paddle construction subsequently passed in the pinned container, followed
by the generated-document recognition run recorded above.

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

The updated local run passed 26 tests using Python 3.12.8 and the exact decoder versions
above. Decoder tests use actual libraries and synthetic PNG/JPEG/WebP/PDF bytes.
Model-call tests explicitly substitute a fake pipeline; they verify the boundary,
not real OCR quality. A subprocess test verifies that missing Paddle dependencies
fail closed with a redacted response.


## Local phone-photo candidate — not deployed

The recorded fa7ca6a056fa/71483aa5… release is unchanged. Candidate source now bounds layout input to its inspected native target (at most1024 per side) and maps validated detector boxes outwards into the original EXIF-oriented reference frame. Full original bytes/hash/dimensions are retained. Recognition reads each ROI directly from that original; regions larger than the model's1,605,632-pixel bound are resized directly into the same pinned smart-resize dimensions **before** PNG encoding. Small original regions are retained; the model's existing minimum-pixel handling remains. No saved crop/reading view replaces the original.

Decode, layout resize output/separable scratch, bounded NumPy/native preprocessing, ROI resize and all recognition work remain charged before allocation/inference to the same100M cumulative budget. No page/region/byte/CPU/memory/token/concurrency ceiling is raised, no region is silently dropped and the entire crop list is preflighted before any region call. Camera-sized NumPy copies and camera-sized recognition PNGs are avoided. Output evidence uses original dimensions/normalized boxes, never working-image dimensions. Existing image/PDF/EXIF, BGR/RGB, malformed-output, byte integrity, cancellation and concurrency controls remain.

Local fake-inference regressions exercise actual12MP JPEG decode, full-reference crop/PNG preparation, the largest permitted layout target and EXIF portrait mapping within the existing budget. They do not establish actual model recognition quality. The candidate changes source identity and must receive new pinned image manifests/pipeline fingerprint before any deployment; do not patch the running container or reuse an old identity. Required next execution proof is the authorized private real-photo comparison plus the retained generated-document suite. Keep all real originals/results outside Git.

## Follow-up bounded-region candidate — not deployed

The first photo candidate passed generated checks but failed its real-photo gate and was rolled back; see the [operator update record](../../../work/ocr-service-update.md). Current local code plans adjacent-region coalescence only when independent recognition crops exceed remaining raster work. Components need overlapping horizontal spans and a vertical gap no greater than3% of original height. Every original detector box stays enclosed, original coordinates are preserved, and a coalesced component is used only if its charged work is lower. All-pairs component construction is bounded by the512-region limit. It does not infer document separation or change original files. The unchanged100M ceiling is reserved before any model call; a remaining over-budget plan still fails. Grouping emits an explicit page warning.

The34 worker tests use synthetic sources/fake recognition and cover camera-sized fragments, indentation/separate columns, original geometry/hash preservation and unaffordable plans before inference. The measured private diagnostic and offline plan are separate evidence; actual new-image recognition remains pending. Model parameters, dependency pins, queue/concurrency and limits are unchanged.

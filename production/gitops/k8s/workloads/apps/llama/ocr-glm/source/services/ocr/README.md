# Private OCR service

This directory implements the contract v1 API, durable queue, an isolated image/PDF decoder and a PaddleOCR-VL-1.6 + PP-DocLayoutV3 adapter backed by a dedicated private vLLM server. It also includes deterministic four-field normalization and pinned container build definitions. **The private service is running on the home server as of 23 October 2026.** Both images built, and six generated-document cases passed actual recognition and full result validation. This small smoke evaluation does not establish representative accuracy or a completed finance-app workflow. Use the [access guide](../../docs/ocr/ACCESS.md) for connection and request examples, and the [runtime record](../../docs/ocr/RUNTIME.md) for the deployed release and verification evidence.

Nothing starts on import. The separate `main.ts` entrypoint is disabled unless `OCR_ENABLED=real` and all startup checks pass. The finance app does not start this service or use its database. Existing mock audit checks remain intact.

## Validation and launch

```sh
npm run test:ocr
npm run check
```

Use a supported Node version (verified with 24.11.0). Tests use disposable private databases, generated documents and explicitly hand-authored provider output. No model or external service is called by this test command. Decoder tests and their isolated Python 3.12 environment are described in [paddle/README.md](paddle/README.md).

[deploy/README.md](deploy/README.md) documents pinned images, hashed dependency locks, model inventories, startup verification and an explicit Compose launch. The supplied example manifest intentionally refuses startup until actual model-file inventories, image identities and the adapter source hash are supplied; the running release has its own completed manifest. Containers use Python 3.12; the host Python installation is not modified. The vLLM service must be dedicated to this worker. Only the fixed-target ingress publishes host loopback port 9417; the API and model server remain on the internal Docker network. A finance app on another machine needs an authenticated tunnel or a separately designed authenticated TLS boundary; the current client permits only literal loopback destinations.

## Components

- `runtime.ts`: dedicated SQLite database, bearer-credential namespaces, stable request receipts, bounded durable jobs, fenced leases, cancellation and retention. Synthetic providers need `allowSynthetic`; real providers need `allowReal`. Both are disabled by default.
- `http.ts`: Fastify factory exposing the authenticated `/v1/ocr/*` contract routes. Exactly two multipart parts, bounded metadata/original bytes, declared MIME/signature and SHA-256 checks; no filename is resolved as a path. Responses are `no-store`, with a shared one-second per-job poll limit.
- `main.ts`: explicitly enabled container entrypoint. It validates credentials and the runtime manifest before opening the private listener, drives one worker, maintains leases/retention and schedules temporary-file cleanup.
- `provider.ts`: passes exact bytes to a private per-job directory and a killable Python process group, validates bounded protocol output, and normalizes evidence. It never retains child diagnostic output or forwards inherited service credentials. Cancellation/timeout kills the group and waits for descendants to disappear. It then retains the execution slot until dedicated vLLM metrics confirm no running or waiting inference, including if file cleanup failed.
- `paddle/`: strict local decoding, EXIF-correct image references and PDF CropBox/Rotate references; sequential pinned layout/region inference without embedded-PDF-text extraction. See its guide for limits and version-specific API use.
- `normalize.ts`: conservative merchant, issuance date, exact integer centavos and explicit currency evidence. The live release uses `receipt-core-ptbr-v1`; current local source is the undeployed `receipt-core-ptbr-v3` follow-up candidate. Missing/ambiguous values abstain; confidence is uncalibrated/null. [Normalization rules and limitations](../../docs/ocr/NORMALIZATION.md).

## Durability and resource bounds

Acceptance commits original bytes, metadata, a queued envelope and its scoped request-key receipt in one SQLite transaction before acknowledging `202`. Exact replays return the current envelope. A changed command under the same key conflicts. Explicit retries preserve source/profile/locale/pipeline and failed-parent lineage, with one child per failed job and at most three attempts. Restart never silently repeats inference.

Defaults are 16 queued jobs, one active document, a 30-second renewable lease and a 300-second execution deadline. The Python child deadline is 295 seconds, leaving time for termination. Pages and region requests are sequential; vLLM also limits active sequences to one. If the inference server cannot confirm idle, the service holds the execution slot and stops dispatching further documents. Restart refuses to proceed while the dedicated inference server remains unavailable or busy. The terminal API state can be recorded before inference drain completes; this does not release the active execution slot.

One live runtime owns the database through an exclusive process lock. A timed-out or cancelled provider retains ownership until its promise settles. A dead owner permits fenced recovery to `worker_interrupted`; late output cannot overwrite a terminal state. An abandoned recovery gate, malformed lock or reused live PID fails closed for operator inspection. This is single-host ownership, not a distributed lock.

Input and output are bounded to 10 MiB, metadata to 64 KiB. Decoder bounds include 50 pages, 16,000 pixels per side, 40 million pixels per page and 100 million aggregate raster pixels including charged processing passes. Layout regions, recognized text, output blocks and generation tokens have additional finite limits. Linux process limits and container CPU/memory/PID constraints apply alongside these logical budgets.

## Originals and retention

The database lives in a private directory (0700), with mode 0600 and rejection of unrelated existing tables. Terminal transitions remove the SQLite input BLOB and zero the runtime-owned in-memory input buffer. SQLite uses `secure_delete=ON`, `synchronous=FULL` and DELETE journaling. This does not promise physical-media, snapshot or backup erasure.

Each worker's 0700 scratch directory contains only the original and trusted worker configuration, with 0600 files. The parent removes it after the process group exits. A startup and periodic sweep removes stale owned `job-*` directories after nine minutes; normal scheduling gives a margin below the ten-minute retention target. Host/service downtime can delay cleanup. No unrelated names or symlinks are followed. Failed cleanup is reported without source contents and does not bypass the inference barrier.

Terminal envelopes remain for 24 hours; namespace/key/digest/job/outcome tombstones remain indefinitely. An expired accepted key returns `410` and cannot create replacement work. The finance app retains its own immutable results and separate review history; service expiry does not replace those app records.

## Pending acceptance

Image builds, pinned model inventories, hardware startup and six actual generated-document cases passed. Exact-request replay, queued cancellation and persistence of a completed result across an orderly API restart also passed. The local test command above does not run PaddleOCR or vLLM; live recognition evidence is recorded separately. Representative document accuracy, sustained throughput, peak memory, active-inference cancellation and abrupt-host-crash recovery still require evaluation. Qwen was explicitly paused by the user to free the GPU and remains paused; resume it only after stopping OCR and on the user's instruction. OCR restart after a host reboot or service failure is manual.

Finance-app outbox/job linkage, authorized original resolution, append-only migration33, employee-own/manager routes, transactional review and PDF/image evidence UI are implemented locally and independently reviewed. Manual entry and the existing mock audit remain available. Generated integration checks passed; four subsequently authorized real receipt/invoice trials exposed resource/parser gaps. The user authorized a controlled update; the first candidate was rolled back after real-photo failures. Current local worker/normalizer-v3 fixes need a new immutable release and actual recognition proof. [Update and rollback record](../../work/ocr-service-update.md). [App checkpoint](../../work/ocr-app-integration.md), [candidate and rollout plan](../../work/ocr-photo-readiness.md).

## Local resource diagnostics — not deployed

The local worker/provider adds validated fixed resource-limit classifications to operator logs, while preserving generic public errors, immutable results, nonretryable resource failures, cleanup and inference draining. No source text or identifiers are logged; child stderr is still discarded. A callback failure cannot change processing outcomes. See [internal protocol and limits](paddle/README.md#process-protocol). This requires a future immutable release identity; do not patch the current container. Server work is on hold pending investigation of reported disconnects.

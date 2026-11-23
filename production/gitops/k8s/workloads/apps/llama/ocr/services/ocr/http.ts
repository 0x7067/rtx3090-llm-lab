import { createHash, timingSafeEqual } from "node:crypto";
import Fastify, { type FastifyReply, type FastifyRequest } from "fastify";
import multipart from "@fastify/multipart";
import { ocrCancelCommandSchema, type OcrJob } from "../../shared/ocr.js";
import {
  OCR_MAX_INPUT_BYTES,
  OCR_MAX_METADATA_BYTES,
  OCR_MAX_OUTPUT_BYTES,
  OcrRuntime,
  OcrRuntimeError,
  requireRequestKey,
  type OcrExecutionLease,
} from "./runtime.js";

export interface OcrHttpOptions {
  runtime: OcrRuntime;
  /** Configure server-side; namespace is never accepted from document metadata. */
  credentials: Array<{ token: string; namespace: string }>;
  maxConcurrentUploads?: number;
  /** Shorter bounded clocks for socket tests; never supplied by service config. */
  executionTimingForTests?: {
    heartbeatMs: number;
    afterAcceptedMs: number;
    totalMs: number;
  };
}

/** No listening socket, background worker or model is started by this factory. */
export async function buildOcrHttpServer(options: OcrHttpOptions) {
  const app = Fastify({
    logger: false,
    bodyLimit: OCR_MAX_METADATA_BYTES,
    requestTimeout: 30_000,
    connectionTimeout: 30_000,
  });
  const credentials = options.credentials.map(({ token, namespace }) => {
    if (
      token.length < 24 ||
      token.length > 4096 ||
      /\s/.test(token) ||
      !namespace ||
      namespace.length > 200
    )
      throw new Error(
        "OCR credentials require a strong token and configured namespace.",
      );
    return { hash: createHash("sha256").update(token).digest(), namespace };
  });
  if (
    new Set(credentials.map((item) => item.hash.toString("hex"))).size !==
    credentials.length
  )
    throw new Error("OCR service tokens must be unique.");
  const limit = options.maxConcurrentUploads ?? 4;
  if (!Number.isInteger(limit) || limit < 1 || limit > 16)
    throw new Error("Invalid OCR upload concurrency.");
  let activeUploads = 0;
  const timing = options.executionTimingForTests ?? {
    heartbeatMs: 5000,
    afterAcceptedMs: 330_000,
    totalMs: 390_000,
  };
  if (
    Object.keys(timing).length !== 3 ||
    !Number.isSafeInteger(timing.heartbeatMs) ||
    timing.heartbeatMs < 1 ||
    timing.heartbeatMs > 5000 ||
    !Number.isSafeInteger(timing.afterAcceptedMs) ||
    timing.afterAcceptedMs < 1 ||
    timing.afterAcceptedMs > 330_000 ||
    !Number.isSafeInteger(timing.totalMs) ||
    timing.totalMs < timing.afterAcceptedMs ||
    timing.totalMs > 390_000 ||
    Math.ceil(timing.afterAcceptedMs / timing.heartbeatMs) > 80
  )
    throw new Error("Invalid bounded OCR execution timing.");
  const namespaces = new WeakMap<FastifyRequest, string>();

  await app.register(multipart, {
    // Treat both named parts as streams, including metadata without a filename.
    // This preserves its raw byte count and avoids implicit JSON field parsing.
    isPartAFile: () => true,
    limits: {
      parts: 2,
      files: 2,
      fields: 0,
      fileSize: OCR_MAX_INPUT_BYTES,
      fieldNameSize: 100,
      headerPairs: 20,
    },
  });
  app.addHook("onRequest", async (request, reply) => {
    reply.header("Cache-Control", "no-store");
    reply.header("X-Content-Type-Options", "nosniff");
    const authorization = request.headers.authorization;
    if (
      typeof authorization !== "string" ||
      !authorization.startsWith("Bearer ") ||
      authorization.length > 4103
    )
      throw new OcrRuntimeError(401, "unauthorized");
    const hash = createHash("sha256").update(authorization.slice(7)).digest();
    // Check every configured token to avoid namespace-position timing leaks.
    let namespace: string | undefined;
    for (const credential of credentials)
      if (timingSafeEqual(hash, credential.hash))
        namespace = credential.namespace;
    if (!namespace) throw new OcrRuntimeError(401, "unauthorized");
    namespaces.set(request, namespace);
  });
  app.setErrorHandler((error, request, reply) => {
    const code = (error as { code?: string }).code;
    let safe =
      error instanceof OcrRuntimeError
        ? error
        : code === "FST_REQ_FILE_TOO_LARGE" ||
            code === "FST_ERR_CTP_BODY_TOO_LARGE"
          ? new OcrRuntimeError(413, "resource_limit")
          : code?.startsWith("FST_") ||
              (error as { statusCode?: number }).statusCode === 400
            ? new OcrRuntimeError(400, "invalid_request")
            : new OcrRuntimeError(503, "service_unavailable", true);
    if (safe.status === 429 || safe.status === 503)
      reply.header("Retry-After", "1");
    let requestKey: string | null = null;
    try {
      requestKey = requireRequestKey(request.headers["idempotency-key"]);
    } catch {
      /* No valid key supplied. */
    }
    reply.code(safe.status).send({
      contractVersion: 1,
      requestKey,
      error: {
        code: safe.code,
        message: safe.message,
        retryable: safe.retryable,
      },
    });
  });
  app.setNotFoundHandler(() => {
    throw new OcrRuntimeError(404, "not_found");
  });

  function respond(reply: FastifyReply, job: OcrJob, status = 200) {
    if (["queued", "running", "cancelling"].includes(job.state))
      reply.header("Retry-After", "1");
    return reply.code(status).send(job);
  }
  async function acceptUpload<T>(
    request: FastifyRequest,
    accept: (metadata: unknown, original: Buffer, mime: string) => T,
  ): Promise<T> {
    if (!request.isMultipart())
      throw new OcrRuntimeError(400, "invalid_request");
    if (activeUploads >= limit)
      throw new OcrRuntimeError(429, "rate_limited", true);
    activeUploads++;
    let original: Buffer | undefined;
    try {
      let metadata: unknown;
      let metadataSeen = false;
      let fileMime: string | undefined;
      let count = 0;
      for await (const part of request.parts()) {
        count++;
        if (
          part.type !== "file" ||
          !["metadata", "file"].includes(part.fieldname) ||
          count > 2
        )
          throw new OcrRuntimeError(400, "invalid_request");
        const isMetadata = part.fieldname === "metadata";
        if ((isMetadata && metadataSeen) || (!isMetadata && original))
          throw new OcrRuntimeError(400, "invalid_request");
        if (isMetadata && part.mimetype !== "application/json")
          throw new OcrRuntimeError(400, "invalid_request");
        const buffers: Buffer[] = [];
        let size = 0;
        for await (const chunk of part.file) {
          size += chunk.length;
          if (
            size > (isMetadata ? OCR_MAX_METADATA_BYTES : OCR_MAX_INPUT_BYTES)
          )
            throw new OcrRuntimeError(413, "resource_limit");
          buffers.push(Buffer.from(chunk));
        }
        if (part.file.truncated)
          throw new OcrRuntimeError(413, "resource_limit");
        const bytes = Buffer.concat(buffers);
        if (isMetadata) {
          metadataSeen = true;
          try {
            metadata = JSON.parse(
              new TextDecoder("utf-8", { fatal: true }).decode(bytes),
            );
          } catch {
            throw new OcrRuntimeError(400, "invalid_request");
          }
        } else {
          original = bytes;
          fileMime = part.mimetype;
        }
      }
      if (count !== 2 || !metadataSeen || !original || !fileMime)
        throw new OcrRuntimeError(400, "invalid_request");
      return accept(metadata, original, fileMime);
    } finally {
      original?.fill(0);
      activeUploads--;
    }
  }
  app.post("/v1/ocr/jobs", async (request, reply) => {
    const key = requireRequestKey(request.headers["idempotency-key"]);
    const result = await acceptUpload(request, (metadata, original, mime) =>
      options.runtime.accept(
        namespaces.get(request)!,
        key,
        metadata,
        original,
        mime,
      ),
    );
    reply.header("Location", `/v1/ocr/jobs/${result.job.jobId}`);
    return respond(reply, result.job, result.replay ? 200 : 202);
  });

  async function streamExecution(
    request: FastifyRequest,
    reply: FastifyReply,
    acquire: (isClosed: () => boolean) => Promise<OcrExecutionLease>,
  ) {
    let lease: OcrExecutionLease | undefined;
    let closed = false;
    let normalEnd = false;
    let started = false;
    let interrupted: Promise<void> | undefined;
    let acceptedTimer: ReturnType<typeof setTimeout> | undefined;
    let heartbeat: ReturnType<typeof setInterval> | undefined;
    const interrupt = () => {
      if (lease) interrupted ??= lease.interruptAndDrain();
      void interrupted?.catch(() => {});
    };
    const disconnect = () => {
      if (normalEnd) return;
      closed = true;
      if (heartbeat) clearInterval(heartbeat);
      if (acceptedTimer) clearTimeout(acceptedTimer);
      interrupt();
    };
    const stop = () => {
      disconnect();
      reply.raw.destroy();
    };
    // IncomingMessage.close also fires after a normal completed upload. The
    // response socket owns this lease and is the only disconnect signal here.
    reply.raw.on("close", disconnect);
    reply.raw.on("error", disconnect);
    const totalTimer = setTimeout(stop, timing.totalMs);
    try {
      lease = await acquire(() => closed || reply.raw.destroyed);
      if (closed || reply.raw.destroyed) {
        interrupt();
        await interrupted;
        return;
      }
      const replayTerminal = !["queued", "running", "cancelling"].includes(
        lease.job.state,
      );
      const maxTotal =
        (replayTerminal
          ? 2 * OCR_MAX_OUTPUT_BYTES
          : OCR_MAX_METADATA_BYTES + OCR_MAX_OUTPUT_BYTES) + 8192;
      let bytesWritten = 0,
        overhead = 0,
        comments = 0;
      const frame = (event: "accepted" | "terminal", job: OcrJob) => {
        const json = JSON.stringify(job),
          size = Buffer.byteLength(json);
        if (
          size > OCR_MAX_OUTPUT_BYTES ||
          (event === "accepted" &&
            !replayTerminal &&
            size > OCR_MAX_METADATA_BYTES)
        )
          throw new OcrRuntimeError(413, "resource_limit");
        const data = Buffer.from(`event: ${event}\ndata: ${json}\n\n`);
        overhead += data.length - size;
        bytesWritten += data.length;
        if (overhead > 8192 || bytesWritten > maxTotal)
          throw new OcrRuntimeError(413, "resource_limit");
        return data;
      };
      const write = (data: Buffer) =>
        new Promise<void>((resolve, reject) => {
          const lost = () => finish(new Error("execution-response-closed"));
          const finish = (error?: Error | null) => {
            reply.raw.off("close", lost);
            reply.raw.off("error", lost);
            if (error) reject(error);
            else resolve();
          };
          reply.raw.once("close", lost);
          reply.raw.once("error", lost);
          if (closed || reply.raw.destroyed) return lost();
          try {
            reply.raw.write(data, finish);
          } catch {
            lost();
          }
        });
      const accepted = frame("accepted", lease.job);
      reply.hijack();
      started = true;
      reply.raw.writeHead(lease.replay ? 200 : 202, {
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
        "Content-Type": "text/event-stream",
        Location: `/v1/ocr/jobs/${lease.job.jobId}`,
        // The deployed credential bridge rewrites one request per connection.
        // Closing after EOF preserves the entire stream hold while preventing
        // a later request from reusing an already-authenticated raw tunnel.
        Connection: "close",
        "X-Accel-Buffering": "no",
      });
      reply.raw.flushHeaders();
      await write(accepted);
      acceptedTimer = setTimeout(stop, timing.afterAcceptedMs);
      heartbeat = setInterval(() => {
        if (closed) return;
        const comment = Buffer.from(": keepalive\n\n");
        comments++;
        overhead += comment.length;
        bytesWritten += comment.length;
        if (comments > 80 || overhead > 8192 || bytesWritten > maxTotal)
          return stop();
        try {
          reply.raw.write(comment, (error) => {
            if (error) stop();
          });
        } catch {
          stop();
        }
      }, timing.heartbeatMs);
      // The first acknowledged bytes precede claim/provider entry. This promise
      // includes provider cleanup, so the manager hold spans the entire run.
      const terminal = await lease.execute();
      if (closed) return;
      clearInterval(heartbeat);
      heartbeat = undefined;
      await write(frame("terminal", terminal));
      normalEnd = true;
      reply.raw.end();
    } catch (error) {
      if (lease) {
        interrupt();
        reply.raw.destroy();
        await interrupted;
      } else if (!started && !closed) throw error;
    } finally {
      clearTimeout(totalTimer);
      if (acceptedTimer) clearTimeout(acceptedTimer);
      if (heartbeat) clearInterval(heartbeat);
      reply.raw.off("close", disconnect);
      reply.raw.off("error", disconnect);
    }
  }
  // find-my-way requires a double colon to register one literal URL colon.
  app.post("/v1/ocr/jobs::execute", async (request, reply) => {
    const key = requireRequestKey(request.headers["idempotency-key"]);
    return streamExecution(request, reply, async (isClosed) =>
      acceptUpload(request, (metadata, original, mime) => {
        if (isClosed())
          throw new OcrRuntimeError(503, "service_unavailable", true);
        return options.runtime.acceptExecution(
          namespaces.get(request)!,
          key,
          metadata,
          original,
          mime,
        );
      }),
    );
  });
  app.get<{ Params: { requestKey: string } }>(
    "/v1/ocr/requests/:requestKey/execute",
    async (request, reply) => {
      const key = requireRequestKey(request.params.requestKey);
      return streamExecution(request, reply, async (isClosed) => {
        if (isClosed())
          throw new OcrRuntimeError(503, "service_unavailable", true);
        return options.runtime.leaseExistingExecution(
          namespaces.get(request)!,
          key,
        );
      });
    },
  );
  app.get<{ Params: { jobId: string } }>(
    "/v1/ocr/jobs/:jobId",
    async (request, reply) =>
      respond(
        reply,
        options.runtime.get(
          namespaces.get(request)!,
          requireRequestKey(request.params.jobId),
          true,
        ),
      ),
  );
  app.get<{ Params: { requestKey: string } }>(
    "/v1/ocr/requests/:requestKey",
    async (request, reply) =>
      respond(
        reply,
        options.runtime.lookup(
          namespaces.get(request)!,
          requireRequestKey(request.params.requestKey),
          true,
        ),
      ),
  );
  app.post<{ Params: { jobId: string } }>(
    "/v1/ocr/jobs/:jobId/cancel",
    async (request, reply) => {
      const parsed = ocrCancelCommandSchema.safeParse(request.body);
      if (!parsed.success) throw new OcrRuntimeError(400, "invalid_request");
      const job = options.runtime.cancel(
        namespaces.get(request)!,
        requireRequestKey(request.params.jobId),
        requireRequestKey(request.headers["idempotency-key"]),
        parsed.data.version,
      );
      return respond(reply, job, job.state === "cancelling" ? 202 : 200);
    },
  );
  return app;
}

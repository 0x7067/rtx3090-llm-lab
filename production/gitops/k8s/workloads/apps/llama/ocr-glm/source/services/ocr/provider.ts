import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import {
  lstat,
  mkdir,
  mkdtemp,
  readdir,
  realpath,
  rm,
  writeFile,
} from "node:fs/promises";
import path from "node:path";
import { z } from "zod";
import {
  ocrPipelineSchema,
  hasValidOcrReaderProvenance,
  ocrRecognizedPagesSchema,
  type OcrPipeline,
  type OcrRequest,
} from "../../shared/ocr.js";
import { normalizeOcrPages } from "./normalize.js";

const executionCodes = [
  "resource_limit",
  "invalid_document",
  "encrypted_document",
  "unsupported_media",
  "timeout",
  "provider_unavailable",
  "provider_failure",
  "invalid_provider_output",
  "worker_interrupted",
] as const;
export type OcrExecutionCode = (typeof executionCodes)[number];
/** Retains only stable redacted operational information, never child stderr/text. */
export class OcrExecutionError extends Error {
  constructor(
    readonly code: OcrExecutionCode,
    readonly retryable: boolean,
  ) {
    super(`OCR execution failed (${code}).`);
    this.name = "OcrExecutionError";
  }
}

const warningSchema = z
  .object({
    code: z.enum([
      "low_quality",
      "multiple_documents",
      "text_render_mismatch",
      "unsupported_currency",
      "page_unreadable",
      "ambiguous_field",
      "unreadable_field",
      "missing_field",
    ]),
    message: z.string().min(1).max(500),
    page: z.number().int().min(1).max(50).nullable(),
  })
  .strict();
const successSchema = z
  .object({
    protocolVersion: z.literal(1),
    methodsUsed: z.tuple([z.literal("ocr")]),
    pages: ocrRecognizedPagesSchema,
    warnings: z.array(warningSchema).max(100),
  })
  .strict();
const failureSchema = z
  .object({
    protocolVersion: z.literal(1),
    error: z
      .object({
        code: z.enum(executionCodes),
        message: z.string().max(500),
        retryable: z.boolean(),
        resourceLimit: z
          .enum([
            "page_dimensions",
            "region_dimensions",
            "pixel_budget",
            "page_count",
            "region_count",
            "token_limit",
            "block_text",
            "total_text",
            "memory",
            "warning_count",
            "input_bytes",
            "input_protocol",
            "output_protocol",
          ])
          .optional(),
      })
      .strict(),
  })
  .strict()
  .refine(
    (value) =>
      value.error.resourceLimit === undefined ||
      value.error.code === "resource_limit",
  );

export type OcrResourceLimitDiagnostic = {
  event: "ocr_resource_limit";
  reason:
    | NonNullable<z.infer<typeof failureSchema>["error"]["resourceLimit"]>
    | "unspecified";
};

export interface OcrWorkerProviderOptions {
  pipeline: OcrPipeline;
  /** Trusted immutable startup config derived by deployment preflight. */
  workerConfig: Record<string, unknown>;
  python: string;
  worker: string;
  workRoot: string;
  /** Required only by the isolated GLM Python runtime in the pinned Linux image. */
  pythonLibraryPath?: "/opt/py312/lib";
  timeoutMs?: number;
  killGraceMs?: number;
  /** Receives only validated fixed classifications, never worker text or source IDs. */
  onResourceLimit?: (diagnostic: OcrResourceLimitDiagnostic) => void;
}
export type PaddleProviderOptions = OcrWorkerProviderOptions;
export interface PaddleProviderDependencies {
  /** Trusted engine adapter or test injection; never a document/HTTP parameter. */
  inferenceIdle?: () => Promise<boolean>;
}

/** Real-provider process boundary. Model dependencies are never installed into the host interpreter. */
export class OcrWorkerProvider {
  readonly pipeline: OcrPipeline;
  private readonly configBytes: string;
  private readonly timeout: number;
  private readonly grace: number;
  private closing = false;
  private readonly inferenceIdle: () => Promise<boolean>;
  constructor(
    private readonly options: OcrWorkerProviderOptions,
    dependencies: PaddleProviderDependencies = {},
  ) {
    this.pipeline = ocrPipelineSchema.parse(options.pipeline);
    if (
      this.pipeline.executionMode !== "real" ||
      this.pipeline.interpretationModel !== null
    )
      throw Error(
        "Worker provider requires a real pipeline without an external interpreter.",
      );
    if (
      ![options.python, options.worker, options.workRoot].every((item) =>
        path.isAbsolute(item),
      )
    )
      throw Error(
        "Paddle provider paths must be absolute trusted configuration.",
      );
    this.configBytes = JSON.stringify(options.workerConfig);
    if (
      options.pythonLibraryPath !== undefined &&
      options.pythonLibraryPath !== "/opt/py312/lib"
    )
      throw Error("Invalid pinned Python library path.");
    if (Buffer.byteLength(this.configBytes) > 65536)
      throw Error("Worker configuration is too large.");
    this.timeout = options.timeoutMs ?? 295_000;
    this.grace = options.killGraceMs ?? 500;
    if (
      !Number.isInteger(this.timeout) ||
      this.timeout < 1 ||
      this.timeout > 295_000 ||
      !Number.isInteger(this.grace) ||
      this.grace < 1 ||
      this.grace > 1000
    )
      throw Error("Invalid bounded worker deadlines.");
    if (
      this.pipeline.inferenceEngine.name !== "vllm" &&
      typeof dependencies.inferenceIdle !== "function"
    )
      throw Error(
        "A native OCR engine requires its own inference drain check.",
      );
    this.inferenceIdle =
      dependencies.inferenceIdle ?? (() => this.checkInferenceIdle());
  }

  private async checkInferenceIdle(): Promise<boolean> {
    const configured = this.options.workerConfig.vllmBaseUrl;
    if (typeof configured !== "string") return false;
    const base = new URL(configured),
      controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 2000);
    try {
      const response = await fetch(new URL("/metrics", base), {
        redirect: "error",
        signal: controller.signal,
      });
      if (!response.ok || !response.body) return false;
      let size = 0;
      const chunks: Buffer[] = [];
      const reader = response.body.getReader();
      try {
        for (;;) {
          const part = await reader.read();
          if (part.done) break;
          size += part.value.length;
          if (size > 1_048_576) {
            controller.abort();
            await reader.cancel();
            return false;
          }
          chunks.push(Buffer.from(part.value));
        }
      } finally {
        reader.releaseLock();
      }
      const metrics = Buffer.concat(chunks).toString("utf8");
      const counts = (name: string) =>
        [
          ...metrics.matchAll(
            new RegExp(
              `^vllm:${name}(?:\\{[^\\n]*\\})?\\s+([0-9]+(?:\\.[0-9]+)?)\\s*$`,
              "gm",
            ),
          ),
        ].map((m) => Number(m[1]));
      const running = counts("num_requests_running"),
        waiting = counts("num_requests_waiting");
      return (
        running.length > 0 &&
        waiting.length > 0 &&
        [...running, ...waiting].every((n) => n === 0)
      );
    } catch {
      return false;
    } finally {
      clearTimeout(timer);
    }
  }

  private async drainInference(): Promise<void> {
    // Client-process exit alone does not acknowledge cancellation inside vLLM.
    // Hold the runtime slot until its dedicated server reports no pending work.
    while (!this.closing) {
      if (await this.inferenceIdle().catch(() => false)) return;
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
    throw new OcrExecutionError("worker_interrupted", true);
  }
  close(): void {
    this.closing = true;
  }

  async prepare(): Promise<void> {
    await mkdir(this.options.workRoot, { recursive: true, mode: 0o700 });
    const root = await lstat(this.options.workRoot);
    if (
      !root.isDirectory() ||
      root.isSymbolicLink() ||
      (root.mode & 0o077) !== 0 ||
      (await realpath(this.options.workRoot)) !== this.options.workRoot
    )
      throw Error("OCR work storage must be a private canonical directory.");
    const worker = await lstat(this.options.worker);
    if (!worker.isFile() || worker.isSymbolicLink())
      throw Error("OCR worker must be a regular trusted file.");
    if (!(await this.inferenceIdle()))
      throw Error(
        "Dedicated vLLM server is unavailable or still processing work.",
      );
    await this.cleanup();
  }

  async cleanup(): Promise<void> {
    // Crashed-worker leftovers have a finite lifetime. Never follow links or erase
    // unrelated names; active subprocesses are bounded to less than five minutes.
    for (const name of await readdir(this.options.workRoot)) {
      if (!/^job-[A-Za-z0-9]{6}$/.test(name)) continue;
      const target = path.join(this.options.workRoot, name),
        stat = await lstat(target);
      if (
        stat.isDirectory() &&
        !stat.isSymbolicLink() &&
        Date.now() - stat.mtimeMs >= 540_000
      )
        await rm(target, { recursive: true });
    }
  }

  async analyze(
    input: { request: OcrRequest; original: Buffer },
    signal: AbortSignal,
  ) {
    if (signal.aborted) throw new OcrExecutionError("worker_interrupted", true);
    const source = input.request.source;
    if (
      input.original.length !== source.sizeBytes ||
      source.sizeBytes > 10_485_760 ||
      createHash("sha256").update(input.original).digest("hex") !==
        source.originalSha256
    )
      throw new OcrExecutionError("invalid_document", false);
    const jobDir = await mkdtemp(path.join(this.options.workRoot, "job-"));
    try {
      const originalPath = path.join(jobDir, "original"),
        configPath = path.join(jobDir, "worker.json");
      await writeFile(originalPath, input.original, {
        mode: 0o600,
        flag: "wx",
      });
      await writeFile(configPath, this.configBytes, {
        mode: 0o600,
        flag: "wx",
      });
      const raw = await this.execute(
        originalPath,
        configPath,
        {
          protocolVersion: 1,
          source: {
            mime: source.mime,
            sizeBytes: source.sizeBytes,
            originalSha256: source.originalSha256,
          },
        },
        signal,
      );
      const parsed = successSchema.safeParse(raw);
      if (!parsed.success)
        throw new OcrExecutionError("invalid_provider_output", false);
      if (!hasValidOcrReaderProvenance(parsed.data.pages, this.pipeline, source.originalSha256))
        throw new OcrExecutionError("invalid_provider_output", false);
      try {
        return normalizeOcrPages({
          pages: parsed.data.pages,
          mode: "real",
          methodsUsed: parsed.data.methodsUsed,
          warnings: parsed.data.warnings,
        });
      } catch {
        throw new OcrExecutionError("invalid_provider_output", false);
      }
    } finally {
      try {
        await rm(jobDir, { recursive: true, force: true });
      } finally {
        await this.drainInference();
      }
    }
  }

  private execute(
    originalPath: string,
    configPath: string,
    command: unknown,
    signal: AbortSignal,
  ): Promise<unknown> {
    return new Promise((resolve, reject) => {
      if (signal.aborted) {
        reject(new OcrExecutionError("worker_interrupted", true));
        return;
      }
      const child = spawn(
        this.options.python,
        [this.options.worker, "--input", originalPath, "--config", configPath],
        {
          detached: true,
          cwd: path.dirname(originalPath),
          stdio: ["pipe", "pipe", "pipe"],
          env: {
            PATH: process.env.PATH ?? "/usr/local/bin:/usr/bin:/bin",
            LANG: "C.UTF-8",
            LC_ALL: "C.UTF-8",
            PYTHONNOUSERSITE: "1",
            PYTHONDONTWRITEBYTECODE: "1",
            HF_HUB_OFFLINE: "1",
            PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK: "True",
            OMP_NUM_THREADS: "2",
            OPENBLAS_NUM_THREADS: "2",
            ...(this.options.pythonLibraryPath
              ? { LD_LIBRARY_PATH: this.options.pythonLibraryPath }
              : {}),
          },
        },
      );
      let error: OcrExecutionError | undefined;
      let graceTimer: ReturnType<typeof setTimeout> | undefined;
      const chunks: Buffer[] = [];
      let bytes = 0;
      let stderrBytes = 0;
      const kill = (kind: NodeJS.Signals) => {
        if (child.pid)
          try {
            process.kill(-child.pid, kind);
          } catch {
            /* Already exited. */
          }
      };
      const stop = (failure: OcrExecutionError, immediately = false) => {
        if (error) return;
        error = failure;
        kill(immediately ? "SIGKILL" : "SIGTERM");
        if (!immediately)
          graceTimer = setTimeout(() => kill("SIGKILL"), this.grace);
      };
      const abort = () =>
        stop(new OcrExecutionError("worker_interrupted", true));
      signal.addEventListener("abort", abort, { once: true });
      const timer = setTimeout(
        () => stop(new OcrExecutionError("timeout", true)),
        this.timeout,
      );
      child.stdout.on("data", (chunk: Buffer) => {
        bytes += chunk.length;
        if (bytes > 10_485_760)
          stop(new OcrExecutionError("invalid_provider_output", false), true);
        else if (!error) chunks.push(chunk);
      });
      child.stderr.on("data", (chunk: Buffer) => {
        stderrBytes += chunk.length;
        if (stderrBytes > 1_048_576)
          stop(new OcrExecutionError("provider_failure", false), true);
        // Drain but never retain stderr; library diagnostics may include source text.
      });
      child.stdin.on("error", () => {
        /* Exit handler decides the redacted outcome. */
      });
      child.on("error", () => {
        error ??= new OcrExecutionError("provider_unavailable", true);
      });
      child.on("close", async (code) => {
        clearTimeout(timer);
        if (graceTimer) clearTimeout(graceTimer);
        signal.removeEventListener("abort", abort);
        // A direct child can exit while a descendant ignores SIGTERM and has
        // already closed its inherited pipes. Kill the entire remaining group
        // even on a successful exit, then retain the slot until init reaps it.
        kill("SIGKILL");
        while (child.pid) {
          try {
            process.kill(-child.pid, 0);
          } catch (failure) {
            if ((failure as NodeJS.ErrnoException).code === "ESRCH") break;
            // An unexpected permission failure is not proof the group exited.
          }
          await new Promise((done) => setTimeout(done, 25));
        }
        if (error) {
          reject(error);
          return;
        }
        let result: unknown;
        try {
          result = JSON.parse(
            new TextDecoder("utf-8", { fatal: true }).decode(
              Buffer.concat(chunks),
            ),
          );
        } catch {
          reject(new OcrExecutionError("invalid_provider_output", false));
          return;
        }
        if (code !== 0) {
          const failure = failureSchema.safeParse(result);
          if (failure.success && failure.data.error.code === "resource_limit") {
            try {
              this.options.onResourceLimit?.({
                event: "ocr_resource_limit",
                reason: failure.data.error.resourceLimit ?? "unspecified",
              });
            } catch {
              // Operator logging must never change a job's outcome or cleanup.
            }
          }
          reject(
            failure.success
              ? new OcrExecutionError(
                  failure.data.error.code,
                  failure.data.error.retryable &&
                    [
                      "provider_unavailable",
                      "provider_failure",
                      "timeout",
                      "worker_interrupted",
                    ].includes(failure.data.error.code),
                )
              : new OcrExecutionError("invalid_provider_output", false),
          );
          return;
        }
        resolve(result);
      });
      child.stdin.end(JSON.stringify(command));
      if (signal.aborted) abort();
    });
  }
}

/** The original provider keeps its exact model/engine identity boundary. */
export class PaddleOcrProvider extends OcrWorkerProvider {
  constructor(
    options: PaddleProviderOptions,
    dependencies: PaddleProviderDependencies = {},
  ) {
    super(options, dependencies);
    if (
      this.pipeline.ocrModel.id !== "PaddlePaddle/PaddleOCR-VL-1.6" ||
      this.pipeline.inferenceEngine.name !== "vllm"
    )
      throw Error("Paddle provider requires PaddleOCR-VL-1.6 with vLLM.");
  }
}

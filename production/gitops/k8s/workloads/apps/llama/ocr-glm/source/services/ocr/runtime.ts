import { createHash, randomUUID } from "node:crypto";
import {
  chmodSync,
  closeSync,
  lstatSync,
  mkdirSync,
  openSync,
  readFileSync,
  realpathSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import path from "node:path";
import { DatabaseSync } from "node:sqlite";
import { OcrExecutionError, type OcrExecutionCode } from "./provider.js";
import {
  ocrJobSchema,
  ocrPipelineSchema,
  ocrRequestSchema,
  type OcrJob,
  type OcrPipeline,
  type OcrRequest,
} from "../../shared/ocr.js";

export const OCR_MAX_INPUT_BYTES = 10_485_760;
export const OCR_MAX_METADATA_BYTES = 65_536;
export const OCR_MAX_OUTPUT_BYTES = 10_485_760;
const DAY = 86_400_000;
const UUID =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
const terminal = (state: OcrJob["state"]) =>
  !["queued", "running", "cancelling"].includes(state);

/** Stable messages deliberately never contain document/provider input. */
const messages = {
  invalid_request: "Invalid OCR request.",
  unsupported_contract: "Unsupported OCR contract version.",
  hash_mismatch: "Original hash does not match metadata.",
  size_mismatch: "Original size does not match metadata.",
  invalid_document: "The original document is invalid.",
  unsupported_media: "Unsupported original media type.",
  resource_limit: "The OCR resource limit was exceeded.",
  unauthorized: "Service authentication required.",
  forbidden: "Service access is not permitted.",
  not_found: "OCR job was not found.",
  expired: "The retained OCR job has expired.",
  idempotency_conflict: "The request key belongs to another command.",
  version_conflict: "The OCR job version has changed.",
  terminal_conflict: "The OCR job already has a terminal outcome.",
  invalid_retry: "This OCR execution cannot be retried.",
  pipeline_unavailable: "The requested OCR pipeline is unavailable.",
  rate_limited: "The OCR capacity limit was reached.",
  service_unavailable: "The OCR service is unavailable.",
} as const;

export class OcrRuntimeError extends Error {
  constructor(
    readonly status: number,
    readonly code: keyof typeof messages,
    readonly retryable = false,
  ) {
    super(messages[code]);
  }
}

export function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value !== null && typeof value === "object") {
    return `{${Object.entries(value)
      .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
      .map(([key, item]) => `${JSON.stringify(key)}:${canonicalJson(item)}`)
      .join(",")}}`;
  }
  return JSON.stringify(value);
}
const digest = (value: string | Uint8Array) =>
  createHash("sha256").update(value).digest("hex");

export function requireRequestKey(value: unknown): string {
  if (typeof value !== "string" || !UUID.test(value))
    throw new OcrRuntimeError(400, "invalid_request");
  return value;
}

/** Signature checking only. This synthetic transport harness has no hardened decoder. */
function signatureMime(bytes: Buffer): OcrRequest["source"]["mime"] | null {
  if (
    bytes.length >= 8 &&
    bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))
  )
    return "image/png";
  if (
    bytes.length >= 3 &&
    bytes[0] === 255 &&
    bytes[1] === 216 &&
    bytes[2] === 255
  )
    return "image/jpeg";
  if (
    bytes.length >= 12 &&
    bytes.toString("ascii", 0, 4) === "RIFF" &&
    bytes.toString("ascii", 8, 12) === "WEBP"
  )
    return "image/webp";
  if (bytes.length >= 5 && bytes.toString("ascii", 0, 5) === "%PDF-")
    return "application/pdf";
  return null;
}

export interface OcrProvider {
  readonly pipeline: OcrPipeline;
  /** The provider reports actual executed stages; the runtime validates its immutable result. */
  analyze(
    input: { request: OcrRequest; original: Buffer },
    signal: AbortSignal,
  ): Promise<{
    state: "completed" | "partial";
    result: unknown;
  }>;
}
export type SyntheticOcrProvider = OcrProvider;

export interface OcrRuntimeOptions {
  /** A dedicated service database, never the finance application's database. */
  databasePath: string;
  allowSynthetic?: boolean;
  allowReal?: boolean;
  providers?: OcrProvider[];
  queueCapacity?: number;
  documentConcurrency?: number;
  executionTimeoutMs?: number;
  leaseMs?: number;
  now?: () => number;
}

interface JobRow {
  job_id: string;
  namespace: string;
  envelope: string;
  metadata: string;
  original: Uint8Array | null;
  lease_token: string | null;
  lease_until: number | null;
  execution_deadline: number | null;
  terminal_at: number | null;
  poll_after: number;
}
interface ReceiptRow {
  action: "submit" | "cancel";
  digest: string;
  job_id: string;
}
interface OcrClaim {
  readonly namespace: string;
  readonly job: OcrJob;
  readonly request: OcrRequest;
  readonly original: Buffer;
  readonly token: string;
  readonly deadline: number;
}

export interface OcrExecutionLease {
  readonly job: OcrJob;
  readonly replay: boolean;
  /** Called only after the accepted response has actually been written. */
  execute(): Promise<OcrJob>;
  /** Transport loss is an interruption, never an implicit user cancellation. */
  interruptAndDrain(): Promise<void>;
}

interface ExecutionOwner {
  namespace: string;
  jobId: string;
  token: string;
  execution?: Promise<OcrJob>;
}

/**
 * Durable private queue. Real providers require explicit enablement after startup preflight.
 */
export class OcrRuntime {
  private readonly db: DatabaseSync;
  private readonly providers = new Map<string, OcrProvider>();
  private readonly now: () => number;
  private readonly capacity: number;
  private readonly concurrency: number;
  private readonly timeout: number;
  private readonly lease: number;
  private readonly active = new Map<string, AbortController>();
  private readonly activeInputs = new Map<string, Buffer>();
  private readonly executions = new Map<string, Promise<unknown>>();
  private executionOwner: ExecutionOwner | undefined;
  private readonly lockPath: string;
  private readonly lockToken = randomUUID();
  private closed = false;

  constructor(options: OcrRuntimeOptions) {
    this.now = options.now ?? Date.now;
    this.capacity = options.queueCapacity ?? 16;
    this.concurrency = options.documentConcurrency ?? 1;
    this.timeout = options.executionTimeoutMs ?? 300_000;
    this.lease = options.leaseMs ?? 30_000;
    for (const [value, max] of [
      [this.capacity, 1000],
      [this.concurrency, 8],
      [this.timeout, 300_000],
      [this.lease, 300_000],
    ]) {
      if (!Number.isSafeInteger(value) || value < 1 || value > max)
        throw new Error("Invalid OCR runtime limits.");
    }
    for (const provider of options.providers ?? []) {
      const pipeline = ocrPipelineSchema.parse(provider.pipeline);
      if (
        (pipeline.executionMode === "mock"
          ? !options.allowSynthetic
          : !options.allowReal) ||
        this.providers.has(pipeline.id)
      )
        throw new Error("Only explicitly enabled OCR providers are available.");
      this.providers.set(pipeline.id, {
        pipeline,
        analyze: provider.analyze.bind(provider),
      });
    }
    if (
      !path.isAbsolute(options.databasePath) ||
      options.databasePath === ":memory:"
    )
      throw new Error("OCR requires an absolute dedicated database path.");
    mkdirSync(path.dirname(options.databasePath), {
      recursive: true,
      mode: 0o700,
    });
    const directory = realpathSync(path.dirname(options.databasePath));
    if ((lstatSync(directory).mode & 0o077) !== 0)
      throw new Error("OCR requires a private service data directory.");
    const databasePath = path.join(
      directory,
      path.basename(options.databasePath),
    );
    try {
      if (lstatSync(databasePath).isSymbolicLink())
        throw new Error("OCR database cannot be a symbolic link.");
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    }
    this.lockPath = `${databasePath}.owner`;
    this.acquireLock();
    let openedDatabase: DatabaseSync | undefined;
    try {
      this.db = openedDatabase = new DatabaseSync(databasePath);
      const unrelated = this.db
        .prepare(
          "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' AND name NOT IN ('ocr_jobs','ocr_receipts')",
        )
        .get();
      if (unrelated) {
        this.db.close();
        throw new Error("OCR requires a separate service database.");
      }
      chmodSync(databasePath, 0o600);
      // DELETE journaling avoids retained WAL copies of input BLOBs. secure_delete
      // clears SQLite freed content; this is not a promise of physical-media erasure.
      this.db.exec(`
      PRAGMA journal_mode = DELETE;
      PRAGMA synchronous = FULL;
      PRAGMA secure_delete = ON;
      PRAGMA busy_timeout = 5000;
      CREATE TABLE IF NOT EXISTS ocr_jobs (
        job_id TEXT PRIMARY KEY, namespace TEXT NOT NULL,
        envelope TEXT NOT NULL, metadata TEXT NOT NULL, original BLOB,
        state TEXT NOT NULL, retry_of TEXT, submitted_at INTEGER NOT NULL,
        lease_token TEXT, lease_until INTEGER, execution_deadline INTEGER,
        terminal_at INTEGER, poll_after INTEGER NOT NULL DEFAULT 0
      ) STRICT;
      CREATE INDEX IF NOT EXISTS ocr_queue ON ocr_jobs(state, submitted_at);
      CREATE UNIQUE INDEX IF NOT EXISTS ocr_one_retry ON ocr_jobs(namespace, retry_of) WHERE retry_of IS NOT NULL;
      CREATE TABLE IF NOT EXISTS ocr_receipts (
        namespace TEXT NOT NULL, request_key TEXT NOT NULL,
        action TEXT NOT NULL, digest TEXT NOT NULL, job_id TEXT NOT NULL,
        accepted_at INTEGER NOT NULL, outcome TEXT,
        PRIMARY KEY(namespace, request_key)
      ) STRICT;
    `);
      this.maintain();
    } catch (error) {
      try {
        openedDatabase?.close();
      } catch {
        /* Database may not have opened. */
      }
      this.releaseLock();
      throw error;
    }
  }

  private acquireLock() {
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        const fd = openSync(this.lockPath, "wx", 0o600);
        try {
          writeFileSync(
            fd,
            JSON.stringify({ pid: process.pid, token: this.lockToken }),
          );
        } finally {
          closeSync(fd);
        }
        return;
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
        // Serialize stale-owner inspection/removal. Never unlink based on a PID
        // read before this gate: a competing startup may have acquired ownership.
        const recoveryPath = `${this.lockPath}.recovery`;
        let recoveryFd: number;
        try {
          recoveryFd = openSync(recoveryPath, "wx", 0o600);
        } catch {
          throw new Error("OCR owner recovery requires operator inspection.");
        }
        try {
          if (lstatSync(this.lockPath).isSymbolicLink())
            throw new Error("Invalid OCR owner lock.");
          let pid: number;
          try {
            pid = JSON.parse(readFileSync(this.lockPath, "utf8")).pid;
          } catch {
            throw new Error("OCR owner lock requires operator inspection.");
          }
          if (!Number.isSafeInteger(pid) || pid <= 0)
            throw new Error("Invalid OCR owner lock.");
          try {
            process.kill(pid, 0);
          } catch (ownerError) {
            if ((ownerError as NodeJS.ErrnoException).code === "ESRCH") {
              unlinkSync(this.lockPath);
              continue;
            }
            throw new Error("OCR service ownership cannot be established.");
          }
          throw new Error(
            "The OCR database already has an active runtime owner.",
          );
        } finally {
          closeSync(recoveryFd);
          unlinkSync(recoveryPath);
        }
      }
    }
    throw new Error("OCR service ownership cannot be established.");
  }

  private releaseLock() {
    try {
      if (
        JSON.parse(readFileSync(this.lockPath, "utf8")).token === this.lockToken
      )
        unlinkSync(this.lockPath);
    } catch {
      /* A missing lock must not cause a shutdown write elsewhere. */
    }
  }

  private transaction<T>(work: () => T): T {
    this.db.exec("BEGIN IMMEDIATE");
    try {
      const value = work();
      this.db.exec("COMMIT");
      return value;
    } catch (error) {
      this.db.exec("ROLLBACK");
      throw error;
    }
  }
  private row(namespace: string, jobId: string) {
    return this.db
      .prepare("SELECT * FROM ocr_jobs WHERE namespace=? AND job_id=?")
      .get(namespace, jobId) as unknown as JobRow | undefined;
  }
  private receipt(namespace: string, key: string) {
    return this.db
      .prepare(
        "SELECT action,digest,job_id FROM ocr_receipts WHERE namespace=? AND request_key=?",
      )
      .get(namespace, key) as unknown as ReceiptRow | undefined;
  }
  private envelope(row: JobRow): OcrJob {
    return ocrJobSchema.parse(JSON.parse(row.envelope));
  }
  private save(row: JobRow, job: OcrJob, terminalAt: number | null = null) {
    const validated = ocrJobSchema.parse(job);
    this.db
      .prepare(
        "UPDATE ocr_jobs SET envelope=?,state=?,terminal_at=? WHERE job_id=?",
      )
      .run(JSON.stringify(validated), validated.state, terminalAt, row.job_id);
    if (terminalAt !== null) {
      this.db
        .prepare(
          "UPDATE ocr_jobs SET lease_token=NULL,lease_until=NULL WHERE job_id=?",
        )
        .run(row.job_id);
      this.db
        .prepare("UPDATE ocr_receipts SET outcome=? WHERE job_id=?")
        .run(validated.state, row.job_id);
      // Delete service input immediately on terminal transitions, stricter than 10 min.
      this.db
        .prepare("UPDATE ocr_jobs SET original=NULL WHERE job_id=?")
        .run(row.job_id);
      this.activeInputs.get(row.job_id)?.fill(0);
    }
    return validated;
  }
  private expireRows(now: number) {
    this.db
      .prepare(
        "UPDATE ocr_jobs SET original=NULL WHERE terminal_at IS NOT NULL AND original IS NOT NULL",
      )
      .run();
    this.db
      .prepare(
        "DELETE FROM ocr_jobs WHERE terminal_at IS NOT NULL AND terminal_at<=?",
      )
      .run(now - DAY);
    // Small key/digest/job/outcome tombstones are deliberately retained indefinitely.
    // No expired accepted key is ever recycled into a new execution.
  }

  accept(
    namespace: string,
    key: string,
    input: unknown,
    original: Buffer,
    partMime: string,
  ): { job: OcrJob; replay: boolean } {
    return this.acceptCommand(namespace, key, input, original, partMime, false);
  }

  private acceptCommand(
    namespace: string,
    key: string,
    input: unknown,
    original: Buffer,
    partMime: string,
    executionLease: boolean,
  ): { job: OcrJob; replay: boolean } {
    key = requireRequestKey(key);
    const parsed = ocrRequestSchema.safeParse(input);
    if (!parsed.success) {
      const version =
        input && typeof input === "object" && "contractVersion" in input
          ? input.contractVersion
          : undefined;
      throw new OcrRuntimeError(
        400,
        version !== undefined && version !== 1
          ? "unsupported_contract"
          : "invalid_request",
      );
    }
    const request = parsed.data;
    if (original.length > OCR_MAX_INPUT_BYTES)
      throw new OcrRuntimeError(413, "resource_limit");
    if (partMime !== request.source.mime)
      throw new OcrRuntimeError(415, "unsupported_media");
    const fileHash = digest(original);
    const commandDigest = digest(
      canonicalJson({ action: "submit", metadata: request, fileHash }),
    );
    return this.transaction(() => {
      this.expireRows(this.now());
      const receipt = this.receipt(namespace, key);
      if (receipt) {
        if (receipt.action !== "submit" || receipt.digest !== commandDigest)
          throw new OcrRuntimeError(409, "idempotency_conflict");
        const existing = this.row(namespace, receipt.job_id);
        if (!existing) throw new OcrRuntimeError(410, "expired");
        const job = this.envelope(existing);
        if (executionLease && !terminal(job.state))
          this.requireExecutionAdmission(job.jobId);
        return { job, replay: true };
      }
      if (original.length !== request.source.sizeBytes)
        throw new OcrRuntimeError(400, "size_mismatch");
      if (fileHash !== request.source.originalSha256)
        throw new OcrRuntimeError(400, "hash_mismatch");
      if (signatureMime(original) !== request.source.mime)
        throw new OcrRuntimeError(400, "invalid_document");
      const provider = this.providers.get(request.pipelineId);
      if (!provider) throw new OcrRuntimeError(422, "pipeline_unavailable");
      let attempt = 1;
      if (request.retryOf !== null) {
        const priorRow = this.row(namespace, request.retryOf);
        const prior = priorRow && this.envelope(priorRow);
        const priorRequest =
          priorRow && (JSON.parse(priorRow.metadata) as OcrRequest);
        const child = this.db
          .prepare("SELECT 1 FROM ocr_jobs WHERE namespace=? AND retry_of=?")
          .get(namespace, request.retryOf);
        if (
          !prior ||
          !priorRequest ||
          prior.state !== "failed" ||
          !prior.error?.retryable ||
          prior.attempt >= 3 ||
          child ||
          canonicalJson({ ...priorRequest, retryOf: null }) !==
            canonicalJson({ ...request, retryOf: null }) ||
          canonicalJson(prior.pipeline) !== canonicalJson(provider.pipeline)
        )
          throw new OcrRuntimeError(409, "invalid_retry");
        attempt = prior.attempt + 1;
      }
      const pending = this.db
        .prepare(
          "SELECT COUNT(*) AS count FROM ocr_jobs WHERE state IN ('queued','running','cancelling')",
        )
        .get() as { count: number };
      if (executionLease) this.requireExecutionAdmission();
      else if (this.executionOwner)
        throw new OcrRuntimeError(429, "rate_limited", true);
      if (pending.count >= this.capacity)
        throw new OcrRuntimeError(429, "rate_limited", true);
      const now = this.now();
      const job = ocrJobSchema.parse({
        contractVersion: 1,
        jobId: randomUUID(),
        requestKey: key,
        retryOf: request.retryOf,
        attempt,
        version: 1,
        source: request.source,
        profile: request.profile,
        pipeline: provider.pipeline,
        submittedAt: new Date(now).toISOString(),
        updatedAt: new Date(now).toISOString(),
        state: "queued",
        result: null,
        error: null,
      });
      this.db
        .prepare(
          "INSERT INTO ocr_jobs(job_id,namespace,envelope,metadata,original,state,retry_of,submitted_at) VALUES(?,?,?,?,?,'queued',?,?)",
        )
        .run(
          job.jobId,
          namespace,
          JSON.stringify(job),
          canonicalJson(request),
          original,
          request.retryOf,
          now,
        );
      this.db
        .prepare(
          "INSERT INTO ocr_receipts(namespace,request_key,action,digest,job_id,accepted_at) VALUES(?,?,'submit',?,?,?)",
        )
        .run(namespace, key, commandDigest, job.jobId, now);
      return { job, replay: false };
    });
  }

  private requireExecutionAdmission(existingJobId?: string): void {
    if (this.closed || this.executionOwner || this.active.size)
      throw new OcrRuntimeError(429, "rate_limited", true);
    const pending = this.db
      .prepare(
        "SELECT job_id,state FROM ocr_jobs WHERE state IN ('queued','running','cancelling')",
      )
      .all() as Array<{ job_id: string; state: string }>;
    if (
      existingJobId
        ? pending.length !== 1 ||
          pending[0].job_id !== existingJobId ||
          pending[0].state !== "queued"
        : pending.length !== 0
    )
      throw new OcrRuntimeError(429, "rate_limited", true);
  }

  /** Synchronous admission/owner attachment leaves no POST202-to-execution gap. */
  acceptExecution(
    namespace: string,
    key: string,
    input: unknown,
    original: Buffer,
    partMime: string,
  ): OcrExecutionLease {
    const accepted = this.acceptCommand(
      namespace,
      key,
      input,
      original,
      partMime,
      true,
    );
    return this.createExecutionLease(namespace, accepted);
  }

  /** Recover an existing receipt without ever accepting another command. */
  leaseExistingExecution(namespace: string, key: string): OcrExecutionLease {
    key = requireRequestKey(key);
    const accepted = this.transaction(() => {
      this.expireRows(this.now());
      const receipt = this.receipt(namespace, key);
      if (!receipt || receipt.action !== "submit")
        throw new OcrRuntimeError(404, "not_found");
      const row = this.row(namespace, receipt.job_id);
      if (!row) throw new OcrRuntimeError(410, "expired");
      const job = this.envelope(row);
      if (!terminal(job.state)) this.requireExecutionAdmission(job.jobId);
      return { job, replay: true };
    });
    return this.createExecutionLease(namespace, accepted);
  }

  private createExecutionLease(
    namespace: string,
    accepted: { job: OcrJob; replay: boolean },
  ): OcrExecutionLease {
    if (terminal(accepted.job.state))
      return {
        ...accepted,
        execute: async () => accepted.job,
        interruptAndDrain: async () => {},
      };
    const owner: ExecutionOwner = {
      namespace,
      jobId: accepted.job.jobId,
      token: randomUUID(),
    };
    this.executionOwner = owner;
    const release = () => {
      if (this.executionOwner === owner && !this.active.has(owner.jobId))
        this.executionOwner = undefined;
    };
    const drain = async () => {
      const execution = this.executions.get(owner.jobId);
      if (execution) await execution.catch(() => {});
    };
    return {
      ...accepted,
      execute: () => {
        owner.execution ??= (async () => {
          try {
            if (this.executionOwner !== owner || this.closed)
              throw new OcrRuntimeError(503, "service_unavailable", true);
            await this.processNextOwned(owner.token);
            await drain();
            if (this.closed)
              throw new OcrRuntimeError(503, "service_unavailable", true);
            let job = this.get(namespace, owner.jobId);
            if (!terminal(job.state)) {
              this.interruptExecution(owner);
              job = this.get(namespace, owner.jobId);
            }
            return job;
          } finally {
            await drain();
            release();
          }
        })();
        return owner.execution;
      },
      interruptAndDrain: async () => {
        this.interruptExecution(owner);
        if (owner.execution) await owner.execution.catch(() => {});
        await drain();
        release();
      },
    };
  }

  private interruptExecution(owner: ExecutionOwner): void {
    if (this.closed || this.executionOwner !== owner) return;
    this.transaction(() => {
      const row = this.row(owner.namespace, owner.jobId);
      if (!row) return;
      const job = this.envelope(row);
      if (job.state === "queued" || job.state === "running")
        this.fail(row, "worker_interrupted", true);
      // An independently authorized cancellation keeps its own outcome.
    });
    this.active.get(owner.jobId)?.abort();
  }

  get(namespace: string, jobId: string, poll = false): OcrJob {
    return this.transaction(() => {
      this.expireRows(this.now());
      const row = this.row(namespace, jobId);
      if (!row) {
        const expired = this.db
          .prepare("SELECT 1 FROM ocr_receipts WHERE namespace=? AND job_id=?")
          .get(namespace, jobId);
        throw new OcrRuntimeError(
          expired ? 410 : 404,
          expired ? "expired" : "not_found",
        );
      }
      if (poll && row.poll_after > this.now())
        throw new OcrRuntimeError(429, "rate_limited", true);
      if (poll)
        this.db
          .prepare("UPDATE ocr_jobs SET poll_after=? WHERE job_id=?")
          .run(this.now() + 1000, jobId);
      return this.envelope(row);
    });
  }

  lookup(namespace: string, key: string, poll = false): OcrJob {
    const receipt = this.receipt(namespace, requireRequestKey(key));
    if (!receipt || receipt.action !== "submit")
      throw new OcrRuntimeError(404, "not_found");
    return this.get(namespace, receipt.job_id, poll);
  }

  cancel(
    namespace: string,
    jobId: string,
    key: string,
    version: number,
  ): OcrJob {
    key = requireRequestKey(key);
    if (!Number.isSafeInteger(version) || version < 1)
      throw new OcrRuntimeError(400, "invalid_request");
    const commandDigest = digest(
      canonicalJson({ action: "cancel", jobId, version }),
    );
    const result = this.transaction(() => {
      this.expireRows(this.now());
      const receipt = this.receipt(namespace, key);
      if (
        receipt &&
        (receipt.action !== "cancel" || receipt.digest !== commandDigest)
      )
        throw new OcrRuntimeError(409, "idempotency_conflict");
      const row = this.row(namespace, jobId);
      if (!row)
        throw new OcrRuntimeError(
          receipt ? 410 : 404,
          receipt ? "expired" : "not_found",
        );
      const job = this.envelope(row);
      if (receipt) return job;
      if (terminal(job.state))
        throw new OcrRuntimeError(409, "terminal_conflict");
      if (job.version !== version)
        throw new OcrRuntimeError(409, "version_conflict");
      const now = this.now();
      this.db
        .prepare(
          "INSERT INTO ocr_receipts(namespace,request_key,action,digest,job_id,accepted_at) VALUES(?,?,'cancel',?,?,?)",
        )
        .run(namespace, key, commandDigest, jobId, now);
      if (job.state === "cancelling") return job;
      const state = job.state === "queued" ? "cancelled" : "cancelling";
      return this.save(
        row,
        {
          ...job,
          state,
          version: job.version + 1,
          updatedAt: new Date(now).toISOString(),
        },
        state === "cancelled" ? now : null,
      );
    });
    if (result.state === "cancelling") this.active.get(jobId)?.abort();
    return result;
  }

  private fail(
    row: JobRow,
    code: OcrExecutionCode,
    retryable: boolean,
  ): OcrJob {
    const job = this.envelope(row);
    const message = {
      timeout: "Document processing exceeded the configured time limit.",
      worker_interrupted: "The document worker was interrupted.",
      provider_unavailable: "The configured document provider is unavailable.",
      provider_failure: "Document processing failed.",
      invalid_provider_output: "The document provider returned invalid output.",
      invalid_document: "The original document is invalid.",
      resource_limit: "The OCR resource limit was exceeded.",
      encrypted_document: "Encrypted documents are unsupported.",
      unsupported_media: "Unsupported original media type.",
    }[code];
    return this.save(
      row,
      {
        ...job,
        state: "failed",
        result: null,
        error: { code, message, retryable },
        version: job.version + 1,
        updatedAt: new Date(this.now()).toISOString(),
      },
      this.now(),
    );
  }

  /** Expired claims become explicit retryable failures, never hidden model reruns. */
  maintain(): void {
    this.transaction(() => {
      const now = this.now();
      const rows = this.db
        .prepare(
          "SELECT * FROM ocr_jobs WHERE state IN ('running','cancelling') AND (lease_until<=? OR execution_deadline<=?)",
        )
        .all(now, now) as unknown as JobRow[];
      for (const row of rows) {
        const job = this.envelope(row);
        if (job.state === "cancelling")
          this.save(
            row,
            {
              ...job,
              state: "cancelled",
              version: job.version + 1,
              updatedAt: new Date(now).toISOString(),
            },
            now,
          );
        else
          this.fail(
            row,
            row.execution_deadline! <= now ? "timeout" : "worker_interrupted",
            true,
          );
        this.active.get(row.job_id)?.abort();
      }
      this.expireRows(now);
    });
  }

  private claim(executionToken?: string): OcrClaim | null {
    if (this.executionOwner?.token !== executionToken) return null;
    this.maintain();
    if (this.active.size >= this.concurrency) return null;
    return this.transaction(() => {
      const running = this.db
        .prepare(
          "SELECT COUNT(*) AS count FROM ocr_jobs WHERE state IN ('running','cancelling')",
        )
        .get() as { count: number };
      if (running.count >= this.concurrency) return null;
      const row = this.db
        .prepare(
          "SELECT * FROM ocr_jobs WHERE state='queued' ORDER BY submitted_at,job_id LIMIT 1",
        )
        .get() as unknown as JobRow | undefined;
      if (!row) return null;
      const prior = this.envelope(row);
      const provider = this.providers.get(prior.pipeline.id);
      if (
        !provider ||
        canonicalJson(provider.pipeline) !== canonicalJson(prior.pipeline)
      ) {
        this.fail(row, "provider_unavailable", true);
        return null;
      }
      if (!row.original) {
        this.fail(row, "worker_interrupted", true);
        return null;
      }
      const now = this.now(),
        token = randomUUID(),
        deadline = now + this.timeout;
      const job = this.save(row, {
        ...prior,
        state: "running",
        version: prior.version + 1,
        updatedAt: new Date(now).toISOString(),
      });
      this.db
        .prepare(
          "UPDATE ocr_jobs SET lease_token=?,lease_until=?,execution_deadline=? WHERE job_id=?",
        )
        .run(token, Math.min(now + this.lease, deadline), deadline, row.job_id);
      return {
        namespace: row.namespace,
        job,
        request: JSON.parse(row.metadata),
        original: Buffer.from(row.original),
        token,
        deadline,
      };
    });
  }

  /** A fenced stale worker cannot extend its lease or overwrite a terminal result. */
  private heartbeat(claim: OcrClaim): boolean {
    const now = this.now();
    const result = this.db
      .prepare(
        "UPDATE ocr_jobs SET lease_until=? WHERE job_id=? AND namespace=? AND lease_token=? AND state='running' AND lease_until>? AND execution_deadline>?",
      )
      .run(
        Math.min(now + this.lease, claim.deadline),
        claim.job.jobId,
        claim.namespace,
        claim.token,
        now,
        now,
      );
    return result.changes === 1;
  }

  private finish(claim: OcrClaim, outcome: unknown): boolean {
    return this.transaction(() => {
      const row = this.row(claim.namespace, claim.job.jobId);
      if (!row || row.lease_token !== claim.token) return false;
      const current = this.envelope(row);
      if (terminal(current.state)) return false;
      const now = this.now();
      if (current.state === "cancelling") {
        this.save(
          row,
          {
            ...current,
            state: "cancelled",
            version: current.version + 1,
            updatedAt: new Date(now).toISOString(),
          },
          now,
        );
        return true;
      }
      if (row.execution_deadline! <= now || row.lease_until! <= now) {
        this.fail(
          row,
          row.execution_deadline! <= now ? "timeout" : "worker_interrupted",
          true,
        );
        return false;
      }
      let parsed: OcrJob | undefined;
      try {
        const output = outcome as { state?: unknown; result?: unknown };
        if (
          !output ||
          typeof output !== "object" ||
          !["completed", "partial"].includes(String(output.state)) ||
          Object.keys(output).some((key) => !["state", "result"].includes(key))
        )
          throw new Error();
        const candidate = {
          ...current,
          state: output.state,
          result: output.result,
          error: null,
          version: current.version + 1,
          updatedAt: new Date(now).toISOString(),
        };
        if (Buffer.byteLength(JSON.stringify(candidate)) > OCR_MAX_OUTPUT_BYTES)
          throw new Error();
        parsed = ocrJobSchema.parse(candidate);
      } catch {
        this.fail(row, "invalid_provider_output", false);
        return true;
      }
      this.save(row, parsed, now);
      return true;
    });
  }

  async processNext(): Promise<boolean> {
    return this.processNextOwned();
  }

  private async processNextOwned(executionToken?: string): Promise<boolean> {
    if (this.closed) return false;
    const claim = this.claim(executionToken);
    if (!claim) return false;
    const controller = new AbortController();
    this.active.set(claim.job.jobId, controller);
    this.activeInputs.set(claim.job.jobId, claim.original);
    let timer: ReturnType<typeof setTimeout> | undefined;
    const heartbeat = setInterval(
      () => {
        if (!this.closed && !this.heartbeat(claim)) controller.abort();
      },
      Math.max(1, Math.min(1000, Math.floor(this.lease / 3))),
    );
    const provider = this.providers.get(claim.job.pipeline.id)!;
    const execution = Promise.resolve().then(() =>
      provider.analyze(
        { request: claim.request, original: claim.original },
        controller.signal,
      ),
    );
    this.executions.set(claim.job.jobId, execution);
    // Keep the slot occupied until provider cleanup and inference draining finish.
    void execution
      .finally(() => {
        claim.original.fill(0);
        this.active.delete(claim.job.jobId);
        this.activeInputs.delete(claim.job.jobId);
        this.executions.delete(claim.job.jobId);
        if (this.closed && this.active.size === 0) this.releaseLock();
      })
      .catch(() => {});
    try {
      const result = await Promise.race([
        execution,
        new Promise<never>((_, reject) => {
          timer = setTimeout(() => {
            controller.abort();
            reject(new Error("timeout"));
          }, this.timeout);
        }),
        new Promise<never>((_, reject) =>
          controller.signal.addEventListener(
            "abort",
            () => reject(new Error("aborted")),
            { once: true },
          ),
        ),
      ]);
      if (!this.closed) this.finish(claim, result);
    } catch (error) {
      if (!this.closed)
        this.transaction(() => {
          const row = this.row(claim.namespace, claim.job.jobId);
          if (!row || row.lease_token !== claim.token) return;
          const current = this.envelope(row);
          if (current.state === "cancelling")
            this.save(
              row,
              {
                ...current,
                state: "cancelled",
                version: current.version + 1,
                updatedAt: new Date(this.now()).toISOString(),
              },
              this.now(),
            );
          else if (!terminal(current.state))
            this.fail(
              row,
              controller.signal.aborted
                ? "timeout"
                : error instanceof OcrExecutionError
                  ? error.code
                  : "provider_failure",
              error instanceof OcrExecutionError ? error.retryable : true,
            );
        });
    } finally {
      if (timer) clearTimeout(timer);
      clearInterval(heartbeat);
    }
    return true;
  }

  close(): void {
    if (this.closed) return;
    if (this.executionOwner) this.interruptExecution(this.executionOwner);
    this.closed = true;
    for (const controller of this.active.values()) controller.abort();
    for (const original of this.activeInputs.values()) original.fill(0);
    this.db.close();
    if (this.active.size === 0) this.releaseLock();
  }
}

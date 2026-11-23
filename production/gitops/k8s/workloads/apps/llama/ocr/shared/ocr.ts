import { z } from "zod";

/** Contract v1 limits; decoding and additional raster passes must also enforce these in the worker. */
export const ocrLimits = {
  metadataBytes: 65_536,
  originalBytes: 10_485_760,
  envelopeBytes: 10_485_760,
  pages: 50,
  pageSide: 16_000,
  pagePixels: 40_000_000,
  totalPixels: 100_000_000,
  textCodePoints: 1_000_000,
  pageBlocks: 10_000,
  totalBlocks: 50_000,
  executionSeconds: 300,
} as const;

function codePointLength(value: string): number {
  let length = 0;
  for (const _ of value) length++;
  return length;
}

// JSON Schema measures string lengths in Unicode code points, not UTF-16 units.
const string = (min: number, max: number) =>
  z.string().refine((value) => {
    if (value.length < min || value.length > max * 2) return false;
    const length = codePointLength(value);
    return length >= min && length <= max;
  }, "String length exceeds the OCR contract bounds.");
const integer = (min: number, max: number) =>
  z.number().int().min(min).max(max);
const uuid = z
  .string()
  .regex(
    /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/,
  );
const hash = z.string().regex(/^[a-f0-9]{64}$/);
const reason = string(1, 500).nullable();
const version = integer(1, 2_147_483_647);
const unique = <T>(values: T[]) => new Set(values).size === values.length;
const issue = (
  ctx: z.RefinementCtx,
  message: string,
  path: (string | number)[] = [],
) => ctx.addIssue({ code: "custom", message, path });

function validDate(value: string): boolean {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
  const [year, month, day] = value.split("-").map(Number);
  const leap = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
  const days = [31, leap ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  return (
    year >= 1 && month >= 1 && month <= 12 && day >= 1 && day <= days[month - 1]
  );
}

const date = z.string().refine(validDate, "Invalid OCR calendar date.");
const timestamp = z.string().refine((value) => {
  const match =
    /^(\d{4}-\d{2}-\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,6}))?Z$/.exec(
      value,
    );
  return (
    !!match &&
    validDate(match[1]) &&
    +match[2] <= 23 &&
    +match[3] <= 59 &&
    +match[4] <= 59
  );
}, "Invalid OCR UTC timestamp.");
const comparableTimestamp = (value: string) =>
  value.slice(0, 19) +
  "." +
  (value.split(".")[1]?.slice(0, -1) ?? "").padEnd(6, "0");

function jsonByteLength(value: unknown): number {
  let bytes = 0;
  for (const char of JSON.stringify(value)) {
    const point = char.codePointAt(0)!;
    bytes += point <= 0x7f ? 1 : point <= 0x7ff ? 2 : point <= 0xffff ? 3 : 4;
  }
  return bytes;
}

const sourceSchema = z
  .object({
    documentId: string(1, 100),
    originalSha256: hash,
    mime: z.enum(["image/jpeg", "image/png", "image/webp", "application/pdf"]),
    sizeBytes: integer(1, ocrLimits.originalBytes),
    originalName: string(1, 200).optional(),
  })
  .strict();

export const ocrRequestSchema = z
  .object({
    contractVersion: z.literal(1),
    source: sourceSchema,
    profile: z.literal("receipt-core-v1"),
    pipelineId: string(1, 100),
    localeHints: z
      .array(string(2, 35))
      .min(1)
      .max(5)
      .refine(unique, "Locale hints must be unique."),
    retryOf: uuid.nullable(),
  })
  .strict()
  .refine(
    (value) => jsonByteLength(value) <= ocrLimits.metadataBytes,
    "OCR metadata exceeds its byte limit.",
  );
export type OcrRequest = z.infer<typeof ocrRequestSchema>;

const modelSchema = z
  .object({ id: string(1, 200), revision: string(1, 200) })
  .strict();
export const ocrPipelineSchema = z
  .object({
    id: string(1, 100),
    fingerprint: hash,
    executionMode: z.enum(["mock", "real"]),
    ocrModel: modelSchema,
    interpretationModel: modelSchema.nullable(),
    layoutModel: modelSchema,
    inferenceEngine: z
      .object({
        name: z.enum(["vllm", "contract-fixture"]),
        version: string(1, 100),
      })
      .strict(),
  })
  .strict()
  .superRefine((pipeline, ctx) => {
    if (pipeline.executionMode === "real") {
      if (pipeline.ocrModel.id !== "PaddlePaddle/PaddleOCR-VL-1.6")
        issue(ctx, "Real OCR requires the pinned PaddleOCR-VL-1.6 identity.", [
          "ocrModel",
          "id",
        ]);
      if (pipeline.inferenceEngine.name !== "vllm")
        issue(ctx, "Real OCR requires the vLLM inference engine.", [
          "inferenceEngine",
          "name",
        ]);
    }
  });
export type OcrPipeline = z.infer<typeof ocrPipelineSchema>;

const boxSchema = z
  .object({
    x: z.number().min(0).max(1),
    y: z.number().min(0).max(1),
    width: z.number().positive().max(1),
    height: z.number().positive().max(1),
  })
  .strict()
  .refine(
    (box) => box.x + box.width <= 1 && box.y + box.height <= 1,
    "OCR region must remain inside its original page.",
  );

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
    message: string(1, 500),
    page: integer(1, ocrLimits.pages).nullable(),
  })
  .strict();
const blockSchema = z
  .object({ id: string(1, 100), text: string(1, 10_000), box: boxSchema })
  .strict();
const pageSchema = z
  .object({
    page: integer(1, ocrLimits.pages),
    width: integer(1, ocrLimits.pageSide),
    height: integer(1, ocrLimits.pageSide),
    status: z.enum(["read", "unreadable"]),
    reason,
    text: string(0, ocrLimits.textCodePoints),
    blocks: z.array(blockSchema).max(ocrLimits.pageBlocks),
    warnings: z.array(warningSchema).max(100),
  })
  .strict()
  .superRefine((page, ctx) => {
    if (page.width * page.height > ocrLimits.pagePixels)
      issue(ctx, "OCR page raster exceeds its pixel limit.");
    if (
      page.status === "unreadable" &&
      (page.reason === null || page.text !== "" || page.blocks.length !== 0)
    )
      issue(
        ctx,
        "Unreadable OCR pages require a reason and empty text and blocks.",
      );
    if (!unique(page.blocks.map((block) => block.id)))
      issue(ctx, "OCR block identifiers must be unique on each page.", [
        "blocks",
      ]);
    let offset = 0;
    for (let i = 0; i < page.blocks.length; i++) {
      const block = page.blocks[i];
      const index = page.text.indexOf(block.text, offset);
      if (index < 0) {
        issue(ctx, "OCR blocks must occur in page text in reading order.", [
          "blocks",
          i,
        ]);
        break; // Once invalid, do not rescan the same large text for every remaining block.
      }
      offset = index + block.text.length;
    }
  });
export const ocrRecognizedPageSchema = pageSchema;
export const ocrRecognizedPagesSchema = z
  .unknown()
  .superRefine((input, ctx) => {
    if (!Array.isArray(input)) return;
    if (input.length > ocrLimits.pages) {
      issue(ctx, "OCR array exceeds its contract limit.");
      return;
    }
    let blockCount = 0;
    for (const [index, value] of input.entries()) {
      const page = record(value);
      if (!page) continue;
      if (Array.isArray(page.blocks)) {
        blockCount += page.blocks.length;
        if (
          page.blocks.length > ocrLimits.pageBlocks ||
          blockCount > ocrLimits.totalBlocks
        ) {
          issue(ctx, "OCR recognized pages exceed their block limit.", [
            index,
            "blocks",
          ]);
          return;
        }
      }
      if (Array.isArray(page.warnings) && page.warnings.length > 100) {
        issue(ctx, "OCR array exceeds its contract limit.", [
          index,
          "warnings",
        ]);
        return;
      }
    }
  })
  .pipe(
    z
      .array(pageSchema)
      .min(1)
      .max(ocrLimits.pages)
      .superRefine((pages, ctx) => {
        if (pages.some((page, index) => page.page !== index + 1))
          issue(ctx, "OCR pages must be ordered from one without gaps.");
        if (
          pages.reduce((sum, page) => sum + page.width * page.height, 0) >
          ocrLimits.totalPixels
        )
          issue(ctx, "OCR document raster exceeds its total pixel limit.");
        if (
          pages.reduce((sum, page) => sum + codePointLength(page.text), 0) >
          ocrLimits.textCodePoints
        )
          issue(ctx, "OCR document text exceeds its total code point limit.");
        for (const [index, page] of pages.entries()) {
          if (
            page.warnings.some(
              (warning) => warning.page !== null && warning.page !== page.page,
            )
          )
            issue(ctx, "Page warnings must reference their own page.", [
              index,
              "warnings",
            ]);
        }
      }),
  );
const evidenceSchema = z
  .object({
    page: integer(1, ocrLimits.pages),
    blockIds: z
      .array(string(1, 100))
      .min(1)
      .max(100)
      .refine(unique, "Evidence block identifiers must be unique."),
    box: boxSchema,
    quote: string(1, 10_000),
  })
  .strict();
const candidateSchema = <T extends z.ZodType>(value: T) =>
  z
    .object({
      id: string(1, 100),
      rawText: string(1, 10_000),
      value,
      confidence: z.number().min(0).max(1).nullable(),
      evidence: z.array(evidenceSchema).min(1).max(20),
    })
    .strict()
    .refine(
      (candidate) =>
        candidate.rawText ===
        candidate.evidence.map((evidence) => evidence.quote).join("\n"),
      "Candidate raw text must preserve its cited quotations joined by newlines.",
    );
const fieldSchema = <T extends z.ZodType>(value: T) =>
  z
    .object({
      status: z.enum([
        "read",
        "ambiguous",
        "missing",
        "unreadable",
        "unsupported",
      ]),
      reason,
      candidates: z.array(candidateSchema(value)).max(5),
    })
    .strict()
    .superRefine((field, ctx) => {
      const count = field.candidates.length;
      if (
        field.status === "read"
          ? count !== 1
          : field.status === "ambiguous"
            ? count < 2
            : count !== 0
      )
        issue(ctx, "Candidate count does not match the OCR field status.", [
          "candidates",
        ]);
      if (field.status !== "read" && field.reason === null)
        issue(ctx, "Unresolved OCR fields require a reason.", ["reason"]);
    });

const resultSchema = z
  .object({
    mode: z.enum(["mock", "real"]),
    methodsUsed: z
      .array(z.enum(["ocr", "pdf_text", "llm"]))
      .max(3)
      .refine(unique, "OCR methods must be unique."),
    documentType: z.enum([
      "receipt",
      "invoice",
      "fiscal_representation",
      "mixed",
      "unknown",
    ]),
    pageCount: integer(1, ocrLimits.pages),
    pages: z.array(pageSchema).min(1).max(ocrLimits.pages),
    fields: z
      .object({
        merchant: fieldSchema(string(1, 300)),
        date: fieldSchema(date),
        total: fieldSchema(integer(0, 100_000_000)),
        currency: fieldSchema(z.string().regex(/^[A-Z]{3}$/)),
      })
      .strict(),
    warnings: z.array(warningSchema).max(100),
    verification: z
      .object({
        fiscalAuthenticity: z.literal(false),
        bankVerification: z.literal(false),
      })
      .strict(),
  })
  .strict()
  .superRefine((result, ctx) => {
    if (result.pageCount !== result.pages.length)
      issue(ctx, "OCR page count must match all returned pages.", [
        "pageCount",
      ]);
    if (result.pages.some((page, index) => page.page !== index + 1))
      issue(ctx, "OCR pages must be ordered from one without gaps.", ["pages"]);
    if (
      result.pages.reduce((sum, page) => sum + page.width * page.height, 0) >
      ocrLimits.totalPixels
    )
      issue(ctx, "OCR document raster exceeds its total pixel limit.", [
        "pages",
      ]);
    if (
      result.pages.reduce((sum, page) => sum + codePointLength(page.text), 0) >
      ocrLimits.textCodePoints
    )
      issue(ctx, "OCR document text exceeds its total code point limit.", [
        "pages",
      ]);
    if (
      result.pages.reduce((sum, page) => sum + page.blocks.length, 0) >
      ocrLimits.totalBlocks
    )
      issue(ctx, "OCR document exceeds its total block limit.", ["pages"]);

    const pages = new Map(result.pages.map((page) => [page.page, page]));
    const warnings = [
      ...result.warnings,
      ...result.pages.flatMap((page) => page.warnings),
    ];
    if (
      warnings.some(
        (warning) => warning.page !== null && !pages.has(warning.page),
      )
    )
      issue(ctx, "OCR warnings must reference an existing page.", ["warnings"]);
    for (const [index, page] of result.pages.entries()) {
      if (
        page.warnings.some(
          (warning) => warning.page !== null && warning.page !== page.page,
        )
      )
        issue(ctx, "Page warnings must reference their own page.", [
          "pages",
          index,
          "warnings",
        ]);
    }
    const multiple = warnings.some(
      (warning) => warning.code === "multiple_documents",
    );
    if (result.documentType === "mixed" && !multiple)
      issue(ctx, "Mixed documents require a multiple-documents warning.", [
        "warnings",
      ]);
    if (
      (multiple || result.documentType === "mixed") &&
      Object.values(result.fields).some(
        (field) => field.status !== "unsupported",
      )
    )
      issue(ctx, "Multiple documents must abstain on all core fields.", [
        "fields",
      ]);

    const ids = new Set<string>();
    for (const [fieldName, field] of Object.entries(result.fields)) {
      for (const [candidateIndex, candidate] of field.candidates.entries()) {
        const path = ["fields", fieldName, "candidates", candidateIndex];
        if (ids.has(candidate.id))
          issue(ctx, "Candidate identifiers must be unique across the job.", [
            ...path,
            "id",
          ]);
        ids.add(candidate.id);
        for (const [evidenceIndex, evidence] of candidate.evidence.entries()) {
          const evidencePath = [...path, "evidence", evidenceIndex];
          const page = pages.get(evidence.page);
          if (!page || page.status !== "read") {
            issue(
              ctx,
              "Candidate evidence must reference a readable original page.",
              evidencePath,
            );
            continue;
          }
          const selected = new Set(evidence.blockIds);
          const blocks = page.blocks.filter((block) => selected.has(block.id));
          if (blocks.length !== selected.size)
            issue(ctx, "Candidate evidence references an unknown block.", [
              ...evidencePath,
              "blockIds",
            ]);
          if (
            !blocks
              .map((block) => block.text)
              .join("\n")
              .includes(evidence.quote)
          )
            issue(
              ctx,
              "Evidence quotation is absent from its blocks in reading order.",
              [...evidencePath, "quote"],
            );
          const outer = evidence.box;
          const tolerance = 0.002;
          if (
            blocks.some(
              ({ box }) =>
                outer.x > box.x + tolerance ||
                outer.y > box.y + tolerance ||
                outer.x + outer.width < box.x + box.width - tolerance ||
                outer.y + outer.height < box.y + box.height - tolerance,
            )
          )
            issue(ctx, "Evidence region must enclose every referenced block.", [
              ...evidencePath,
              "box",
            ]);
        }
      }
    }
  });

const errorSchema = z
  .object({
    code: z.enum([
      "invalid_request",
      "unsupported_contract",
      "hash_mismatch",
      "size_mismatch",
      "invalid_document",
      "unauthorized",
      "forbidden",
      "not_found",
      "expired",
      "idempotency_conflict",
      "version_conflict",
      "terminal_conflict",
      "invalid_retry",
      "resource_limit",
      "unsupported_media",
      "encrypted_document",
      "pipeline_unavailable",
      "rate_limited",
      "service_unavailable",
      "timeout",
      "provider_unavailable",
      "provider_failure",
      "invalid_provider_output",
      "worker_interrupted",
    ]),
    message: string(1, 500),
    retryable: z.boolean(),
  })
  .strict();
export const ocrHttpErrorSchema = z
  .object({
    contractVersion: z.literal(1),
    requestKey: uuid.nullable(),
    error: errorSchema,
  })
  .strict();
export type OcrHttpError = z.infer<typeof ocrHttpErrorSchema>;
export const ocrCancelCommandSchema = z.object({ version }).strict();
export type OcrCancelCommand = z.infer<typeof ocrCancelCommandSchema>;

/** Readable regions can survive a page-level gap; they do not prove full coverage. */
export function hasIncompleteOcrCoverage(
  result: Pick<z.infer<typeof resultSchema>, "pages" | "warnings">,
): boolean {
  return (
    result.warnings.some((warning) => warning.code === "page_unreadable") ||
    result.pages.some((page) =>
      page.warnings.some((warning) => warning.code === "page_unreadable"),
    )
  );
}

const jobSchema = z
  .object({
    contractVersion: z.literal(1),
    jobId: uuid,
    requestKey: uuid,
    retryOf: uuid.nullable(),
    attempt: integer(1, 3),
    version,
    source: sourceSchema,
    profile: z.literal("receipt-core-v1"),
    pipeline: ocrPipelineSchema,
    submittedAt: timestamp,
    updatedAt: timestamp,
    state: z.enum([
      "queued",
      "running",
      "cancelling",
      "completed",
      "partial",
      "failed",
      "cancelled",
    ]),
    result: resultSchema.nullable(),
    error: errorSchema.nullable(),
  })
  .strict()
  .superRefine((job, ctx) => {
    if (
      comparableTimestamp(job.updatedAt) < comparableTimestamp(job.submittedAt)
    )
      issue(ctx, "OCR update cannot precede submission.", ["updatedAt"]);
    if (
      (job.retryOf === null) !== (job.attempt === 1) ||
      job.retryOf === job.jobId
    )
      issue(ctx, "OCR retry lineage is inconsistent.", ["retryOf"]);
    if (job.state === "completed" || job.state === "partial") {
      if (job.result === null || job.error !== null)
        issue(
          ctx,
          "Successful terminal OCR jobs require a result and no error.",
        );
    } else if (job.state === "failed") {
      if (job.result !== null || job.error === null)
        issue(ctx, "Failed OCR jobs require an error and no result.");
    } else if (job.result !== null || job.error !== null) {
      issue(
        ctx,
        "Pending and cancelled OCR jobs cannot expose results or errors.",
      );
    }
    const result = job.result;
    if (result !== null) {
      if (result.mode !== job.pipeline.executionMode)
        issue(ctx, "OCR result mode must match its pipeline.", [
          "result",
          "mode",
        ]);
      if (job.source.mime !== "application/pdf" && result.pageCount !== 1)
        issue(ctx, "A supported image must have exactly one page.", [
          "result",
          "pageCount",
        ]);
      if (
        result.methodsUsed.includes("pdf_text") &&
        job.source.mime !== "application/pdf"
      )
        issue(ctx, "PDF text extraction requires a PDF original.", [
          "result",
          "methodsUsed",
        ]);
      if (result.mode === "real" && !result.methodsUsed.includes("ocr"))
        issue(ctx, "Real PaddleOCR results must record OCR execution.", [
          "result",
          "methodsUsed",
        ]);
      if (result.mode === "mock" && result.methodsUsed.length !== 0)
        issue(ctx, "Mock fixtures cannot claim executed model stages.", [
          "result",
          "methodsUsed",
        ]);
      if (
        result.methodsUsed.includes("llm") &&
        job.pipeline.interpretationModel === null
      )
        issue(ctx, "Interpretation requires a pinned interpretation model.", [
          "result",
          "methodsUsed",
        ]);
      const complete =
        result.pages.every((page) => page.status === "read") &&
        Object.values(result.fields).every(
          (field) => field.status === "read",
        ) &&
        !hasIncompleteOcrCoverage(result);
      if (job.state === "completed" && !complete)
        issue(
          ctx,
          "Completed OCR requires every page and core field to be read with no unresolved coverage warning.",
          ["state"],
        );
      if (job.state === "partial") {
        if (complete || !result.pages.some((page) => page.status === "read"))
          issue(
            ctx,
            "Partial OCR requires a readable page and an unresolved page, coverage gap or field.",
            ["state"],
          );
        if (
          result.warnings.length === 0 &&
          result.pages.every((page) => page.warnings.length === 0)
        )
          issue(ctx, "Partial OCR requires warnings explaining its gaps.", [
            "result",
            "warnings",
          ]);
      }
    }
    if (jsonByteLength(job) > ocrLimits.envelopeBytes)
      issue(ctx, "OCR envelope exceeds its serialized byte limit.");
  });

const record = (value: unknown): Record<string, unknown> | undefined =>
  value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;

// Zod array limits run after element parsing. Reject excessive container sizes first,
// so a bounded network body cannot expand into hundreds of thousands of issues.
const jobArrayPreflight = z.unknown().superRefine((input, ctx) => {
  const result = record(record(input)?.result);
  if (!result) return;
  const bounded = (
    value: unknown,
    limit: number,
    path: (string | number)[],
  ): boolean => {
    if (!Array.isArray(value) || value.length <= limit) return true;
    issue(ctx, "OCR array exceeds its contract limit.", path);
    return false;
  };
  if (
    !bounded(result.pages, ocrLimits.pages, ["result", "pages"]) ||
    !bounded(result.warnings, 100, ["result", "warnings"]) ||
    !bounded(result.methodsUsed, 3, ["result", "methodsUsed"])
  )
    return;
  let blocks = 0;
  for (const [index, inputPage] of (Array.isArray(result.pages)
    ? result.pages
    : []
  ).entries()) {
    const page = record(inputPage);
    if (!page) continue;
    if (
      !bounded(page.blocks, ocrLimits.pageBlocks, [
        "result",
        "pages",
        index,
        "blocks",
      ]) ||
      !bounded(page.warnings, 100, ["result", "pages", index, "warnings"])
    )
      return;
    blocks += Array.isArray(page.blocks) ? page.blocks.length : 0;
    if (blocks > ocrLimits.totalBlocks) {
      issue(ctx, "OCR document exceeds its total block limit.", [
        "result",
        "pages",
      ]);
      return;
    }
  }
  const fields = record(result.fields);
  for (const name of ["merchant", "date", "total", "currency"] as const) {
    const field = record(fields?.[name]);
    if (!field) continue;
    const path: (string | number)[] = ["result", "fields", name, "candidates"];
    if (!bounded(field.candidates, 5, path)) return;
    for (const [candidateIndex, inputCandidate] of (Array.isArray(
      field.candidates,
    )
      ? field.candidates
      : []
    ).entries()) {
      const candidate = record(inputCandidate);
      if (!candidate) continue;
      const evidencePath = [...path, candidateIndex, "evidence"];
      if (!bounded(candidate.evidence, 20, evidencePath)) return;
      for (const [evidenceIndex, inputEvidence] of (Array.isArray(
        candidate.evidence,
      )
        ? candidate.evidence
        : []
      ).entries()) {
        const evidence = record(inputEvidence);
        if (
          evidence &&
          !bounded(evidence.blockIds, 100, [
            ...evidencePath,
            evidenceIndex,
            "blockIds",
          ])
        )
          return;
      }
    }
  }
});

export const ocrJobSchema = jobArrayPreflight.pipe(jobSchema);
export type OcrJob = z.infer<typeof ocrJobSchema>;

function sameJson(left: unknown, right: unknown): boolean {
  if (left === right) return true;
  if (
    left === null ||
    right === null ||
    typeof left !== "object" ||
    typeof right !== "object"
  )
    return false;
  const a = left as Record<string, unknown>,
    b = right as Record<string, unknown>;
  const keys = Object.keys(a);
  return (
    keys.length === Object.keys(b).length &&
    keys.every((key) => Object.hasOwn(b, key) && sameJson(a[key], b[key]))
  );
}

/** Never include provider bodies, document text or expected identities in thrown errors. */
export function assertOcrJobIdentity(
  job: OcrJob,
  expected: {
    requestKey: string;
    request: OcrRequest;
    pipeline: OcrPipeline;
    jobId?: string;
  },
): void {
  if (
    !ocrJobSchema.safeParse(job).success ||
    !ocrRequestSchema.safeParse(expected.request).success ||
    !ocrPipelineSchema.safeParse(expected.pipeline).success ||
    !uuid.safeParse(expected.requestKey).success ||
    job.requestKey !== expected.requestKey ||
    (expected.jobId !== undefined && job.jobId !== expected.jobId) ||
    job.retryOf !== expected.request.retryOf ||
    job.profile !== expected.request.profile ||
    expected.request.pipelineId !== expected.pipeline.id ||
    !sameJson(job.source, expected.request.source) ||
    !sameJson(job.pipeline, expected.pipeline)
  )
    throw new Error("OCR response does not match the requested job.");
}

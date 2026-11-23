import { z } from "zod";
import type { ActorContext } from "./actor.js";
import type { Expense } from "./types.js";
import type { OcrJob, OcrPipeline, OcrRequest } from "./ocr.js";
export const ocrAppReason = z.string().trim().min(10).max(1500);
const key = z.string().uuid();
const version = z.number().int().positive();
const originalSha256 = z.string().regex(/^[a-f0-9]{64}$/);
export const ocrAppCreateSchema = z
  .object({
    expenseVersion: version,
    originalSha256,
    reason: ocrAppReason,
    requestKey: key,
  })
  .strict();
export const ocrAppActionSchema = z
  .object({ jobVersion: version, reason: ocrAppReason, requestKey: key })
  .strict();
export const ocrAppReviewSchema = z.discriminatedUnion("state", [
  z
    .object({
      state: z.literal("accepted"),
      reason: ocrAppReason,
      expenseVersion: version,
      resultVersion: version,
      originalSha256,
      requestKey: key,
    })
    .strict(),
  z
    .object({
      state: z.literal("corrected"),
      correctedValue: z.union([
        z.string().trim().min(1).max(300),
        z.number().int().min(0).max(100_000_000),
      ]),
      reason: ocrAppReason,
      expenseVersion: version,
      resultVersion: version,
      originalSha256,
      requestKey: key,
    })
    .strict(),
  z
    .object({
      state: z.literal("dismissed"),
      reason: ocrAppReason,
      expenseVersion: version,
      resultVersion: version,
      originalSha256,
      requestKey: key,
    })
    .strict(),
]);
export type OcrAppCreate = z.infer<typeof ocrAppCreateSchema>;
export type OcrAppAction = z.infer<typeof ocrAppActionSchema>;
export type OcrAppReviewCommand = z.infer<typeof ocrAppReviewSchema>;
export type OcrAppState = OcrJob["state"] | "outbox" | "attention";
export type OcrAppField = "merchant" | "date" | "total" | "currency";
export interface OcrAppReview {
  candidateId: string;
  field: OcrAppField;
  state: OcrAppReviewCommand["state"];
  candidateValue: string | number;
  reviewedValue: string | number | null;
  appliedValue: string | number | null;
  before: string | number | null;
  after: string | number | null;
  expenseVersion: number;
  resultVersion: number;
  originalSha256: string;
  actor: ActorContext;
  reason: string;
  at: string;
}
export interface OcrAppEvent {
  sequence: number;
  action: string;
  actor: ActorContext | null;
  reason: string;
  at: string;
  details?: unknown;
}
export interface OcrAppJob {
  id: string;
  expenseId: string;
  documentId: string;
  version: number;
  state: OcrAppState;
  source: OcrRequest["source"];
  pipeline: OcrPipeline;
  requestedBy: ActorContext;
  processingBy: ActorContext;
  requestedExpenseVersion: number;
  createdAt: string;
  updatedAt: string;
  parentId: string | null;
  attempt: number;
  remote: OcrJob | null;
  lastError: string | null;
  nextAttemptAt: string | null;
  processing: boolean;
  sourceCurrent: boolean;
  sourceIssues: string[];
  expenseVersion: number;
  reviews: OcrAppReview[];
  history: OcrAppEvent[];
}
export interface OcrAppReviewReply {
  job: OcrAppJob;
  expense: Expense;
}
export interface OcrAppAvailability {
  enabled: boolean;
  pipeline: OcrPipeline | null;
  notice: string;
}

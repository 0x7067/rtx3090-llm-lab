import { z } from "zod";
import { categories } from "./types.js";

export const evidenceSchema = z
  .object({
    source: z.enum(["expense", "document", "policy", "duplicate_case"]),
    reference: z.string().min(1).max(200),
    field: z.string().min(1).max(80),
    value: z.string().max(1500),
  })
  .strict();
export const findingSchema = z
  .object({
    id: z.string().min(1).max(100),
    severity: z.enum(["info", "warning", "high"]),
    origin: z.literal("rule"),
    title: z.string().min(1).max(160),
    explanation: z.string().min(1).max(1500),
    action: z.enum([
      "request_info",
      "compare",
      "review_policy",
      "check_fields",
    ]),
    evidence: z.array(evidenceSchema).min(1).max(20),
  })
  .strict();
export const categorySuggestionSchema = z
  .object({
    id: z.string().min(1).max(100),
    field: z.literal("category"),
    value: z.enum(categories),
    explanation: z.string().min(1).max(1500),
    mode: z.literal("mock"),
    confidence: z.null(),
    evidence: z.array(evidenceSchema).min(1).max(20),
  })
  .strict();
export const extractionSuggestionSchema = z
  .object({
    id: z.string().min(1).max(100),
    field: z.enum(["merchant", "date", "amount"]),
    value: z.string().min(1).max(160),
    documentId: z.string().min(1).max(100),
    page: z.number().int().positive().max(1000),
    box: z
      .object({
        x: z.number().min(0).max(1),
        y: z.number().min(0).max(1),
        width: z.number().positive().max(1),
        height: z.number().positive().max(1),
      })
      .strict()
      .refine((b) => b.x + b.width <= 1 && b.y + b.height <= 1),
    mode: z.literal("mock-fixture"),
    confidence: z.null(),
  })
  .strict();
export const auditReportSchema = z
  .object({
    schemaVersion: z.literal(1),
    provider: z.literal("local-mock-v1"),
    mode: z.literal("mock"),
    summary: z.string().min(1).max(2000),
    coverage: z
      .object({
        enteredFields: z.literal(true),
        xmlDocuments: z.number().int().min(0),
        unreadDocuments: z.number().int().min(0),
        ocr: z.literal(false),
        fiscalAuthenticity: z.literal(false),
        bankVerification: z.literal(false),
      })
      .strict(),
    findings: z.array(findingSchema).max(100),
    suggestions: z.array(categorySuggestionSchema).max(10),
    extractions: z.array(extractionSuggestionSchema).max(3).default([]),
  })
  .strict();
export type AuditEvidence = z.infer<typeof evidenceSchema>;
export type AuditFinding = z.infer<typeof findingSchema>;
export type CategorySuggestion = z.infer<typeof categorySuggestionSchema>;
export type ExtractionSuggestion = z.infer<typeof extractionSuggestionSchema>;
export type AuditReport = z.infer<typeof auditReportSchema>;
export interface FindingReview {
  findingId: string;
  state: "confirmed" | "dismissed";
  reason: string;
  actor: string;
  version: number;
  createdAt: string;
}
export interface SuggestionReview {
  suggestionId: string;
  state: "applied" | "dismissed";
  reason: string;
  actor: string;
  createdAt: string;
}
export interface ExtractionReview {
  extractionId: string;
  state: "accepted" | "corrected" | "dismissed";
  reason: string;
  actor: string;
  proposedValue: string;
  finalValue: string | null;
  createdAt: string;
}
export interface AuditJob {
  id: string;
  expenseId: string;
  sourceFingerprint: string;
  state: "queued" | "running" | "completed" | "failed";
  attempts: number;
  version: number;
  createdAt: string;
  startedAt: string | null;
  finishedAt: string | null;
  error: string | null;
  stale: boolean;
  report: AuditReport | null;
  findingReviews: FindingReview[];
  suggestionReviews: SuggestionReview[];
  extractionReviews: ExtractionReview[];
}

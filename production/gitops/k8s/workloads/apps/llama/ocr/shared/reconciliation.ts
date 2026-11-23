import { z } from "zod";
import { payableDate } from "./payables.js";
import type { AuditEvent, Document } from "./types.js";
import type { MileageDeclaration } from "./mileage.js";

export const reconciliationReason = z.string().trim().min(5).max(1500);
export const reconciliationVersion = z.number().int().positive();
export const statementAccountSchema = z
  .object({
    label: z.string().trim().min(2).max(100),
    mode: z.enum(["simulation", "manual-unverified"]),
    currency: z.literal("BRL"),
  })
  .strict();
export const statementMappingSchema = z
  .object({
    delimiter: z.enum([",", ";", "\t"]),
    dateFormat: z.enum(["YYYY-MM-DD", "DD/MM/YYYY"]),
    decimal: z.enum([",", "."]),
    reference: z.string().trim().min(1).max(100),
    date: z.string().trim().min(1).max(100),
    description: z.string().trim().min(1).max(100),
    amount: z.string().trim().min(1).max(100),
    // Signed: negative = debit, positive = credit. Unsigned requires explicit tokens.
    direction: z.discriminatedUnion("mode", [
      z.object({ mode: z.literal("signed") }).strict(),
      z
        .object({
          mode: z.literal("column"),
          column: z.string().trim().min(1).max(100),
          debit: z.string().trim().min(1).max(40),
          credit: z.string().trim().min(1).max(40),
        })
        .strict()
        .refine(
          (v) => v.debit !== v.credit,
          "Débito e crédito devem ser distintos.",
        ),
    ]),
  })
  .strict()
  .refine((v) => {
    const names = [
      v.reference,
      v.date,
      v.description,
      v.amount,
      ...(v.direction.mode === "column" ? [v.direction.column] : []),
    ];
    return new Set(names).size === names.length;
  }, "Mapeie cada campo para uma coluna diferente.");
export const statementImportSchema = z
  .object({
    accountId: z.string().uuid(),
    filename: z.string().trim().min(1).max(160),
    csv: z.string().min(1).max(40000),
    mapping: statementMappingSchema,
  })
  .strict();
export const statementRowSchema = z
  .object({
    reference: z.string().trim().min(1).max(160),
    date: payableDate,
    description: z.string().trim().min(1).max(500),
    amount: z.number().int().positive().max(100_000_000),
    direction: z.enum(["debit", "credit"]),
  })
  .strict();
const matchBase = {
  entryVersion: reconciliationVersion,
  amount: z.number().int().positive().max(100_000_000),
  targetToken: z.string().regex(/^[a-f0-9]{64}$/),
  reason: reconciliationReason,
  requestKey: z.string().uuid(),
};
export const reconciliationMatchSchema = z.discriminatedUnion("kind", [
  z
    .object({
      ...matchBase,
      kind: z.literal("payment"),
      targetKind: z.enum(["reimbursement", "payment_batch"]),
      targetId: z.string().uuid(),
    })
    .strict(),
  z
    .object({
      ...matchBase,
      kind: z.literal("refund"),
      originalMatchId: z.string().uuid(),
    })
    .strict(),
]);
export const reconciliationDecisionSchema = z
  .object({
    version: reconciliationVersion,
    reason: reconciliationReason,
    requestKey: z.string().uuid(),
  })
  .strict();
export type StatementAccountFields = z.infer<typeof statementAccountSchema>;
export type StatementMapping = z.infer<typeof statementMappingSchema>;
export type StatementImportFields = z.infer<typeof statementImportSchema>;
export type StatementRow = z.infer<typeof statementRowSchema>;
export type ReconciliationMatchFields = z.infer<
  typeof reconciliationMatchSchema
>;
export type ReconciliationDecision = z.infer<
  typeof reconciliationDecisionSchema
>;
export interface StatementAccount extends StatementAccountFields {
  id: string;
  createdAt: string;
  bankConfirmed: false;
}
export interface StatementImport {
  id: string;
  accountId: string;
  filename: string;
  sourceHash: string;
  mapping: StatementMapping;
  state: "preview" | "committed";
  version: number;
  createdAt: string;
  confirmedAt: string | null;
  rows: Array<StatementRow & { existingEntryId: string | null }>;
  entryIds: string[];
  history: AuditEvent[];
}
export interface StatementEntry extends StatementRow {
  id: string;
  accountId: string;
  importId: string;
  mode: StatementAccountFields["mode"];
  state: "open" | "excluded";
  version: number;
  createdAt: string;
  bankConfirmed: false;
  matched: number;
  remaining: number;
  reconciliation: "unmatched" | "partial" | "matched" | "excluded";
  matches: ReconciliationMatch[];
  history: AuditEvent[];
}
export interface ReconciliationTarget {
  kind: "reimbursement" | "payment_batch";
  id: string;
  label: string;
  date: string;
  reference: string;
  amount: number;
  version: number;
  mode: StatementAccountFields["mode"];
  evidence: "manual-payment-report" | "simulated-payment-batch";
  allocated: number;
  refunded: number;
  remaining: number;
  token: string;
  bankConfirmed: false;
}
export interface ReconciliationMatch {
  id: string;
  entryId: string;
  kind: "payment" | "refund";
  originalMatchId: string | null;
  target: ReconciliationTarget;
  amount: number;
  state: "active" | "released";
  reason: string;
  createdAt: string;
  releasedAt: string | null;
  releaseReason: string | null;
}
export interface ReconciliationCandidate {
  target: ReconciliationTarget;
  suggestedAmount: number;
  reasons: string[];
  confidence: null;
  originalMatchId: string | null;
}
export interface ReconciliationEvidence {
  target: ReconciliationTarget;
  sources: Array<{
    id: string;
    label: string;
    date: string;
    amount: number;
    documents: Document[];
    mileage?: MileageDeclaration;
  }>;
  history: AuditEvent[];
}

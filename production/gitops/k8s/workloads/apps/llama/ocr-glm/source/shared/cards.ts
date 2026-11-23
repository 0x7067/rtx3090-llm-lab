import { z } from "zod";
import {
  categories,
  type AuditEvent,
  type Document,
  type Expense,
} from "./types.js";
import { payableDate } from "./payables.js";

export const cardReason = z.string().trim().min(5).max(1500);
export const cardVersion = z.number().int().positive();
export const cardControlsSchema = z
  .object({
    label: z.string().trim().min(3).max(100),
    purpose: z.string().trim().min(5).max(1500),
    monthlyLimit: z.number().int().min(0).max(100_000_000),
    categories: z
      .array(z.enum(categories))
      .min(1)
      .max(5)
      .refine((v) => new Set(v).size === v.length, "Não repita categorias."),
  })
  .strict();
export const cardCreateSchema = cardControlsSchema.extend({
  owner: z.string().regex(/^[a-zA-Z0-9-]{1,80}$/),
});
export type CardControls = z.infer<typeof cardControlsSchema>;
export type CardFields = z.infer<typeof cardCreateSchema>;
export interface SimulatedCard extends CardFields {
  /** Frozen identity binding for newly provisioned owners; absent on legacy cards. */
  ownerMemberId?: string;
  id: string;
  mode: "local-simulation";
  currency: "BRL";
  state: "active" | "frozen" | "closed";
  version: number;
  createdAt: string;
  updatedAt: string;
  history: AuditEvent[];
}
export const cardChargeSchema = z
  .object({
    merchant: z.string().trim().min(2).max(160),
    amount: z.number().int().positive().max(100_000_000),
    category: z.enum(categories),
  })
  .strict();
export const cardImportRowSchema = cardChargeSchema.extend({
  cardId: z.string().uuid(),
  sourceRef: z.string().trim().min(1).max(120),
  date: payableDate,
  state: z.enum(["pending", "settled"]),
});
export const cardImportSchema = z
  .object({
    source: z.literal("synthetic-statement"),
    rows: z.array(cardImportRowSchema).min(1).max(100),
  })
  .strict()
  .refine(
    (v) =>
      new Set(v.rows.map((r) => `${r.cardId}/${r.sourceRef}`)).size ===
      v.rows.length,
    "Há referências repetidas no mesmo arquivo.",
  );
export type CardImportFields = z.infer<typeof cardImportSchema>;
const csvColumn = z.string().trim().min(1).max(100);
export const cardCsvMappingSchema = z
  .object({
    delimiter: z.enum([",", ";", "\t"]),
    dateFormat: z.enum(["YYYY-MM-DD", "DD/MM/YYYY"]),
    decimal: z.enum([",", "."]),
    reference: csvColumn,
    merchant: csvColumn,
    amount: csvColumn,
    date: csvColumn,
    category: z.discriminatedUnion("mode", [
      z
        .object({ mode: z.literal("fixed"), value: z.enum(categories) })
        .strict(),
      z.object({ mode: z.literal("column"), column: csvColumn }).strict(),
    ]),
    state: z.discriminatedUnion("mode", [
      z
        .object({
          mode: z.literal("fixed"),
          value: z.enum(["pending", "settled"]),
        })
        .strict(),
      z
        .object({
          mode: z.literal("column"),
          column: csvColumn,
          pending: z.string().trim().min(1).max(40),
          settled: z.string().trim().min(1).max(40),
        })
        .strict()
        .refine(
          (v) => v.pending !== v.settled,
          "Situações devem ser distintas.",
        ),
    ]),
  })
  .strict()
  .refine((v) => {
    const names = [
      v.reference,
      v.merchant,
      v.amount,
      v.date,
      ...(v.category.mode === "column" ? [v.category.column] : []),
      ...(v.state.mode === "column" ? [v.state.column] : []),
    ];
    return new Set(names).size === names.length;
  }, "Mapeie cada campo para uma coluna diferente.");
export const cardCsvImportSchema = z
  .object({
    cardId: z.string().uuid(),
    filename: z.string().trim().min(1).max(160),
    csv: z.string().min(1).max(40000),
    mapping: cardCsvMappingSchema,
  })
  .strict();
export type CardCsvImportFields = z.infer<typeof cardCsvImportSchema>;
export type CardCsvMapping = z.infer<typeof cardCsvMappingSchema>;
export type CardCharge = z.infer<typeof cardChargeSchema>;
export interface CardRefund {
  preparedBy?: import("./actor.js").ActorContext;
  id: string;
  amount: number;
  date: string;
  reference: string;
  reason: string;
  createdAt: string;
}
export interface CardMatch {
  receipt?: import("./card-receipts.js").CardReceiptAttribution;
  id: string;
  expenseId: string;
  expenseVersion: number;
  amount: number;
  documents: Pick<Document, "id" | "sha256" | "fiscal">[];
  state: "active" | "released";
  reason: string;
  actor: string;
  createdAt: string;
  releasedAt?: string;
  releaseReason?: string;
}
export interface CardAuthorizationLink {
  id: string;
  statementId: string;
  authorizationId: string;
  reason: string;
  actor: string;
  createdAt: string;
  state: "active" | "released";
  releasedAt?: string;
  releaseReason?: string;
}
export interface CardAuthorizationCandidate {
  id: string;
  version: number;
  merchant: string;
  amount: number;
  date: string;
  state: "pending" | "settled";
  reasons: string[];
}
export const cardAuthorizationLinkSchema = z
  .object({
    authorizationId: z.string().uuid(),
    version: cardVersion,
    authorizationVersion: cardVersion,
    reason: cardReason,
    requestKey: z.string().uuid(),
  })
  .strict();
export const cardAuthorizationReleaseSchema = z
  .object({
    version: cardVersion,
    authorizationVersion: cardVersion,
    reason: cardReason,
    requestKey: z.string().uuid(),
  })
  .strict();
export interface CardTransaction extends CardCharge {
  captureCoverage?: import("./card-captures.js").CardCaptureCoverage;
  receiptCoverage?: {reserved:number;covered:number;remaining:number;available:number;stale:number;overReserved:boolean};
  id: string;
  cardId: string;
  owner: string;
  ownerMemberId?: string;
  sourceRef: string;
  date: string;
  origin: "local-authorization" | "synthetic-statement";
  mode: "local-simulation";
  state:
    | "pending"
    | "settled"
    | "partially_refunded"
    | "refunded"
    | "reversed"
    | "declined";
  version: number;
  createdAt: string;
  updatedAt: string;
  controls: {
    version: number;
    state: SimulatedCard["state"];
    monthlyLimit: number;
    categories: CardControls["categories"];
    usedBefore: number;
  };
  warnings: string[];
  refunds: CardRefund[];
  matches: CardMatch[];
  authorizationLink?: CardAuthorizationLink | null;
  followUp: {
    state: "requested" | "waived";
    dueDate: string;
    note: string;
    actor: string;
    updatedAt: string;
  } | null;
  history: AuditEvent[];
}
export interface CardImportPreview {
  id: string;
  version: number;
  state: "preview" | "committed";
  fields: CardImportFields;
  sourceFile?: { filename: string; sha256: string; mapping: CardCsvMapping };
  cardVersions: Record<string, number>;
  rows: {
    sourceRef: string;
    cardId: string;
    disposition: "new" | "replay";
    transactionId: string | null;
    warnings: string[];
  }[];
  createdAt: string;
  committedAt: string | null;
  transactionIds: string[];
  notice: string;
}
export interface CardCandidate {
  id: string;
  version: number;
  owner: string;
  merchant: string;
  amount: number;
  date: string;
  category: string;
  costCenter: string;
  allocations?: Expense["allocations"];
  documents: Document[];
  reasons: string[];
  provider: "deterministic-mock";
  confidence: null;
}
export interface CardOverview {
  mode: "local-simulation";
  month: string;
  currency: "BRL";
  notice: string;
  grossSettled: number;
  refundsOfTheseCharges: number;
  netSettled: number;
  pending: number;
  bankConfirmedPaid: 0;
  cards: { card: SimulatedCard; used: number; available: number }[];
  missingReceipts: {
    transactionId: string;
    owner: string;
    merchant: string;
    amount: number;
    date: string;
    version: number;
    followUp: CardTransaction["followUp"];
  }[];
  limitNotice: string;
}
export type CardReceipt = null | {
  match: CardMatch;
  expense: Pick<
    CardCandidate,
    | "id"
    | "version"
    | "owner"
    | "merchant"
    | "amount"
    | "date"
    | "category"
    | "costCenter"
    | "allocations"
    | "documents"
  > & { status: string };
  notice: string;
};
export const cardTransactionActionSchema = z.discriminatedUnion("action", [
  z
    .object({
      action: z.literal("settle"),
      version: cardVersion,
      reason: cardReason,
    })
    .strict(),
  z
    .object({
      action: z.literal("reverse"),
      version: cardVersion,
      reason: cardReason,
    })
    .strict(),
  z
    .object({
      action: z.literal("refund"),
      version: cardVersion,
      reason: cardReason,
      amount: z.number().int().positive().max(100_000_000),
      date: payableDate,
      reference: z.string().trim().min(3).max(120),
      requestKey: z.string().uuid(),
    })
    .strict(),
]);
export type CardTransactionAction = z.infer<typeof cardTransactionActionSchema>;

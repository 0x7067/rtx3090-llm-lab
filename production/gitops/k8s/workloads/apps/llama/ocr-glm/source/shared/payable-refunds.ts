import { z } from "zod";
import type { ActorContext } from "./actor.js";
import type {
  PayablePaymentSource,
  PayablePayment,
} from "./payable-payments.js";
import type {
  ReconciliationMatch,
  StatementRow,
  StatementMapping,
} from "./reconciliation.js";

export const payableRefundFieldsSchema = z
  .object({
    refundMatchId: z.string().uuid(),
    amount: z.number().int().positive().max(100_000_000),
    understood: z.literal(true),
  })
  .strict();
const common = {
  billVersion: z.number().int().positive(),
  basisToken: z.string().regex(/^[a-f0-9]{64}$/),
  reason: z.string().trim().min(10).max(1500),
  requestKey: z.string().uuid(),
};
export const payableRefundProposalSchema = z
  .object({ ...common, fields: payableRefundFieldsSchema })
  .strict();
export const payableRefundCommandSchema = z.discriminatedUnion("action", [
  z
    .object({
      ...common,
      version: z.number().int().positive(),
      action: z.literal("edit"),
      fields: payableRefundFieldsSchema,
    })
    .strict(),
  z
    .object({
      ...common,
      version: z.number().int().positive(),
      action: z.enum([
        "withdraw",
        "accept",
        "reject",
        "request-retraction",
        "accept-retraction",
        "reject-retraction",
      ]),
    })
    .strict(),
]);
export type PayableRefundFields = z.infer<typeof payableRefundFieldsSchema>;
export type PayableRefundProposal = z.infer<typeof payableRefundProposalSchema>;
export type PayableRefundCommand = z.infer<typeof payableRefundCommandSchema>;
export type PayableRefundState =
  | "proposed"
  | "accepted"
  | "rejected"
  | "withdrawn"
  | "retraction-review"
  | "retracted";
export const refundReserves = (state: PayableRefundState) =>
  ["proposed", "accepted", "retraction-review"].includes(state);
export const refundReported = (state: PayableRefundState) =>
  ["accepted", "retraction-review"].includes(state);
export interface RefundStatement {
  id: string;
  filename: string;
  sha256: string;
  mapping: StatementMapping;
}
export interface RefundEntry extends StatementRow {
  id: string;
  importId: string;
  accountId: string;
}
export interface PayableRefundSource {
  paymentId: string;
  paymentFields: PayablePayment["fields"];
  paymentAcceptance: NonNullable<PayablePayment["acceptance"]>;
  invoice: PayablePaymentSource;
  account: { id: string; label: string; mode: "manual-unverified" };
  debit: RefundEntry;
  credit: RefundEntry;
  debitMatch: ReconciliationMatch;
  refundMatch: ReconciliationMatch;
  statements: RefundStatement[];
}
export interface PayableRefundEvent {
  id: number;
  action: string;
  actor: ActorContext;
  reason: string;
  at: string;
  before: unknown;
  after: unknown;
}
export interface PayableRefund {
  id: string;
  billId: string;
  version: number;
  state: PayableRefundState;
  fields: PayableRefundFields;
  source: PayableRefundSource;
  contributors: ActorContext[];
  createdBy: ActorContext;
  createdAt: string;
  updatedAt: string;
  acceptance: {
    actor: ActorContext;
    at: string;
    version: number;
    reason: string;
  } | null;
  retractionRequestedBy: ActorContext | null;
  bankConfirmed: false;
  history: PayableRefundEvent[];
}
export interface PayableRefundCandidate {
  matchId: string;
  source: PayableRefundSource | null;
  issues: string[];
  gross: number;
  reserved: number;
  remaining: number;
}
export interface PayableRefundReview {
  billId: string;
  billVersion: number;
  candidates: PayableRefundCandidate[];
  records: PayableRefund[];
  current: Record<string, { current: boolean; issues: string[] }>;
  proposed: number;
  reported: number;
  bankConfirmed: false;
  basisToken: string;
}

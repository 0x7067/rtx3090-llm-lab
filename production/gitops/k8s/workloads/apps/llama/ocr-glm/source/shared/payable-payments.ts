import { z } from "zod";
import { payableDate } from "./payables.js";
import type { Vendor } from "./payables.js";
import type { ActorContext } from "./actor.js";
import type { AccountingSource } from "./accounting.js";
export const payablePaymentFieldsSchema = z
  .object({
    amount: z.number().int().positive().max(100_000_000),
    paidOn: payableDate,
    reference: z.string().trim().min(2).max(160),
    understood: z.literal(true),
  })
  .strict();
export type PayablePaymentFields = z.infer<typeof payablePaymentFieldsSchema>;
const common = {
  billVersion: z.number().int().positive(),
  basisToken: z.string().regex(/^[a-f0-9]{64}$/),
  reason: z.string().trim().min(10).max(1500),
  requestKey: z.string().uuid(),
};
export const payablePaymentProposalSchema = z
  .object({ ...common, fields: payablePaymentFieldsSchema })
  .strict();
export const payablePaymentCommandSchema = z.discriminatedUnion("action", [
  z
    .object({
      ...common,
      action: z.literal("edit"),
      version: z.number().int().positive(),
      fields: payablePaymentFieldsSchema,
    })
    .strict(),
  z
    .object({
      ...common,
      action: z.enum([
        "withdraw",
        "accept",
        "reject",
        "request-retraction",
        "accept-retraction",
        "reject-retraction",
      ]),
      version: z.number().int().positive(),
    })
    .strict(),
]);
export type PayablePaymentProposal = z.infer<
  typeof payablePaymentProposalSchema
>;
export type PayablePaymentCommand = z.infer<typeof payablePaymentCommandSchema>;
export interface PayablePaymentSource {
  source: AccountingSource;
  billVersion: number;
  invoiceNumber: string;
  vendor: Vendor;
  approvalFingerprint: string;
}
export type PayablePaymentState =
  | "proposed"
  | "accepted"
  | "rejected"
  | "withdrawn"
  | "retraction-review"
  | "retracted";
export interface PayablePaymentEvent {
  id: number;
  action: string;
  actor: ActorContext;
  reason: string;
  at: string;
  before: unknown;
  after: unknown;
}
export interface PayablePayment {
  id: string;
  billId: string;
  version: number;
  state: PayablePaymentState;
  fields: PayablePaymentFields;
  source: PayablePaymentSource;
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
  history: PayablePaymentEvent[];
}
export interface PayablePaymentReview {
  commercialBalance?: import("./payable-credit-notes.js").InvoiceCommercialBalance;
  billId: string;
  billVersion: number;
  source: PayablePaymentSource | null;
  issues: string[];
  records: PayablePayment[];
  proposed: number;
  declared: number;
  reserved: number;
  remaining: number;
  basisToken: string;
  bankConfirmed: false;
}
export const paymentReserves = (state: PayablePaymentState) =>
  ["proposed", "accepted", "retraction-review"].includes(state);
export const paymentDeclared = (state: PayablePaymentState) =>
  ["accepted", "retraction-review"].includes(state);

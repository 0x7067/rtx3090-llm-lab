import { z } from "zod";
import { payableDate, type BillFields, type Vendor } from "./payables.js";
import type { Document } from "./types.js";
import type { ActorContext } from "./actor.js";
import type { ApprovalRun } from "./approvals.js";

export const creditNoteFieldsSchema = z
  .object({
    reference: z.string().trim().min(1).max(80),
    issuedOn: payableDate,
    kind: z.enum(["price", "return", "commercial"]),
    amount: z.number().int().positive().max(100_000_000),
    includedTax: z.number().int().nonnegative().max(100_000_000).nullable(),
    lines: z
      .array(
        z.discriminatedUnion("component", [
          z
            .object({
              component: z.literal("item"),
              billItemIndex: z.number().int().min(0).max(49),
              amount: z.number().int().positive().max(100_000_000),
            })
            .strict(),
          z
            .object({
              component: z.literal("freight"),
              amount: z.number().int().positive().max(100_000_000),
            })
            .strict(),
        ]),
      )
      .min(1)
      .max(51),
  })
  .strict()
  .refine(
    (v) => v.lines.reduce((n, l) => n + l.amount, 0) === v.amount,
    "Os componentes devem somar o crédito bruto.",
  )
  .refine(
    (v) => v.includedTax === null || v.includedTax <= v.amount,
    "O tributo declarado não pode exceder o crédito bruto.",
  )
  .refine(
    (v) =>
      new Set(
        v.lines.map((l) =>
          l.component === "item" ? `item:${l.billItemIndex}` : "freight",
        ),
      ).size === v.lines.length,
    "Não repita componentes do crédito.",
  );
const common = {
  billVersion: z.number().int().positive(),
  basisToken: z.string().regex(/^[a-f0-9]{64}$/),
  reason: z.string().trim().min(10).max(1500),
  requestKey: z.string().uuid(),
  understood: z.literal(true),
};
export const creditNoteProposalSchema = z
  .object({ ...common, fields: creditNoteFieldsSchema })
  .strict();
export const creditNoteCommandSchema = z
  .object({
    ...common,
    version: z.number().int().positive(),
    action: z.enum([
      "accept",
      "reject",
      "withdraw",
      "request-retraction",
      "retract",
      "reject-retraction",
    ]),
  })
  .strict();
export type CreditNoteFields = z.infer<typeof creditNoteFieldsSchema>;
export type CreditNoteProposal = z.infer<typeof creditNoteProposalSchema>;
export type CreditNoteCommand = z.infer<typeof creditNoteCommandSchema>;
export type CreditNoteState =
  | "proposed"
  | "accepted"
  | "rejected"
  | "withdrawn"
  | "retraction-review"
  | "retracted";
export const creditReserves = (state: CreditNoteState) =>
  ["proposed", "accepted", "retraction-review"].includes(state);
export const creditAccepted = (state: CreditNoteState) =>
  ["accepted", "retraction-review"].includes(state);
export interface CreditNoteSource {
  billId: string;
  fields: BillFields;
  vendor: Vendor;
  documents: Document[];
  approval: ApprovalRun;
}
export interface PayableCreditNote {
  id: string;
  billId: string;
  version: number;
  state: CreditNoteState;
  fields: CreditNoteFields;
  source: CreditNoteSource;
  sourceToken: string;
  original: Document;
  preparedBy: ActorContext;
  preparedAt: string;
  reason: string;
  acceptedBy?: ActorContext;
  acceptedAt?: string;
  acceptanceReason?: string;
  correctionRequestedBy?: ActorContext;
}
export interface CreditNoteEvent {
  id: number;
  creditId: string;
  action: string;
  actor: ActorContext;
  at: string;
  reason: string;
  before: PayableCreditNote | null;
  after: PayableCreditNote;
}
export interface CreditNoteReview {
  commercialBalance?: InvoiceCommercialBalance;
  billId: string;
  billVersion: number;
  basisToken: string;
  source: CreditNoteSource | null;
  sourceIssues: string[];
  progressionIssues: string[];
  records: (PayableCreditNote & { current: boolean; issues: string[] })[];
  events: CreditNoteEvent[];
  proposed: number;
  accepted: number;
  remainingCreditCapacity: number;
  cashRecovered: null;
}
/** Current commercial facts; reported payments/refunds are not bank evidence. */
export interface InvoiceCommercialBalance {
  token: string;
  originalGross: number;
  proposedCredits: number | null;
  acceptedCredits: number | null;
  netObligation: number | null;
  declaredPayments: number;
  proposedPayments: number;
  reviewedRefunds: number;
  netReportedSettlement: number;
  commercialUnpaid: number | null;
  supplierReceivable: number | null;
  availableForNewPayment: number | null;
  reservedAboveNet: number | null;
  underReview: boolean;
  current: boolean;
  issues: string[];
  bankConfirmed: false;
  cashRecovered: null;
}

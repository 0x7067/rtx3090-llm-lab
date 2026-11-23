import { z } from "zod";
import { payableDate } from "./payables.js";
import type { ActorContext } from "./actor.js";
import type { Reimbursement } from "./types.js";
import type {
  ApprovedExpenseFundingSource,
  ExpenseFundingPart,
} from "./expense-funding.js";
export type ReimbursementInstallmentState =
  "scheduled" | "paid" | "cancelled" | "retraction-review" | "retracted";
export interface ReimbursementInstallmentSummary {
  id: string;
  expenseId: string;
  version: number;
  state: ReimbursementInstallmentState;
  amount: number;
  date: string;
  reference: string;
  parts: ExpenseFundingPart[];
  sourceFingerprint: string;
  paidVersion: number | null;
  legacy: boolean;
  legacyTargetVersion?: number;
}
export interface ReimbursementInstallmentEvent {
  id: number;
  action: string;
  actor: ActorContext;
  at: string;
  reason: string;
  before: unknown;
  after: unknown;
}
export interface ReimbursementInstallment extends ReimbursementInstallmentSummary {
  source: ApprovedExpenseFundingSource;
  createdBy: ActorContext;
  createdAt: string;
  contributors: ActorContext[];
  payment: {
    actor: ActorContext;
    at: string;
    reason: string;
    version: number;
    date: string;
    reference: string;
  } | null;
  retractionRequestedBy: ActorContext | null;
  legacySnapshot?: Reimbursement;
  bankConfirmed: false;
  history: ReimbursementInstallmentEvent[];
}
export interface ReimbursementLedgerSummary {
  version: number;
  records: ReimbursementInstallmentSummary[];
}
export { installmentReserves, installmentDeclared } from "./expense-funding.js";
const common = {
  expenseVersion: z.number().int().positive(),
  ledgerVersion: z.number().int().nonnegative(),
  basisToken: z.string().regex(/^[a-f0-9]{64}$/),
  reason: z.string().trim().min(10).max(1500),
  requestKey: z.string().uuid(),
};
export const installmentFieldsSchema = z
  .object({
    amount: z.number().int().positive().max(100_000_000),
    date: payableDate,
    reference: z.string().trim().min(5).max(200),
    parts: z
      .array(
        z
          .object({
            allocationId: z.string().min(1).max(80),
            amount: z.number().int().positive().max(100_000_000),
          })
          .strict(),
      )
      .min(1)
      .max(30),
    understood: z.literal(true),
  })
  .strict();
export const reimbursementLedgerActivateSchema = z
  .object({ ...common, understood: z.literal(true) })
  .strict();
export const reimbursementInstallmentCreateSchema = z
  .object({ ...common, fields: installmentFieldsSchema })
  .strict();
export const reimbursementInstallmentCommandSchema = z.discriminatedUnion(
  "action",
  [
    z
      .object({
        ...common,
        version: z.number().int().positive(),
        action: z.literal("revise"),
        fields: installmentFieldsSchema,
      })
      .strict(),
    z
      .object({
        ...common,
        version: z.number().int().positive(),
        action: z.literal("report-paid"),
        date: payableDate,
        reference: z.string().trim().min(5).max(200),
        understood: z.literal(true),
      })
      .strict(),
    z
      .object({
        ...common,
        version: z.number().int().positive(),
        action: z.enum([
          "cancel",
          "request-retraction",
          "accept-retraction",
          "reject-retraction",
        ]),
      })
      .strict(),
  ],
);
export type ReimbursementLedgerActivate = z.infer<
  typeof reimbursementLedgerActivateSchema
>;
export type ReimbursementInstallmentCreate = z.infer<
  typeof reimbursementInstallmentCreateSchema
>;
export type ReimbursementInstallmentCommand = z.infer<
  typeof reimbursementInstallmentCommandSchema
>;

export interface ReimbursementLedgerReport {
  expense: import("./types.js").Expense;
  active: boolean;
  ledgerVersion: number;
  source: ApprovedExpenseFundingSource | null;
  records: ReimbursementInstallment[];
  sourceIssues: string[];
  holds: string[];
  cardReserved: number;
  reserved: number;
  available: number;
  declared: number;
  scheduled: number;
  parts: (import("./types.js").ExpenseAllocation & {
    reserved: number;
    available: number;
    cardReserved: number;
    reimbursementReserved: number;
  })[];
  canActivate: boolean;
  basisToken: string;
  bankConfirmed: false;
}

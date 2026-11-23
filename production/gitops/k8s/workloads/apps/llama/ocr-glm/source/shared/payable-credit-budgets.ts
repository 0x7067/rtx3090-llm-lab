import { z } from "zod";
import { accountingMonth } from "./accounting.js";
import type { ActorContext } from "./actor.js";
import type { PayableCreditNote } from "./payable-credit-notes.js";

export const creditBudgetAllocationsSchema = z
  .array(
    z
      .object({
        month: accountingMonth,
        amount: z.number().int().positive().max(100_000_000),
      })
      .strict(),
  )
  .min(1)
  .max(36)
  .refine(
    (parts) => new Set(parts.map((p) => p.month)).size === parts.length,
    "Informe cada mês apenas uma vez.",
  );
const common = {
  billVersion: z.number().int().positive(),
  basisToken: z.string().regex(/^[a-f0-9]{64}$/),
  reason: z.string().trim().min(10).max(1500),
  requestKey: z.string().uuid(),
  understood: z.literal(true),
};
export const creditBudgetProposalSchema = z
  .object({ ...common, allocations: creditBudgetAllocationsSchema })
  .strict();
export const creditBudgetCommandSchema = z
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
export type CreditBudgetProposal = z.infer<typeof creditBudgetProposalSchema>;
export type CreditBudgetCommand = z.infer<typeof creditBudgetCommandSchema>;
export type CreditBudgetAllocation = z.infer<
  typeof creditBudgetAllocationsSchema
>[number];
export type CreditBudgetState =
  | "proposed"
  | "accepted"
  | "rejected"
  | "withdrawn"
  | "retraction-review"
  | "retracted";
export interface CreditBudgetSource {
  note: PayableCreditNote;
  originalMonths: CreditBudgetAllocation[];
  costCenter: string;
  category: string;
  monthBasis: "purchase" | "issue" | "service";
  token: string;
}
export interface CreditBudgetRecord {
  id: string;
  billId: string;
  creditId: string;
  version: number;
  state: CreditBudgetState;
  source: CreditBudgetSource;
  allocations: CreditBudgetAllocation[];
  preparedBy: ActorContext;
  preparedAt: string;
  reason: string;
  acceptedBy?: ActorContext;
  acceptedAt?: string;
  acceptanceReason?: string;
  correctionRequestedBy?: ActorContext;
}
export interface CreditBudgetEvent {
  id: number;
  recordId: string;
  action: string;
  actor: ActorContext;
  at: string;
  reason: string;
  before: CreditBudgetRecord | null;
  after: CreditBudgetRecord;
}
export interface CreditBudgetContext {
  billId: string;
  creditId: string;
  billVersion: number;
  basisToken: string;
  source: CreditBudgetSource | null;
  issues: string[];
  records: (CreditBudgetRecord & { history: CreditBudgetEvent[] })[];
  months: {
    month: string;
    gross: number;
    reserved: number;
    remaining: number;
    period: { state: "open" | "locked"; version: number } | null;
  }[];
}
export interface CreditBudgetEffect {
  token: string;
  proposed: number | null;
  accepted: number | null;
  applied: CreditBudgetAllocation[];
  current: boolean;
  underReview: boolean;
  issues: string[];
  records: {
    id: string;
    creditId: string;
    version: number;
    state: CreditBudgetState;
    reference: string;
    allocations: CreditBudgetAllocation[];
  }[];
  cashRecovered: null;
}
export interface CreditBudgetMonthReview {
  month: string;
  token: string;
  pending: string[];
  sourceIssues: { billId: string; reason: string }[];
  accepted: number;
  records: CreditBudgetRecord[];
}
export const budgetCreditAccepted = (state: CreditBudgetState) =>
  ["accepted", "retraction-review"].includes(state);
export const budgetCreditReserves = (state: CreditBudgetState) =>
  ["proposed", "accepted", "retraction-review"].includes(state);

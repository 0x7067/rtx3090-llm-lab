import { z } from "zod";
import type { ActorContext } from "./actor.js";
import type { Expense } from "./types.js";
import { payableDate } from "./payables.js";
export const advanceMoney = z.number().int().positive().max(100_000_000);
export const advanceReason = z.string().trim().min(10).max(1500);
export const advanceFields = z
  .object({
    title: z.string().trim().min(3).max(160),
    purpose: z.string().trim().min(10).max(1500),
    amount: advanceMoney,
    costCenter: z.string().trim().min(1).max(100),
    accountBy: payableDate,
  })
  .strict();
export type AdvanceFields = z.infer<typeof advanceFields>;
const command = z
  .object({ requestKey: z.string().uuid(), reason: advanceReason })
  .strict();
const versioned = command.extend({ version: z.number().int().positive() });
export const advanceCreate = command.extend({ fields: advanceFields });
export const advanceCommand = z.discriminatedUnion("action", [
  versioned.extend({ action: z.literal("edit"), fields: advanceFields }),
  versioned.extend({
    action: z.enum([
      "submit",
      "clarify",
      "approve",
      "reject",
      "cancel",
      "close",
      "reopen",
    ]),
  }),
  versioned.extend({
    action: z.literal("cash"),
    kind: z.enum(["funding", "return"]),
    amount: advanceMoney,
    reportedOn: payableDate,
    reference: z.string().trim().min(3).max(160),
  }),
  versioned.extend({
    action: z.literal("cash-decision"),
    recordId: z.string().uuid(),
    decision: z.enum(["accept", "reject", "void"]),
  }),
  versioned.extend({
    action: z.literal("apply"),
    expenseId: z.string().uuid(),
    expenseVersion: z.number().int().positive(),
    amount: advanceMoney,
  }),
  versioned.extend({
    action: z.literal("application-decision"),
    recordId: z.string().uuid(),
    decision: z.enum(["accept", "reject", "release"]),
  }),
]);
export type AdvanceCommand = z.infer<typeof advanceCommand>;
export interface AdvanceDecision {
  actor: ActorContext;
  reason: string;
  at: string;
}
export interface AdvanceCash {
  id: string;
  kind: "funding" | "return";
  amount: number;
  reportedOn: string;
  reference: string;
  state: "proposed" | "accepted" | "rejected" | "voided";
  prepared: AdvanceDecision;
  review: AdvanceDecision | null;
  bankConfirmed: false;
}
export interface AdvanceSource {
  expense: Expense;
  approvalFingerprint: string;
}
export interface AdvanceApplication {
  id: string;
  expenseId: string;
  amount: number;
  costCenter: string;
  state: "proposed" | "accepted" | "rejected" | "released";
  source: AdvanceSource;
  prepared: AdvanceDecision;
  review: AdvanceDecision | null;
}
export interface AdvanceSettlement {
  funding: number;
  returned: number;
  applied: number;
  outstanding: number;
  reservedApplications: number;
  reservedReturns: number;
  available: number;
  bankConfirmed: false;
  actualCash: null;
}
export interface AdvanceClosure extends AdvanceDecision {
  settlement: AdvanceSettlement;
  cash: AdvanceCash[];
  applications: AdvanceApplication[];
  reopened: AdvanceDecision | null;
}
export interface AdvanceRecord {
  id: string;
  version: number;
  owner: ActorContext;
  fields: AdvanceFields;
  state:
    | "draft"
    | "submitted"
    | "needs-info"
    | "approved"
    | "rejected"
    | "cancelled"
    | "closed";
  contributors: string[];
  createdAt: string;
  approval:
    | (AdvanceDecision & {
        fields: AdvanceFields;
        center: { id: string; name: string; version: number } | null;
      })
    | null;
  cash: AdvanceCash[];
  applications: AdvanceApplication[];
  closures: AdvanceClosure[];
}
export interface AdvanceHistory {
  before: AdvanceRecord | null;
  after: AdvanceRecord;
  action: string;
  actor: ActorContext;
  reason: string;
  at: string;
}
export interface AdvanceView extends AdvanceRecord {
  settlement: AdvanceSettlement;
  applicationStatus: { id: string; current: boolean; issues: string[] }[];
  history: AdvanceHistory[];
  consultedOn: string;
}
export const advanceLabels: Record<AdvanceRecord["state"], string> = {
  draft: "Rascunho",
  submitted: "Em análise",
  "needs-info": "Precisa de ajuste",
  approved: "Aprovado",
  rejected: "Recusado",
  cancelled: "Cancelado",
  closed: "Encerrado administrativamente",
};

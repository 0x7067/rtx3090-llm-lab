import { z } from "zod";
import type { AuditEvent } from "./types.js";
import { accountingReason, accountingVersion } from "./accounting.js";

export const closingTasks = ["entries", "evidence", "movements"] as const;
export const closingTaskLabels = {
  entries: "Lançamentos e rateios conferidos",
  evidence: "Origens e documentos conferidos",
  movements: "Movimentos do mês conferidos",
} satisfies Record<(typeof closingTasks)[number], string>;
export const closingReviewSchema = z
  .object({
    version: accountingVersion,
    periodVersion: accountingVersion,
    basisToken: z.string().regex(/^[a-f0-9]{64}$/),
    task: z.enum(closingTasks),
    complete: z.boolean(),
    reason: accountingReason,
    requestKey: z.string().uuid(),
  })
  .strict();
export type ClosingReview = z.infer<typeof closingReviewSchema>;
export interface ClosingTask {
  responsibility?:import("./accounting-responsibilities.js").ClosingResponsibility;
  reviewedActor?:import("./actor.js").ActorContext|null;
  id: (typeof closingTasks)[number];
  complete: boolean;
  basisToken: string | null;
  reviewedBy: string | null;
  reviewedAt: string | null;
  reason: string | null;
}
export interface ClosingChecklist {
  responsibilitiesEnabled?:true;
  month: string;
  version: number;
  tasks: ClosingTask[];
  history: AuditEvent[];
}
export interface ClosingSnapshot {
  statementClose?: import("./accounting-statement-close.js").StatementCloseReport;
  budgetCredits?: import("./payable-credit-budgets.js").CreditBudgetMonthReview;
  sourceClose?: import("./accounting-source-close.js").SourceCloseReport;
  checklistVersion: number;
  basisToken: string;
  tasks: ClosingTask[];
  journalCount: number;
  debit: number;
  credit: number;
  closedBy: string;
  closedAt: string;
}
export interface ClosingReadiness {
  statementClose?: import("./accounting-statement-close.js").StatementCloseReport;
  budgetCredits?: import("./payable-credit-budgets.js").CreditBudgetMonthReview;
  sourceClose?: import("./accounting-source-close.js").SourceCloseReport;
  month: string;
  periodVersion: number;
  periodState: "open" | "locked";
  enabled: boolean;
  checklist: ClosingChecklist | null;
  tasks: (ClosingTask & { state: "pending" | "current" | "stale" } & Partial<import("./accounting-responsibilities.js").ClosingResponsibilityStatus>)[];
  basisToken: string;
  drafts: string[];
  sourceIssues: {
    journalId: string;
    state: "current" | "changed" | "unavailable" | "not-linked";
    token: string | null;
  }[];
  movements: {
    rows: { accountId: string; debit: number; credit: number }[];
    journalCount: number;
    debit: number;
    credit: number;
    scope: "local-posted-movements";
    openingBalance: null;
    erpConfirmed: boolean;
  };
  ready: boolean;
  lastClose: ClosingSnapshot | null;
  scope: "local-accounting-review";
  fiscalCertified: false;
  erpConfirmed: false;
}

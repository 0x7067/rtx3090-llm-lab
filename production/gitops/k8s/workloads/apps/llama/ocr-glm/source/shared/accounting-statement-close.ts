import { z } from "zod";
import type { ActorContext } from "./actor.js";
import { accountingVersion } from "./accounting.js";
import type { StatementAccount } from "./reconciliation.js";
import type { StatementEntryReview } from "./reconciliation-evidence.js";

export const statementCloseAccounts = z
  .array(z.string().uuid())
  .min(1)
  .max(20)
  .refine((ids) => new Set(ids).size === ids.length, "Não repita contas.");
const base = z.object({
  version: z.number().int().nonnegative(),
  periodVersion: accountingVersion,
  basisToken: z.string().regex(/^[a-f0-9]{64}$/),
  reason: z.string().trim().min(10).max(1500),
  requestKey: z.string().uuid(),
});
export const statementCloseCommandSchema = z.discriminatedUnion("action", [
  base
    .extend({
      action: z.literal("include-accounts"),
      accountIds: statementCloseAccounts,
      understood: z.literal(true),
    })
    .strict(),
  base
    .extend({
      action: z.literal("exclude"),
      entryId: z.string().uuid(),
      understood: z.literal(true),
    })
    .strict(),
  base
    .extend({ action: z.literal("release"), exclusionId: z.string().uuid() })
    .strict(),
]);
export type StatementCloseCommand = z.infer<typeof statementCloseCommandSchema>;
export interface StatementCloseExclusion {
  id: string;
  review: StatementEntryReview;
  reason: string;
  preparedBy: ActorContext;
  preparedAt: string;
  state: "active" | "released";
  releasedBy?: ActorContext;
  releasedAt?: string;
  releaseReason?: string;
}
export interface StatementCloseEvent {
  action: "accounts-included" | "excluded" | "released";
  actor: ActorContext;
  at: string;
  reason: string;
  accountIds?: string[];
  exclusionId?: string;
}
export interface StatementCloseControl {
  month: string;
  version: number;
  accountIds: string[];
  enabledBy: ActorContext;
  enabledAt: string;
  reason: string;
  exclusions: StatementCloseExclusion[];
  history: StatementCloseEvent[];
}
export interface StatementCloseReport {
  month: string;
  periodVersion: number;
  periodState: "open" | "locked";
  control: StatementCloseControl | null;
  availableAccounts: StatementAccount[];
  consultedAccountIds: string[];
  accounts: {
    account: StatementAccount;
    debit: number;
    credit: number;
    matchedDebit: number;
    matchedCredit: number;
    remainingDebit: number;
    remainingCredit: number;
    excludedRows: number;
  }[];
  rows: (StatementEntryReview & {
    exclusionId: string | null;
    resolved: boolean;
  })[];
  staleExclusionIds: string[];
  unresolved: number;
  ready: boolean;
  basisToken: string;
  scope: "selected-manual-statement-accounts";
  bankConfirmed: false;
  balance: null;
}

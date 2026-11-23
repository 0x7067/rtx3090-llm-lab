import { z } from "zod";
import {
  accountingDate,
  type AccountingSource,
  type Journal,
  type LedgerAccount,
} from "./accounting.js";
import type { ActorContext } from "./actor.js";
import type {
  AdvanceRecord,
  AdvanceCash,
  AdvanceApplication,
} from "./advances.js";

export const advanceAccountingKind = z.enum([
  "funding",
  "return",
  "application",
]);
export type AdvanceAccountingKind = z.infer<typeof advanceAccountingKind>;
export const advanceFundingAllocations = z
  .array(
    z
      .object({
        journalId: z.string().uuid(),
        amount: z.number().int().positive().max(100_000_000),
      })
      .strict(),
  )
  .min(1)
  .max(200)
  .refine(
    (rows) => new Set(rows.map((r) => r.journalId)).size === rows.length,
    "Não repita o mesmo recebimento contábil.",
  );
const common = {
  advanceId: z.string().uuid(),
  recordId: z.string().uuid(),
  basisToken: z.string().regex(/^[a-f0-9]{64}$/),
  date: accountingDate,
  understood: z.literal(true),
};
export const advanceAccountingFields = z.discriminatedUnion("kind", [
  z
    .object({
      ...common,
      kind: z.literal("funding"),
      advanceAccountId: z.string().uuid(),
      cashAccountId: z.string().uuid(),
    })
    .strict(),
  z
    .object({
      ...common,
      kind: z.literal("return"),
      cashAccountId: z.string().uuid(),
      fundings: advanceFundingAllocations,
    })
    .strict(),
  z
    .object({
      ...common,
      kind: z.literal("application"),
      recognitionId: z.string().uuid(),
      fundings: advanceFundingAllocations,
    })
    .strict(),
]);
export type AdvanceAccountingFields = z.infer<typeof advanceAccountingFields>;
export interface AdvanceAccountingSource {
  advanceId: string;
  owner: ActorContext;
  approval: NonNullable<AdvanceRecord["approval"]>;
  record: AdvanceCash | AdvanceApplication;
  kind: AdvanceAccountingKind;
  expense?: AccountingSource;
  token: string;
  bankConfirmed: false;
}
export type AdvanceAccount = Pick<
  LedgerAccount,
  "id" | "code" | "name" | "kind"
>;
export interface AdvanceFundingParent {
  id: string;
  version: number;
  date: string;
  source: AdvanceAccountingSource;
  amount: number;
  advanceAccount: AdvanceAccount;
  cashAccount: AdvanceAccount;
  token: string;
}
export interface AdvanceRecognitionParent {
  id: string;
  version: number;
  date: string;
  source: AccountingSource;
  lines: Journal["lines"];
  accounts: Journal["accounts"];
  liabilityAccount: AdvanceAccount;
  amount: number;
}
export interface AdvanceAccountingBasis {
  kind: "reviewed-general-advance-v1";
  leg: AdvanceAccountingKind;
  source: AdvanceAccountingSource;
  amount: number;
  advanceAccount: AdvanceAccount;
  cashAccount?: AdvanceAccount;
  fundings: { parent: AdvanceFundingParent; amount: number }[];
  recognition?: AdvanceRecognitionParent;
  token: string;
  bankConfirmed: false;
}
export interface AdvanceAccountingContext {
  source: AdvanceAccountingSource;
  claimedBy: string[];
  advanceAccountId: string | null;
  fundingPolicies: { journalId: string; advanceAccountId: string }[];
  fundings: (AdvanceFundingParent & { reserved: number; remaining: number })[];
  recognitions: (AdvanceRecognitionParent & {
    reserved: number;
    remaining: number;
  })[];
}
export interface AdvanceAccountingCandidate {
  advanceId: string;
  recordId: string;
  kind: AdvanceAccountingKind;
  label: string;
  owner: string;
  amount: number;
  date: string;
  state: "available" | "reserved" | "pending" | "unavailable";
  issues: string[];
  journals: { id: string; state: Journal["state"]; version: number }[];
}

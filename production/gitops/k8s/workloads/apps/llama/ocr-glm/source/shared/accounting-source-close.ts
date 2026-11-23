import { z } from "zod";
import {
  accountingVersion,
  accountingSourceId,
  type AccountingSource,
  type Journal,
} from "./accounting.js";
import type { ActorContext } from "./actor.js";
import type { Document, ExpenseFields } from "./types.js";
import type { MileageDeclaration } from "./mileage.js";
export const sourceCloseDecisionSchema = z
  .object({
    basisToken: z.string().regex(/^[a-f0-9]{64}$/),
    checklistVersion: accountingVersion,
  })
  .strict();
export type SourceCloseDecision = z.infer<typeof sourceCloseDecisionSchema>;
export const settlementCloseKinds = [
  "reimbursement_payment",
  "reimbursement_installment",
  "payable_payment",
  "payable_refund",
  "advance_funding",
  "advance_return",
  "advance_application",
] as const;
export type SettlementCloseKind = (typeof settlementCloseKinds)[number];
export const sourceCloseKind = z.enum([
  "expense",
  "payable",
  "payable_credit",
  "card_refund",
  "payable_batch_item",
  ...settlementCloseKinds,
]);
export interface SettlementCloseEvidence {
  kind: SettlementCloseKind;
  claimKey: string;
  parentId: string;
  reference: string;
  dateBasis:
    | "declared-payment"
    | "imported-refund-credit"
    | "declared-advance-cash"
    | "independent-application-acceptance";
  amountBasis: "declared" | "legacy-expense-reference";
  evidence: unknown;
  reviews: { label: string; actor: ActorContext; at: string; reason: string }[];
  bankConfirmed: false;
}

const base = z.object({
  version: z.number().int().nonnegative(),
  periodVersion: accountingVersion,
  basisToken: z.string().regex(/^[a-f0-9]{64}$/),
  reason: z.string().trim().min(10).max(1500),
  requestKey: z.string().uuid(),
});
export const sourceCloseCommandSchema = z.discriminatedUnion("action", [
  base
    .extend({ action: z.literal("enable"), understood: z.literal(true) })
    .strict(),
  base
    .extend({
      action: z.literal("enable-credits"),
      understood: z.literal(true),
    })
    .strict(),
  base
    .extend({
      action: z.literal("enable-settlements"),
      understood: z.literal(true),
    })
    .strict(),
  base
    .extend({
      action: z.literal("enable-card-refunds"),
      understood: z.literal(true),
    })
    .strict(),
  base
    .extend({
      action: z.literal("enable-payment-batches"),
      understood: z.literal(true),
    })
    .strict(),
  base
    .extend({
      action: z.literal("exclude"),
      kind: sourceCloseKind,
      sourceId: accountingSourceId,
      understood: z.literal(true),
    })
    .strict(),
  base
    .extend({ action: z.literal("release"), exclusionId: z.string().uuid() })
    .strict(),
]);
export type SourceCloseCommand = z.infer<typeof sourceCloseCommandSchema>;
export type SourceCloseStatus =
  | "recognized"
  | "partial"
  | "unrecognized"
  | "proposal"
  | "other-period"
  | "source-issue"
  | "shared-identity";
export const sourceCloseLabels: Record<SourceCloseStatus, string> = {
  recognized: "Reconhecida nesta competência",
  partial: "Contabilização parcial nesta competência",
  unrecognized: "Sem reconhecimento ativo",
  proposal: "Proposta ainda não contabilizada",
  "other-period": "Reconhecimento em outra competência",
  "source-issue": "Origem ou reconhecimento precisa de conferência",
  "shared-identity": "Documento ou chave fiscal em outro reconhecimento",
};
export interface SourceCloseCandidate {
  kind:
    | AccountingSource["kind"]
    | "payable_credit"
    | "card_refund"
    | "payable_batch_item"
    | SettlementCloseKind;
  id: string;
  label: string;
  date: string;
  amount: number;
  version: number;
  documents: Document[];
  declaration:
    | (Pick<
        ExpenseFields,
        "category" | "costCenter" | "purpose" | "allocations"
      > & { mileage?: MileageDeclaration })
    | null;
  source: AccountingSource | null;
  creditNote?: import("./payable-credit-notes.js").PayableCreditNote;
  settlement?: SettlementCloseEvidence;
  cardRefund?: import("./accounting-card-refunds.js").CardRefundCloseEvidence;
  batchSimulation?: import("./accounting-payment-batches.js").PaymentBatchCloseEvidence;
  issue: string | null;
  fingerprint: string;
  status: SourceCloseStatus;
  journals: (Pick<
    Journal,
    "id" | "date" | "state" | "version" | "description"
  > & { direct: boolean; current: boolean; amount?: number })[];
}
export interface SourceCloseExclusion {
  id: string;
  source: SourceCloseCandidate;
  reason: string;
  preparedBy: ActorContext;
  preparedAt: string;
  state: "active" | "released";
  releasedBy?: ActorContext;
  releasedAt?: string;
  releaseReason?: string;
}
export interface SourceCloseEvent {
  action:
    | "enabled"
    | "credits-enabled"
    | "settlements-enabled"
    | "card-refunds-enabled"
    | "payment-batches-enabled"
    | "excluded"
    | "released";
  actor: ActorContext;
  reason: string;
  at: string;
  exclusion?: SourceCloseExclusion;
}
export interface SourceCloseControl {
  month: string;
  version: number;
  enabledBy: ActorContext;
  enabledAt: string;
  reason: string;
  includeCredits?: true;
  includeSettlements?: true;
  includeCardRefunds?: true;
  includePaymentBatches?: true;
  exclusions: SourceCloseExclusion[];
  history: SourceCloseEvent[];
}
export interface SourceCloseReport {
  month: string;
  periodVersion: number;
  periodState: "open" | "locked";
  control: SourceCloseControl | null;
  basisToken: string;
  rows: (SourceCloseCandidate & {
    exclusionId: string | null;
    resolved: boolean;
  })[];
  staleExclusionIds: string[];
  unresolved: number;
  ready: boolean;
  scope: "approved-source-document-month";
  bankConfirmed: false;
  fiscalCertified: false;
}

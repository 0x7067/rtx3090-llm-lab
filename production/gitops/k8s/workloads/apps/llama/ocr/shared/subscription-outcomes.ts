import { z } from "zod";
import { payableDate } from "./payables.js";
import type { AuditEvent } from "./types.js";
import type { Subscription, SubscriptionRevision } from "./subscriptions.js";
import type {
  SubscriptionObservation,
  SubscriptionSource,
} from "./subscription-evidence.js";

export const subscriptionCycleDatesSchema = z
  .object({
    beforeDate: payableDate,
    afterDate: payableDate,
  })
  .strict()
  .refine(
    (v) => v.beforeDate < v.afterDate,
    "O ciclo inicial deve preceder o ciclo final.",
  );
export const subscriptionCycleComparisonSchema = z
  .object({
    subscriptionId: z.string().uuid(),
    beforeDate: payableDate,
    afterDate: payableDate,
    basisToken: z.string().regex(/^[a-f0-9]{64}$/),
  })
  .strict()
  .refine(
    (v) => v.beforeDate < v.afterDate,
    "O ciclo inicial deve preceder o ciclo final.",
  );
export type SubscriptionCycleComparisonFields = z.infer<
  typeof subscriptionCycleComparisonSchema
>;
export interface SubscriptionComparisonObservation extends SubscriptionObservation {
  currentSource: SubscriptionSource | null;
  currentClaimKeys: string[];
  identityCorrections: {
    id: string;
    version: number;
    state: "accepted";
    matchId: string;
    fingerprint: string;
    reason: string;
    reviewedAt: string | null;
  }[];
}
export interface SubscriptionCycleSnapshot {
  billingDate: string;
  terms: SubscriptionRevision;
  observations: SubscriptionComparisonObservation[];
  grossAmount: number;
  refundAmount: number | null;
  hasSimulation: boolean;
  afterRecordedEnd: boolean;
  outsideNonrenewingTerm: boolean;
}
export interface SubscriptionCycleComparisonPreview {
  kind: "observed-cycles";
  subscriptionId: string;
  beforeDate: string;
  afterDate: string;
  basisToken: string;
  before: SubscriptionCycleSnapshot;
  after: SubscriptionCycleSnapshot;
  grossChange: number;
  approvedAmountChange: number;
  refundChange: number | null;
  subscriptionContext: Pick<Subscription, "state" | "cancellation">;
  bankConfirmed: false;
  realizedSavings: null;
  recoveredCash: null;
  notice: string;
}
export interface SubscriptionCycleComparison extends SubscriptionCycleComparisonPreview {
  id: string;
  version: number;
  state: "proposed" | "accepted" | "rejected" | "withdrawn" | "revoked";
  reason: string;
  /** Absent only on historical records; never inferred from display names. */
  preparedBy?: string;
  preparedAt: string;
  reviewedAt: string | null;
  current: boolean;
  issues: string[];
  history: AuditEvent[];
}

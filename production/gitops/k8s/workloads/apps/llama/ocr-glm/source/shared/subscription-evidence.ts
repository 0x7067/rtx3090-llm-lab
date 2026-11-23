import { z } from "zod";
import { payableDate } from "./payables.js";
import type { AccountingSource } from "./accounting.js";
import type { AuditEvent, ExpenseAllocation } from "./types.js";
import { directoryKey } from "./company.js";
import type { SubscriptionRevision } from "./subscriptions.js";
import { addDays, anchoredMonth, recurrenceDates } from "./subscriptions.js";
import type { CardRefund, CardTransaction } from "./cards.js";

export const subscriptionSourceKind = z.enum([
  "expense",
  "payable",
  "card-charge",
  "card-refund",
]);
export const subscriptionSourceTiming = z.enum([
  "in-cycle",
  "adjacent-cycle-adjustment",
  "timing-dispute",
]);
export function sourceFitsCycle(
  sourceDate: string,
  billingDate: string,
  terms: SubscriptionRevision,
  timing: z.infer<typeof subscriptionSourceTiming>,
) {
  const anchor = terms.fields.billingStart;
  const cadence = terms.fields.billingCadence;
  const next = recurrenceDates(
    anchor,
    cadence,
    addDays(billingDate, 1),
    anchoredMonth(billingDate, 13),
  )[0];
  if (!next) return false;
  const inCycle = sourceDate >= billingDate && sourceDate < next;
  if (timing === "in-cycle") return inCycle;
  const previous = recurrenceDates(
    anchor,
    cadence,
    addDays(billingDate, -370),
    addDays(billingDate, -1),
  ).at(-1);
  const afterNext = recurrenceDates(
    anchor,
    cadence,
    addDays(next, 1),
    anchoredMonth(next, 13),
  )[0];
  const adjacent =
    !inCycle &&
    ((!!previous && sourceDate >= previous && sourceDate < billingDate) ||
      (!!afterNext && sourceDate >= next && sourceDate < afterNext));
  if (timing === "adjacent-cycle-adjustment") return adjacent;
  return (
    !inCycle &&
    !adjacent &&
    sourceDate >= addDays(billingDate, -366) &&
    sourceDate <= addDays(billingDate, 366)
  );
}
export interface SubscriptionCardSource extends Omit<
  AccountingSource,
  "kind" | "type" | "costCenter" | "approvalId"
> {
  kind: "card-charge" | "card-refund";
  type: "simulated-card-charge" | "simulated-card-refund";
  costCenter: null;
  approvalId: null;
  evidenceMode: "simulation";
  card: {
    transactionId: string;
    cardId: string;
    owner: CardTransaction["owner"];
    ownerId: string | null;
    origin: CardTransaction["origin"];
    sourceRef: string;
    transactionDate: string;
    transactionAmount: number;
    receiptCostCenter: string | null;
    receiptAllocations?: ExpenseAllocation[];
    receiptAllocationScope?: {
      covered: number;
      unassigned: number;
      stale: number;
    };
    refund?: CardRefund;
  };
}
export type SubscriptionSource =
  (AccountingSource & { cardTransactionId?: string }) | SubscriptionCardSource;

export function subscriptionSourceAllocations(source: SubscriptionSource) {
  return "card" in source ? source.card.receiptAllocations : source.allocations;
}

// Unknown card centers stay unknown. A known split never becomes unrestricted.
export function subscriptionCenterAmount(
  source: SubscriptionSource,
  center: string,
): number | null {
  const allocations = subscriptionSourceAllocations(source);
  if ("card" in source && source.card.receiptAllocationScope)
    return Math.min(
      source.amount,
      (allocations ?? [])
        .filter((row) => directoryKey(row.costCenter) === directoryKey(center))
        .reduce((n, row) => n + row.amount, 0),
    );
  if (allocations?.length)
    return Math.min(
      source.amount,
      allocations
        .filter((row) => directoryKey(row.costCenter) === directoryKey(center))
        .reduce((n, row) => n + row.amount, 0),
    );
  const known =
    "card" in source ? source.card.receiptCostCenter : source.costCenter;
  return known === null
    ? null
    : directoryKey(known) === directoryKey(center)
      ? source.amount
      : 0;
}

export const subscriptionObservationSchema = z
  .object({
    subscriptionId: z.string().uuid(),
    subscriptionVersion: z.number().int().positive(),
    billingDate: payableDate,
    sourceKind: subscriptionSourceKind,
    sourceId: z.string().uuid(),
    sourceToken: z.string().regex(/^[a-f0-9]{64}$/),
    allocationAmount: z.number().int().positive().optional(),
    allocationBasis: z.string().trim().min(10).max(500).optional(),
    sourceTiming: subscriptionSourceTiming.optional(),
    timingExplanation: z.string().trim().min(10).max(500).optional(),
    adjustsObservationId: z.string().uuid().optional(),
  })
  .strict()
  .refine(
    (v) => (v.sourceKind === "card-refund") === !!v.adjustsObservationId,
    "Vincule o estorno à observação aceita da cobrança original; outras fontes não aceitam esse campo.",
  )
  .refine(
    (v) => v.sourceKind !== "card-refund" || !v.sourceTiming,
    "O estorno usa o ciclo da cobrança original e não aceita ajuste de data separado.",
  )
  .refine(
    (v) =>
      (v.sourceTiming === "adjacent-cycle-adjustment" ||
        v.sourceTiming === "timing-dispute") === !!v.timingExplanation,
    "Explique separadamente por que a fonte pertence a outro ciclo.",
  );
export type SubscriptionObservationFields = z.infer<
  typeof subscriptionObservationSchema
>;
export interface SubscriptionObservation extends SubscriptionObservationFields {
  id: string;
  version: number;
  state: "proposed" | "accepted" | "rejected" | "released";
  source: SubscriptionSource;
  terms: SubscriptionRevision;
  /** Absent only on historical records; never inferred from display names. */
  preparedBy?: string;
  preparedAt: string;
  reviewedAt: string | null;
  reason: string;
  history: AuditEvent[];
  current: boolean;
  issues: string[];
  identityCorrectionMatchIds?: string[];
  bankConfirmed: false;
}

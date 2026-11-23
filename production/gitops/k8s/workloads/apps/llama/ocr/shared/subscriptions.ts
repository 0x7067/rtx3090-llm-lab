import { z } from "zod";
import { payableDate } from "./payables.js";
import { directoryId } from "./company.js";
import type { AuditEvent } from "./types.js";

// Subscription calendar decisions use the same business day as the pt-BR UI.
// Event timestamps remain UTC instants for audit history.
export const subscriptionBusinessDate = (instant: Date | string) =>
  new Date(instant).toLocaleDateString("en-CA", {
    timeZone: "America/Sao_Paulo",
  });

export const subscriptionReason = z.string().trim().min(5).max(1500);
export const subscriptionVersion = z.number().int().positive();
const cadence = z.union([z.literal(1), z.literal(3), z.literal(12)]);
export const subscriptionFieldsSchema = z
  .object({
    vendorId: z.string().uuid(),
    service: z.string().trim().min(3).max(160),
    serviceGroup: z.string().trim().min(3).max(100),
    purpose: z.string().trim().min(5).max(1500),
    ownerId: directoryId,
    centerId: directoryId,
    amount: z.number().int().positive().max(100_000_000),
    billingStart: payableDate,
    billingCadence: cadence,
    renewalDate: payableDate,
    renewalCadence: cadence,
    autoRenew: z.boolean(),
    noticeDays: z.number().int().min(0).max(365),
    contractReference: z.string().trim().min(5).max(300),
  })
  .strict()
  .refine(
    (v) => v.renewalDate > v.billingStart,
    "A renovação deve suceder o início da cobrança.",
  );
export type SubscriptionFields = z.infer<typeof subscriptionFieldsSchema>;
export interface SubscriptionContext {
  vendor: {
    id: string;
    name: string;
    taxId: string;
    version: number;
    active: boolean;
    onHold: boolean;
  };
  owner: { id: string; name: string; version: number; active: boolean };
  center: {
    id: string;
    name: string;
    code: string;
    version: number;
    active: boolean;
  };
}
export interface SubscriptionProposal {
  fields: SubscriptionFields;
  effectiveFrom: string;
  context: SubscriptionContext;
  contextToken: string;
  preparedAt: string;
  preparedBy: string;
}
export interface SubscriptionRevision extends SubscriptionProposal {
  number: number;
  approvedAt: string;
  approvedBy: string;
  reason: string;
}
export type SubscriptionState =
  | "draft"
  | "active"
  | "cancellation-requested"
  | "cancellation-recorded"
  | "ended"
  | "discarded";
export interface Subscription {
  id: string;
  version: number;
  state: SubscriptionState;
  proposal: SubscriptionProposal | null;
  revisions: SubscriptionRevision[];
  cancellation: null | {
    requestedAt: string;
    requestedBy: string;
    requestReason: string;
    effectiveEnd: string | null;
    reference: string | null;
    recordedAt: string | null;
    recordedBy: string | null;
    supplierConfirmed: false;
  };
  createdAt: string;
  updatedAt: string;
  mode: "local-manual";
  history: AuditEvent[];
  contextIssues: string[];
}
export const subscriptionWindowSchema = z
  .object({ from: payableDate, to: payableDate })
  .strict()
  .refine(
    (v) => v.to >= v.from && daysBetween(v.from, v.to) <= 366,
    "Informe um intervalo de até 366 dias.",
  );
export const cancellationSchema = z
  .object({
    effectiveEnd: payableDate,
    reference: z.string().trim().min(5).max(300),
  })
  .strict();
export const overlapReviewSchema = z
  .object({
    leftId: z.string().uuid(),
    rightId: z.string().uuid(),
    basisToken: z.string().regex(/^[a-f0-9]{64}$/),
    version: z.number().int().min(0),
    resolution: z.enum(["distinct-use", "consolidation-proposed", "dismissed"]),
    ownerId: directoryId,
    dueDate: payableDate.nullable(),
  })
  .strict()
  .refine(
    (v) => v.leftId !== v.rightId,
    "Selecione duas assinaturas diferentes.",
  )
  .refine(
    (v) => v.resolution !== "consolidation-proposed" || v.dueDate !== null,
    "Informe um prazo para investigar a consolidação.",
  );
export type OverlapReviewFields = z.infer<typeof overlapReviewSchema>;
export interface SubscriptionOverlap extends OverlapReviewFields {
  id: string;
  reason: string;
  reviewedAt: string;
  reviewedBy: string;
  owner: { id: string; name: string; version: number };
  history: AuditEvent[];
}
export function daysBetween(a: string, b: string) {
  return Math.round(
    (Date.parse(b + "T12:00:00Z") - Date.parse(a + "T12:00:00Z")) / 86_400_000,
  );
}
export function addDays(date: string, days: number) {
  return new Date(Date.parse(date + "T12:00:00Z") + days * 86_400_000)
    .toISOString()
    .slice(0, 10);
}
// Always derive from the original anchor. Feb 28 must not turn Mar 31 into Mar 28.
export function anchoredMonth(anchor: string, offset: number) {
  const [year, month, day] = anchor.split("-").map(Number);
  const first = new Date(Date.UTC(year, month - 1 + offset, 1, 12));
  const lastDay = new Date(
    Date.UTC(first.getUTCFullYear(), first.getUTCMonth() + 1, 0, 12),
  ).getUTCDate();
  first.setUTCDate(Math.min(day, lastDay));
  return first.toISOString().slice(0, 10);
}
export function recurrenceDates(
  anchor: string,
  interval: 1 | 3 | 12,
  from: string,
  to: string,
): string[] {
  const monthIndex = (d: string) =>
    Number(d.slice(0, 4)) * 12 + Number(d.slice(5, 7)) - 1;
  let n = Math.max(
    0,
    Math.floor((monthIndex(from) - monthIndex(anchor)) / interval) - 1,
  );
  const dates: string[] = [];
  for (;;) {
    const date = anchoredMonth(anchor, n++ * interval);
    if (date > to) return dates;
    if (date >= from) dates.push(date);
  }
}
export function effectiveRevision(
  s: Pick<Subscription, "revisions">,
  on: string,
) {
  return [...s.revisions].reverse().find((r) => r.effectiveFrom <= on);
}
export function nextRenewal(fields: SubscriptionFields, on: string) {
  if (!fields.autoRenew || fields.renewalDate >= on) return fields.renewalDate;
  return recurrenceDates(
    fields.renewalDate,
    fields.renewalCadence,
    on,
    anchoredMonth(on, 13),
  )[0];
}
export interface SubscriptionForecastLine {
  subscriptionId: string;
  revision: number;
  service: string;
  date: string;
  amount: number;
  vendorId: string;
  ownerId: string;
  centerId: string;
  estimated: true;
}
export function projectSubscription(
  s: Subscription,
  from: string,
  to: string,
): SubscriptionForecastLine[] {
  subscriptionWindowSchema.parse({ from, to });
  if (["draft", "discarded"].includes(s.state)) return [];
  const lines: SubscriptionForecastLine[] = [];
  for (let i = 0; i < s.revisions.length; i++) {
    const r = s.revisions[i],
      f = r.fields;
    for (const date of recurrenceDates(
      f.billingStart,
      f.billingCadence,
      from,
      to,
    )) {
      if (
        date < r.effectiveFrom ||
        (s.revisions[i + 1] && date >= s.revisions[i + 1].effectiveFrom)
      )
        continue;
      if (s.cancellation?.effectiveEnd && date >= s.cancellation.effectiveEnd)
        continue;
      if (!f.autoRenew && date >= f.renewalDate) continue;
      lines.push({
        subscriptionId: s.id,
        revision: r.number,
        service: f.service,
        date,
        amount: f.amount,
        vendorId: f.vendorId,
        ownerId: f.ownerId,
        centerId: f.centerId,
        estimated: true,
      });
    }
  }
  return lines.sort((a, b) => a.date.localeCompare(b.date));
}

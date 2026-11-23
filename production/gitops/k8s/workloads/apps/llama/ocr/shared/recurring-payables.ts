import { z } from "zod";
import { payableDate } from "./payables.js";
import { directoryId } from "./company.js";
import { daysBetween, recurrenceDates } from "./subscriptions.js";
import type { ActorContext } from "./actor.js";
import type { SubscriptionContext } from "./subscriptions.js";
import type { Document } from "./types.js";

const uuid = z.string().uuid();
const version = z.number().int().min(0);
const reason = z.string().trim().min(5).max(1500);
const token = z.string().regex(/^[a-f0-9]{64}$/);
const command = { version, reason, requestKey: uuid };
export const recurringFieldsSchema = z
  .object({
    name: z.string().trim().min(3).max(160),
    vendorId: uuid,
    ownerId: directoryId,
    centerId: directoryId,
    amount: z.number().int().positive().max(100_000_000),
    firstDue: payableDate,
    lastDue: payableDate.nullable(),
    cadence: z.union([z.literal(1), z.literal(3), z.literal(12)]),
    reference: z.string().trim().min(5).max(300),
    basis: reason,
  })
  .strict()
  .refine(
    (f) => !f.lastDue || f.lastDue >= f.firstDue,
    "O fim não pode anteceder o primeiro vencimento.",
  );
export const recurringProposalSchema = z
  .object({
    ...command,
    id: uuid.optional(),
    fields: recurringFieldsSchema,
    effectiveFrom: payableDate,
  })
  .strict();
export const recurringDecisionSchema = z
  .object({
    ...command,
    action: z.enum(["accept", "reject", "withdraw"]),
  })
  .strict();
export const recurringStopSchema = z
  .object({ ...command, effectiveEnd: payableDate })
  .strict();
export const recurringMatchSchema = z
  .object({
    ...command,
    dueDate: payableDate,
    termsToken: token,
    billId: uuid,
    sourceToken: token,
    basis: reason,
  })
  .strict();
export const recurringMatchDecisionSchema = z
  .object({
    ...command,
    action: z.enum(["accept", "reject", "withdraw", "release"]),
  })
  .strict();
export const recurringWindowSchema = z
  .object({ from: payableDate, to: payableDate })
  .strict()
  .refine(
    (w) => w.to >= w.from && daysBetween(w.from, w.to) <= 366,
    "Escolha um intervalo de até 366 dias.",
  );
export type RecurringFields = z.infer<typeof recurringFieldsSchema>;
export interface RecurringProposal {
  fields: RecurringFields;
  effectiveFrom: string;
  context: SubscriptionContext;
  contextToken: string;
  preparers: ActorContext[];
  preparedAt: string;
}
export interface RecurringRevision extends RecurringProposal {
  number: number;
  approvedBy: ActorContext;
  approvedAt: string;
  reason: string;
}
export interface RecurringSchedule {
  id: string;
  name: string;
  version: number;
  proposal: RecurringProposal | null;
  revisions: RecurringRevision[];
  stop: {
    effectiveEnd: string;
    actor: ActorContext;
    reason: string;
    createdAt: string;
  } | null;
  createdAt: string;
  updatedAt: string;
}
export interface RecurringOccurrence {
  scheduleId: string;
  dueDate: string;
  terms: RecurringRevision;
  termsToken: string;
  expectedAmount: number;
}
export interface RecurringBillSource {
  billId: string;
  version: number;
  vendorId: string;
  vendorName: string;
  invoiceNumber: string;
  issueDate: string;
  dueDate: string;
  costCenter: string;
  amount: number;
  approvalId: string;
  documents: Document[];
  token: string;
  claimKeys: string[];
  bankConfirmed: false;
}
export interface RecurringMatch {
  id: string;
  scheduleId: string;
  version: number;
  state: "proposed" | "accepted" | "rejected" | "withdrawn" | "released";
  occurrence: RecurringOccurrence;
  source: RecurringBillSource;
  basis: string;
  preparedBy: ActorContext;
  preparedAt: string;
  decision?: { actor: ActorContext; reason: string; createdAt: string };
  release?: { actor: ActorContext; reason: string; createdAt: string };
}
export interface RecurringHistory {
  entity: string;
  before: RecurringSchedule | RecurringMatch | null;
  after: RecurringSchedule | RecurringMatch;
  actor: ActorContext;
  reason: string;
  createdAt: string;
}
export interface RecurringWorkspace {
  asOf: string;
  window: { from: string; to: string };
  schedules: (RecurringSchedule & { issues: string[] })[];
  occurrences: (RecurringOccurrence & {
    matchId: string | null;
    status: "expected" | "overdue" | "pending" | "matched" | "stale";
    actualAmount: number | null;
    variance: number | null;
  })[];
  matches: (RecurringMatch & { current: boolean; issues: string[] })[];
  candidates: (
    | { eligible: true; source: RecurringBillSource; reservedBy: string[] }
    | { eligible: false; billId: string; reason: string }
  )[];
  vendors: {
    id: string;
    name: string;
    version: number;
    active: boolean;
    onHold: boolean;
  }[];
  owners: { id: string; name: string }[];
  centers: { id: string; name: string }[];
  history: RecurringHistory[];
  bankConfirmed: false;
}
export function recurringRevisionAt(schedule: RecurringSchedule, date: string) {
  if (schedule.stop && date > schedule.stop.effectiveEnd) return undefined;
  return [...schedule.revisions].reverse().find((r) => r.effectiveFrom <= date);
}
export function recurringDates(
  schedule: RecurringSchedule,
  from: string,
  to: string,
) {
  recurringWindowSchema.parse({ from, to });
  const dates = new Set<string>();
  for (const revision of schedule.revisions) {
    for (const date of recurrenceDates(
      revision.fields.firstDue,
      revision.fields.cadence,
      from,
      to,
    )) {
      if (
        (!revision.fields.lastDue || date <= revision.fields.lastDue) &&
        recurringRevisionAt(schedule, date)?.number === revision.number
      )
        dates.add(date);
    }
  }
  return [...dates].sort();
}

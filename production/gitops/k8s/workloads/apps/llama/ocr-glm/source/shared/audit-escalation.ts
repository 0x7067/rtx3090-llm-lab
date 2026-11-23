import { z } from "zod";
import { directoryId } from "./company.js";
import { payableDate } from "./payables.js";
import { daysBetween } from "./subscriptions.js";
import type { ActorContext } from "./actor.js";
import type { AuditCase } from "./audit-cases.js";

export const auditEscalationSchema = z
  .object({
    version: z.number().int().positive(),
    ownerId: directoryId,
    ownerVersion: z.number().int().positive(),
    dueDate: payableDate,
    asOf: payableDate,
    reason: z.string().trim().min(5).max(1500),
    requestKey: z.string().uuid(),
  })
  .strict();

export interface AuditDeadline {
  asOf: string;
  state: "open" | "due_today" | "overdue" | "closed";
  overdueDays: number;
}
export function auditDeadline(
  record: Pick<AuditCase, "state" | "dueDate">,
  asOf: string,
): AuditDeadline {
  const state =
    record.state !== "open"
      ? "closed"
      : record.dueDate < asOf
        ? "overdue"
        : record.dueDate === asOf
          ? "due_today"
          : "open";
  return {
    asOf,
    state,
    overdueDays: state === "overdue" ? daysBetween(record.dueDate, asOf) : 0,
  };
}
export interface AuditEscalation {
  id: string;
  caseId: string;
  before: {
    version: number;
    owner: { id: string; name: string };
    dueDate: string;
  };
  after: {
    version: number;
    owner: { id: string; name: string };
    dueDate: string;
  };
  asOf: string;
  overdueDays: number;
  actor: ActorContext;
  reason: string;
  sourceStale: boolean;
  createdAt: string;
  notificationSent: false;
}
export interface AuditEscalationContext {
  record: AuditCase;
  asOf: string;
  owners: { id: string; name: string; version: number }[];
}

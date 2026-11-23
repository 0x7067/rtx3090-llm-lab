import { z } from "zod";
import { directoryId } from "./company.js";
import { payableDate } from "./payables.js";
import type { AuditFinding } from "./intelligence.js";
import type { Status, ExpenseAllocation } from "./types.js";
import type { MileageDeclaration } from "./mileage.js";
import type { AuditDeadline, AuditEscalation } from "./audit-escalation.js";

const reason = z.string().trim().min(5).max(1500);
const requestKey = z.string().uuid();
const version = z.number().int().positive();

export const auditCaseCreateSchema = z
  .object({
    ownerId: directoryId,
    dueDate: payableDate,
    suspectedAmount: z.number().int().positive().max(100_000_000).nullable(),
    reason,
    requestKey,
  })
  .strict();
export const auditCaseEditSchema = z
  .object({
    ownerId: directoryId,
    dueDate: payableDate,
    version,
    reason,
    requestKey,
  })
  .strict();
export const auditCaseCommentSchema = z
  .object({ version, text: reason, requestKey })
  .strict();
export const auditCaseResolveSchema = z
  .object({
    version,
    state: z.enum(["confirmed", "dismissed", "superseded"]),
    reason,
    requestKey,
  })
  .strict();
export const auditCaseLinkSchema = z
  .object({
    expenseId: z.string().trim().min(1).max(200),
    version,
    reason,
    requestKey,
  })
  .strict();
export const auditCaseUnlinkSchema = z
  .object({ version, reason, requestKey })
  .strict();
export const auditCaseSourceKindSchema = z.enum([
  "card_transaction",
  "payable_bill",
  "statement_entry",
]);
export const auditCaseSourceLinkSchema = z
  .object({
    kind: auditCaseSourceKindSchema,
    sourceId: z.string().trim().min(1).max(200),
    version,
    reason,
    requestKey,
  })
  .strict();
export const auditCaseSourceReleaseSchema = auditCaseUnlinkSchema;

export type AuditCaseCreate = z.infer<typeof auditCaseCreateSchema>;
export type AuditCaseEdit = z.infer<typeof auditCaseEditSchema>;
export type AuditCaseComment = z.infer<typeof auditCaseCommentSchema>;
export type AuditCaseResolve = z.infer<typeof auditCaseResolveSchema>;
export type AuditCaseLink = z.infer<typeof auditCaseLinkSchema>;
export type AuditCaseUnlink = z.infer<typeof auditCaseUnlinkSchema>;
export type AuditCaseSourceKind = z.infer<typeof auditCaseSourceKindSchema>;
export type AuditCaseSourceLink = z.infer<typeof auditCaseSourceLinkSchema>;
export type AuditCaseSourceRelease = z.infer<
  typeof auditCaseSourceReleaseSchema
>;
export type AuditCaseState = "open" | AuditCaseResolve["state"];
export interface AuditCaseSourceSnapshot {
  kind: AuditCaseSourceKind;
  id: string;
  label: string;
  amount: number;
  date: string;
  state: string;
  version: number;
  provenance: string;
  details: { label: string; value: string }[];
  originals: { label: string; url: string }[];
  fingerprint: string;
}
export interface AuditCaseRelatedSource {
  id: string;
  source: AuditCaseSourceSnapshot;
  active: boolean;
  sourceStale: boolean;
  createdAt: string;
  removedAt: string | null;
}
/** Manager-only navigation context; amounts are deliberately not aggregated. */
export interface PayableInvestigation {
  id: string;
  title: string;
  expense: Pick<AuditCase["source"]["expense"], "merchant" | "date">;
  state: AuditCaseState;
  owner: { id: string; name: string };
  dueDate: string;
  sourceStale: boolean;
  links: Pick<AuditCaseRelatedSource, "id" | "active" | "sourceStale" | "createdAt" | "removedAt">[];
}
export interface AuditCaseRelatedExpense {
  id: string;
  expense: {
    id: string;
    merchant: string;
    amount: number;
    date: string;
    status: Status;
    version: number;
    allocations?: ExpenseAllocation[];
    mileage?: MileageDeclaration;
  };
  active: boolean;
  sourceStale: boolean;
  createdAt: string;
  removedAt: string | null;
}
export interface AuditCaseEvent {
  sequence: number;
  action:
    | "created"
    | "assigned"
    | "commented"
    | "resolved"
    | "linked"
    | "unlinked"
    | "linked_source"
    | "unlinked_source"
    | "escalated";
  actor: string;
  reason: string;
  owner: { id: string; name: string };
  dueDate: string;
  state: AuditCaseState;
  createdAt: string;
}
export interface AuditCase {
  id: string;
  jobId: string;
  findingId: string;
  expenseId: string;
  source: {
    expense: {
      id: string;
      merchant: string;
      amount: number;
      date: string;
      allocations?: ExpenseAllocation[];
      mileage?: MileageDeclaration;
    };
    finding: AuditFinding;
  };
  sourceStale: boolean;
  owner: { id: string; name: string };
  dueDate: string;
  state: AuditCaseState;
  suspectedAmount: number | null;
  preventedAmount: null;
  recoveredAmount: null;
  relatedExpenses: AuditCaseRelatedExpense[];
  relatedSources: AuditCaseRelatedSource[];
  version: number;
  createdAt: string;
  updatedAt: string;
  history: AuditCaseEvent[];
  deadline?: AuditDeadline;
  escalations?: AuditEscalation[];
}

import { z } from "zod";
import {
  accountingDate,
  accountingMonth,
  accountingReason,
  accountingVersion,
} from "./accounting.js";
import { closingTasks, type ClosingTask } from "./accounting-close.js";
export const closingWorkDate = (date: Date) =>
  date.toLocaleDateString("en-CA", { timeZone: "America/Sao_Paulo" });
export const closingMemberId = z.string().regex(/^[A-Za-z0-9-]{1,80}$/);
export const closingActorSchema = z
  .object({
    subjectId: z.string().uuid(),
    companyId: z.string().uuid(),
    memberId: closingMemberId,
    role: z.enum(["employee", "finance", "manager"]),
    displayName: z.string().min(2).max(100),
    authentication: z.enum(["demo", "password"]),
  })
  .strict();
const financeActor = closingActorSchema.extend({ role: z.literal("finance") });
const managerActor = closingActorSchema.extend({ role: z.literal("manager") });
export const closingResponsibilitySchema = z
  .object({
    assignee: financeActor,
    dueDate: accountingDate,
    assignedBy: managerActor,
    assignedAt: z.string().datetime(),
    reason: accountingReason,
  })
  .strict();
export type ClosingResponsibility = z.infer<typeof closingResponsibilitySchema>;
const assignment = z
  .object({
    task: z.enum(closingTasks),
    memberId: closingMemberId,
    dueDate: accountingDate,
  })
  .strict();
export const closingResponsibilityCommandSchema = z
  .object({
    action: z.enum(["enable", "assign"]),
    version: accountingVersion,
    periodVersion: accountingVersion,
    contextToken: z.string().regex(/^[a-f0-9]{64}$/),
    assignments: z.array(assignment).min(1).max(3),
    reason: accountingReason,
    requestKey: z.string().uuid(),
    understood: z.literal(true),
  })
  .strict()
  .superRefine((v, ctx) => {
    if (
      new Set(v.assignments.map((a) => a.task)).size !== v.assignments.length ||
      (v.action === "enable"
        ? v.assignments.length !== 3
        : v.assignments.length !== 1)
    )
      ctx.addIssue({
        code: "custom",
        path: ["assignments"],
        message:
          "A ativação atribui as três conferências distintas; a reatribuição altera uma por vez.",
      });
  });
export type ClosingResponsibilityCommand = z.infer<
  typeof closingResponsibilityCommandSchema
>;
export const closingResponsibilityReceiptSchema = z
  .object({
    requestKey: z.string().uuid(),
    action: z.enum(["enable", "assign"]),
    month: accountingMonth,
    periodVersion: accountingVersion,
    checklistVersion: accountingVersion,
    contextToken: z.string().regex(/^[a-f0-9]{64}$/),
    basisToken: z.string().regex(/^[a-f0-9]{64}$/),
    actor: managerActor,
    recordedAt: z.string().datetime(),
    reason: accountingReason,
    assignments: z
      .array(
        z
          .object({
            task: z.enum(closingTasks),
            responsibility: closingResponsibilitySchema,
          })
          .strict(),
      )
      .length(3),
    scope: z.literal("local-closing-responsibilities"),
    notificationsSent: z.literal(false),
    financialEntriesCreated: z.literal(false),
  })
  .strict()
  .refine((v) => new Set(v.assignments.map((a) => a.task)).size === 3, {
    message: "Conferências repetidas no comprovante.",
  });
export type ClosingResponsibilityReceipt = z.infer<
  typeof closingResponsibilityReceiptSchema
>;
export interface ClosingResponsibilityStatus {
  responsibilityIssue: string | null;
  deadlineState: "unassigned" | "upcoming" | "due" | "overdue" | "complete";
  completedLate: boolean;
}
export type ClosingResponsibleTask = ClosingTask & {
  state: "pending" | "current" | "stale";
} & ClosingResponsibilityStatus;
export interface ClosingResponsibilityReport {
  month: string;
  asOf: string;
  periodVersion: number;
  periodState: "open" | "locked";
  checklistEnabled: boolean;
  enabled: boolean;
  version: number | null;
  basisToken: string;
  contextToken: string;
  viewer: z.infer<typeof closingActorSchema>;
  choices: z.infer<typeof financeActor>[];
  tasks: ClosingResponsibleTask[];
  scope: "local-closing-responsibilities";
  notificationsSent: false;
}
export const closingWorklistQuerySchema = z
  .object({
    cursor: accountingMonth.optional(),
    limit: z.coerce.number().int().min(1).max(24).default(12),
    scope: z.enum(["mine", "team"]).default("mine"),
  })
  .strict();
export type ClosingWorklistQuery = z.infer<typeof closingWorklistQuerySchema>;
export interface ClosingWorklist {
  asOf: string;
  viewer: z.infer<typeof closingActorSchema>;
  scope: "mine" | "team";
  nextCursor: string | null;
  items: (ClosingResponsibleTask & {
    month: string;
    checklistVersion: number;
    periodVersion: number;
    canReview: boolean;
  })[];
  notificationsSent: false;
  financialEntriesCreated: false;
}
export const closingResponsibleTaskSchema = z
  .object({
    id: z.enum(closingTasks),
    complete: z.boolean(),
    basisToken: z
      .string()
      .regex(/^[a-f0-9]{64}$/)
      .nullable(),
    reviewedBy: closingMemberId.nullable(),
    reviewedAt: z.string().datetime().nullable(),
    reason: accountingReason.nullable(),
    responsibility: closingResponsibilitySchema.optional(),
    reviewedActor: closingActorSchema.nullable().optional(),
    state: z.enum(["pending", "current", "stale"]),
    responsibilityIssue: z.string().min(1).max(1000).nullable(),
    deadlineState: z.enum([
      "unassigned",
      "upcoming",
      "due",
      "overdue",
      "complete",
    ]),
    completedLate: z.boolean(),
  })
  .strict();
export const closingResponsibilityReportSchema = z
  .object({
    month: accountingMonth,
    asOf: accountingDate,
    periodVersion: accountingVersion,
    periodState: z.enum(["open", "locked"]),
    checklistEnabled: z.boolean(),
    enabled: z.boolean(),
    version: accountingVersion.nullable(),
    basisToken: z.string().regex(/^[a-f0-9]{64}$/),
    contextToken: z.string().regex(/^[a-f0-9]{64}$/),
    viewer: closingActorSchema,
    choices: z.array(financeActor),
    tasks: z.array(closingResponsibleTaskSchema).max(3),
    scope: z.literal("local-closing-responsibilities"),
    notificationsSent: z.literal(false),
  })
  .strict()
  .superRefine((r, ctx) => {
    if (
      r.viewer.role === "employee" ||
      (r.checklistEnabled
        ? r.version === null || r.tasks.length !== 3
        : r.version !== null || r.tasks.length !== 0) ||
      (r.enabled && !r.checklistEnabled) ||
      new Set(r.tasks.map((t) => t.id)).size !== r.tasks.length ||
      new Set(r.choices.map((m) => m.memberId)).size !== r.choices.length ||
      r.choices.some((m) => m.companyId !== r.viewer.companyId) ||
      r.tasks.some(
        (t) =>
          (!r.enabled &&
            (t.responsibility !== undefined ||
              t.deadlineState !== "unassigned")) ||
          (r.enabled && !t.responsibility && !t.responsibilityIssue),
      )
    )
      ctx.addIssue({
        code: "custom",
        message: "Composição de responsáveis inconsistente.",
      });
    if (
      r.enabled &&
      r.tasks.some(
        (t) =>
          !consistentTask(t, r.asOf) ||
          (t.state === "current" && t.basisToken !== r.basisToken) ||
          (t.responsibility &&
            !t.responsibilityIssue &&
            (t.responsibility.assignee.companyId !== r.viewer.companyId ||
              t.responsibility.dueDate < r.month + "-01")),
      )
    )
      ctx.addIssue({
        code: "custom",
        message: "Estado ou prazo da conferência inconsistente.",
      });
  });
export const closingWorklistSchema = z
  .object({
    asOf: accountingDate,
    viewer: closingActorSchema,
    scope: z.enum(["mine", "team"]),
    nextCursor: accountingMonth.nullable(),
    items: z
      .array(
        closingResponsibleTaskSchema
          .extend({
            month: accountingMonth,
            checklistVersion: accountingVersion,
            periodVersion: accountingVersion,
            canReview: z.boolean(),
          })
          .strict(),
      )
      .max(72),
    notificationsSent: z.literal(false),
    financialEntriesCreated: z.literal(false),
  })
  .strict()
  .superRefine((r, ctx) => {
    const same = (a: typeof r.viewer, b: typeof r.viewer) =>
      a.memberId === b.memberId &&
      a.subjectId === b.subjectId &&
      a.companyId === b.companyId;
    if (
      r.viewer.role === "employee" ||
      new Set(r.items.map((t) => t.month + ":" + t.id)).size !==
        r.items.length ||
      r.items.some(
        (t) =>
          (!t.responsibility && !t.responsibilityIssue) ||
          (r.scope === "mine" &&
            (!t.responsibility ||
              !same(r.viewer, t.responsibility.assignee))) ||
          t.canReview !==
            (r.viewer.role === "finance" &&
              !t.responsibilityIssue &&
              !!t.responsibility &&
              same(r.viewer, t.responsibility.assignee)),
      )
    )
      ctx.addIssue({
        code: "custom",
        message: "Fila de responsabilidades inconsistente.",
      });
    if (
      r.items.some(
        (t) =>
          !consistentTask(t, r.asOf) ||
          (t.responsibility &&
            !t.responsibilityIssue &&
            t.responsibility.assignee.companyId !== r.viewer.companyId),
      )
    )
      ctx.addIssue({
        code: "custom",
        message: "Estado de prazo inválido na fila.",
      });
  });
function consistentTask(
  t: z.infer<typeof closingResponsibleTaskSchema>,
  asOf: string,
) {
  if (t.complete === (t.state === "pending")) return false;
  const r = t.responsibility;
  if (!r)
    return (
      !!t.responsibilityIssue &&
      t.deadlineState === "unassigned" &&
      !t.completedLate &&
      t.state !== "current"
    );
  if (
    t.state === "current" &&
    (!t.reviewedActor ||
      t.responsibilityIssue ||
      t.reviewedActor.role !== "finance" ||
      t.reviewedBy !== r.assignee.memberId ||
      !t.reviewedAt ||
      !t.reason ||
      t.reviewedActor.subjectId !== r.assignee.subjectId ||
      t.reviewedActor.companyId !== r.assignee.companyId ||
      t.reviewedActor.memberId !== r.assignee.memberId)
  )
    return false;
  if (!t.responsibilityIssue && r.assignee.companyId !== r.assignedBy.companyId)
    return false;
  const expected =
    t.state === "current" && !t.responsibilityIssue
      ? "complete"
      : r.dueDate < asOf
        ? "overdue"
        : r.dueDate === asOf
          ? "due"
          : "upcoming";
  return (
    t.deadlineState === expected &&
    t.completedLate ===
      !!(
        t.state === "current" &&
        !t.responsibilityIssue &&
        t.reviewedAt &&
        closingWorkDate(new Date(t.reviewedAt)) > r.dueDate
      )
  );
}

import { z } from "zod";
import { expenseFieldsSchema, expenseIdSchema } from "./expense-fields.js";
import {
  mileageExpenseFieldsSchema,
  mileageRateFieldsSchema,
} from "./mileage.js";
import {
  correctionGrantSchema,
  correctionIdentitySchema,
} from "./expense-correction-permissions.js";
import { categories } from "./types.js";
import type { ApprovalRun } from "./approvals.js";

export const correctionCandidateSchema = z.discriminatedUnion("kind", [
  z
    .object({ kind: z.literal("ordinary"), fields: expenseFieldsSchema })
    .strict(),
  z
    .object({
      kind: z.literal("mileage"),
      fields: mileageExpenseFieldsSchema,
      rateVersion: z.number().int().positive().nullable(),
    })
    .strict(),
]);
export const expenseCorrectionCommandSchema = z
  .object({
    version: z.number().int().positive(),
    grantId: z.string().uuid(),
    grantVersion: z.number().int().positive(),
    contextToken: z.string().regex(/^[a-f0-9]{64}$/),
    candidate: correctionCandidateSchema,
    reason: z.string().trim().min(10).max(1500),
    requestKey: z.string().uuid(),
    understood: z.literal(true),
  })
  .strict();
export type ExpenseCorrectionCommand = z.infer<
  typeof expenseCorrectionCommandSchema
>;
export const expenseCorrectionPreviewSchema =
  expenseCorrectionCommandSchema.omit({
    reason: true,
    requestKey: true,
    understood: true,
  });
export const correctionMileageRateSchema = mileageRateFieldsSchema
  .extend({
    version: z.number().int().positive(),
    actor: z.string(),
    reason: z.string(),
    createdAt: z.string().datetime(),
  })
  .strict();
export const correctionMileageSchema = z
  .object({
    origin: z.string().max(200),
    destination: z.string().max(200),
    distanceMeters: z.number().int().min(0).max(10_000_000),
    method: z.literal("manual"),
    rate: correctionMileageRateSchema.nullable(),
    calculatedAmount: z.number().int().nonnegative(),
    rounding: z.literal("half-up-cent"),
    calculatedAt: z.string().datetime(),
  })
  .strict();
const categoryCheck = z
  .object({
    category: z.enum(categories),
    amount: z.number().int().nonnegative(),
    limit: z.number().int().nonnegative(),
    excess: z.number().int().nonnegative(),
    version: z.number().int().positive(),
    enforcement: z.enum(["advisory", "exception", "block"]).optional(),
  })
  .strict();
const policy = z
  .object({
    limit: z.number().int().nonnegative(),
    excess: z.number().int().nonnegative(),
    version: z.number().int().positive(),
    enforcement: z.enum(["advisory", "exception", "block"]).optional(),
    categoryChecks: z.array(categoryCheck).optional(),
  })
  .strict();
const snapshotContract = expenseFieldsSchema
  .extend({
    id: expenseIdSchema,
    owner: z.string().regex(/^[a-zA-Z0-9-]{1,80}$/),
    employee: z.string(),
    status: z.enum([
      "draft",
      "needs_info",
      "submitted",
      "approved",
      "rejected",
      "cancelled",
    ]),
    version: z.number().int().positive(),
    mileage: correctionMileageSchema.nullable(),
    policy,
    documents: z
      .array(
        z
          .object({
            id: z.string(),
            name: z.string(),
            mime: z.string(),
            size: z.number().int().nonnegative(),
            sha256: z.string().min(1).max(100),
            demo: z.boolean(),
            fiscalKey: z.string().nullable(),
          })
          .strict()
          .refine(
            (d) =>
              (/^[a-f0-9]{64}$/.test(d.sha256) && d.size > 0) ||
              (d.id === "demo-receipt" &&
                d.demo &&
                d.sha256 === "synthetic-example" &&
                d.size === 0),
          ),
      )
      .max(5),
  })
  .strict();
export type CorrectionSnapshot = z.infer<typeof snapshotContract>;
// Validate without applying input trimming to already-recorded before/after evidence.
export const correctionSnapshotSchema = z.custom<CorrectionSnapshot>(
  (value) => snapshotContract.safeParse(value).success,
);
// Preserve rich versioned approval evidence; inspect decision core without stripping older optional facts.
const approvalCore = z
  .object({
    id: z.string().uuid(),
    subject: z
      .object({
        id: z.string(),
        kind: z.literal("expense"),
        owner: z.string(),
        fingerprint: z.string().regex(/^[a-f0-9]{64}$/),
      })
      .passthrough(),
    state: z.enum([
      "pending",
      "approved",
      "rejected",
      "needs_info",
      "cancelled",
    ]),
    steps: z.array(
      z
        .object({
          approver: z.string(),
          state: z.enum(["waiting", "approved", "rejected", "needs_info"]),
        })
        .passthrough(),
    ),
    createdAt: z.string().datetime(),
  })
  .passthrough();
const approval = z
  .custom<ApprovalRun>((v) => approvalCore.safeParse(v).success)
  .nullable();
export const expenseCorrectionRecordSchema = z
  .object({
    id: z.string().uuid(),
    expenseId: expenseIdSchema,
    requestKey: z.string().uuid(),
    reason: z.string().trim().min(10).max(1500),
    at: z.string().datetime(),
    editor: correctionIdentitySchema.refine((v) => v.role !== "employee"),
    grant: correctionGrantSchema,
    before: correctionSnapshotSchema,
    after: correctionSnapshotSchema,
    beforeApproval: approval,
    afterApproval: approval,
    scope: z.literal("audited-delegated-expense-correction"),
  })
  .strict()
  .refine(
    (v) =>
      v.expenseId === v.before.id &&
      v.expenseId === v.after.id &&
      v.before.owner === v.after.owner &&
      v.before.employee === v.after.employee &&
      v.before.version + 1 === v.after.version &&
      v.grant.editorId === v.editor.memberId &&
      v.grant.editor.subjectId === v.editor.subjectId &&
      v.grant.companyId === v.editor.companyId &&
      [v.beforeApproval, v.afterApproval].every(
        (r) =>
          !r ||
          (r.subject.id === v.expenseId && r.subject.owner === v.before.owner),
      ) &&
      JSON.stringify(v.before.documents) === JSON.stringify(v.after.documents),
  );
export type ExpenseCorrectionRecord = z.infer<
  typeof expenseCorrectionRecordSchema
>;
export const expenseCorrectionPreviewReplySchema = z
  .object({
    record: expenseCorrectionRecordSchema,
    input: expenseCorrectionPreviewSchema,
    scope: z.literal("unsaved-expense-correction-preview"),
    saved: z.literal(false),
  })
  .strict();
export const correctionContextSchema = z
  .object({
    expense: correctionSnapshotSchema,
    grant: correctionGrantSchema,
    history: z.array(expenseCorrectionRecordSchema),
    approval,
    issues: z.array(z.string().min(1)),
    contextToken: z.string().regex(/^[a-f0-9]{64}$/),
    scope: z.literal("scoped-expense-correction-context"),
  })
  .strict()
  .refine(
    (v) =>
      v.history.every((r) => r.expenseId === v.expense.id) &&
      (!v.approval || v.approval.subject.id === v.expense.id) &&
      v.history.every((r) => r.editor.companyId === v.grant.companyId),
  );
export type CorrectionContext = z.infer<typeof correctionContextSchema>;
export const correctionListSchema = z
  .object({
    grant: correctionGrantSchema,
    items: z
      .array(
        z
          .object({
            expense: correctionSnapshotSchema,
            issues: z.array(z.string().min(1)),
          })
          .strict(),
      )
      .max(50),
    nextCursor: expenseIdSchema.nullable(),
    scope: z.literal("scoped-expense-corrections"),
  })
  .strict()
  .refine(
    (v) => new Set(v.items.map((r) => r.expense.id)).size === v.items.length,
  );

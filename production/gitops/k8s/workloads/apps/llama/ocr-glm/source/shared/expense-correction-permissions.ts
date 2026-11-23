import { z } from "zod";
import { directoryId, companySchemas } from "./company.js";
import type { ActorContext } from "./actor.js";
import type { ExpenseFields } from "./types.js";

export const correctionCapabilities = [
  "details",
  "classification",
  "amount",
] as const;
export type CorrectionCapability = (typeof correctionCapabilities)[number];
export const correctionScopeSchema = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("company") }).strict(),
  z.object({ kind: z.literal("member"), memberId: directoryId }).strict(),
  z.object({ kind: z.literal("team"), teamId: directoryId }).strict(),
  z.object({ kind: z.literal("center"), centerId: directoryId }).strict(),
]);
export type CorrectionScope = z.infer<typeof correctionScopeSchema>;
export const permissionDay = (now: Date) =>
  now.toLocaleDateString("en-CA", { timeZone: "America/Sao_Paulo" });
const date = z
  .string()
  .regex(/^\d{4}-\d{2}-\d{2}$/)
  .refine((v) => {
    const d = new Date(v + "T12:00:00Z");
    return Number.isFinite(+d) && d.toISOString().slice(0, 10) === v;
  });
export const correctionGrantFieldsSchema = z
  .object({
    editorId: directoryId,
    scope: correctionScopeSchema,
    capabilities: z
      .array(z.enum(correctionCapabilities))
      .min(1)
      .max(3)
      .refine((v) => new Set(v).size === v.length),
    expiresOn: date.nullable(),
  })
  .strict();
export type CorrectionGrantFields = z.infer<typeof correctionGrantFieldsSchema>;
const command = {
  reason: z.string().trim().min(10).max(1500),
  requestKey: z.string().uuid(),
  contextToken: z.string().regex(/^[a-f0-9]{64}$/),
};
export const correctionGrantCommandSchema = z
  .object({
    ...command,
    version: z.number().int().nonnegative(),
    fields: correctionGrantFieldsSchema,
  })
  .strict();
export const correctionGrantRevokeSchema = z
  .object({
    ...command,
    version: z.number().int().positive(),
  })
  .strict();
export type CorrectionGrantCommand = z.infer<
  typeof correctionGrantCommandSchema
>;
export type CorrectionGrantRevoke = z.infer<typeof correctionGrantRevokeSchema>;
export interface CorrectionGrant extends CorrectionGrantFields {
  id: string;
  version: number;
  active: boolean;
  companyId: string;
  editor: ActorContext;
  editorVersion: number;
  scopeLabel: string;
  scopeSubject: ActorContext | null;
  createdAt: string;
  updatedAt: string;
  administeredBy: ActorContext;
  reason: string;
}
export interface CorrectionGrantEvent {
  sequence: number;
  requestKey: string;
  grantId: string;
  action: "grant" | "revise" | "revoke";
  actor: ActorContext;
  at: string;
  reason: string;
  before: CorrectionGrant | null;
  after: CorrectionGrant;
}
export interface CorrectionGrantReceipt {
  action: CorrectionGrantEvent["action"];
  requestKey: string;
  grant: CorrectionGrant;
  event: CorrectionGrantEvent;
  scope: "local-expense-correction-permissions";
}
const actorSchema = z
  .object({
    companyId: z.string().uuid(),
    subjectId: z.string().uuid(),
    memberId: directoryId,
    role: z.enum(["employee", "manager", "finance"]),
    displayName: z.string().min(2).max(100),
    authentication: z.enum(["demo", "password"]),
  })
  .strict();
export { actorSchema as correctionIdentitySchema };
export const correctionGrantSchema = correctionGrantFieldsSchema
  .extend({
    id: z.string().uuid(),
    version: z.number().int().positive(),
    active: z.boolean(),
    companyId: z.string().uuid(),
    editor: actorSchema.extend({ role: z.enum(["manager", "finance"]) }),
    editorVersion: z.number().int().nonnegative(),
    scopeLabel: z.string().min(2).max(100),
    scopeSubject: actorSchema.nullable(),
    createdAt: z.string().datetime(),
    updatedAt: z.string().datetime(),
    administeredBy: actorSchema.extend({ role: z.literal("manager") }),
    reason: z.string().trim().min(10).max(1500),
  })
  .strict()
  .refine(
    (v) =>
      v.editorId === v.editor.memberId &&
      v.editor.companyId === v.companyId &&
      v.administeredBy.companyId === v.companyId &&
      (v.scope.kind === "member"
        ? v.scopeSubject?.memberId === v.scope.memberId &&
          v.scopeSubject.companyId === v.companyId
        : v.scopeSubject === null),
  );
export const correctionGrantEventSchema = z
  .object({
    sequence: z.number().int().positive(),
    requestKey: z.string().uuid(),
    grantId: z.string().uuid(),
    action: z.enum(["grant", "revise", "revoke"]),
    actor: actorSchema.extend({ role: z.literal("manager") }),
    at: z.string().datetime(),
    reason: z.string().trim().min(10).max(1500),
    before: correctionGrantSchema.nullable(),
    after: correctionGrantSchema,
  })
  .strict()
  .refine(
    (v) =>
      v.grantId === v.after.id &&
      v.actor.memberId === v.after.administeredBy.memberId &&
      v.actor.subjectId === v.after.administeredBy.subjectId &&
      v.actor.companyId === v.after.companyId &&
      v.at === v.after.updatedAt &&
      v.reason === v.after.reason &&
      (v.action === "grant"
        ? v.before === null && v.after.version === 1 && v.after.active
        : !!v.before &&
          v.before.id === v.after.id &&
          v.before.version + 1 === v.after.version &&
          v.before.active &&
          v.after.active === (v.action === "revise")),
  );
export const correctionGrantReceiptSchema = z
  .object({
    action: z.enum(["grant", "revise", "revoke"]),
    requestKey: z.string().uuid(),
    grant: correctionGrantSchema,
    event: correctionGrantEventSchema,
    scope: z.literal("local-expense-correction-permissions"),
  })
  .strict()
  .refine(
    (v) =>
      v.action === v.event.action &&
      v.requestKey === v.event.requestKey &&
      JSON.stringify(v.grant) === JSON.stringify(v.event.after),
  );
const directoryMeta = {
  id: directoryId,
  version: z.number().int().positive(),
  createdAt: z.string().datetime(),
  updatedAt: z.string().datetime(),
  demoPersona: z.enum(["ana", "manager", "finance"]).nullable(),
};
export const correctionPermissionsCatalogSchema = z
  .object({
    viewer: actorSchema.refine((v) => v.role !== "employee"),
    asOf: date,
    grants: z.array(
      z
        .object({
          grant: correctionGrantSchema,
          issue: z.string().min(1).nullable(),
        })
        .strict(),
    ),
    choices: z
      .object({
        editors: z.array(
          z
            .object({
              actor: actorSchema.refine((v) => v.role !== "employee"),
              version: z.number().int().nonnegative(),
            })
            .strict(),
        ),
        owners: z.array(actorSchema),
        memberships: z.array(
          z
            .object({
              memberId: directoryId,
              subjectId: z.string().uuid(),
              teamId: directoryId.nullable(),
              costCenterId: directoryId.nullable(),
              directoryVersion: z.number().int().positive().nullable(),
              directoryActive: z.boolean().nullable(),
              loginActive: z.boolean(),
            })
            .strict(),
        ),
        teams: z.array(companySchemas.teams.extend(directoryMeta).strict()),
        centers: z.array(companySchemas.centers.extend(directoryMeta).strict()),
      })
      .strict()
      .nullable(),
    contextToken: z.string().regex(/^[a-f0-9]{64}$/),
    scope: z.literal("local-expense-correction-permissions"),
  })
  .strict()
  .refine(
    (v) =>
      (v.viewer.role === "manager") === (v.choices !== null) &&
      new Set(v.grants.map((g) => g.grant.id)).size === v.grants.length &&
      v.grants.every(
        (g) =>
          g.grant.companyId === v.viewer.companyId &&
          (v.viewer.role === "manager" ||
            (g.grant.editor.memberId === v.viewer.memberId &&
              g.grant.editor.subjectId === v.viewer.subjectId)) &&
          (g.issue !== null ||
            (g.grant.active &&
              (!g.grant.expiresOn || g.grant.expiresOn >= v.asOf))),
      ),
  );
export type CorrectionPermissionsCatalog = z.infer<
  typeof correctionPermissionsCatalogSchema
>;

/** Inspect normalized ordinary expense fields; this grants no write by itself. */
export function changedCorrectionCapabilities(
  before: ExpenseFields,
  after: ExpenseFields,
): CorrectionCapability[] {
  const changed: CorrectionCapability[] = [];
  if (
    ["merchant", "date", "purpose"].some(
      (k) =>
        before[k as keyof ExpenseFields] !== after[k as keyof ExpenseFields],
    )
  )
    changed.push("details");
  const classification = (v: ExpenseFields) => [
    v.category,
    v.costCenter,
    v.allocations
      ?.map((p) => [
        p.id,
        p.description,
        p.category,
        p.costCenter,
        p.servicePeriod
          ? [
              p.servicePeriod.from,
              p.servicePeriod.to,
              p.servicePeriod.basis,
              p.servicePeriod.months.map((m) => [m.month, m.amount]),
            ]
          : null,
      ])
      .sort((a, b) => String(a[0]).localeCompare(String(b[0]))) ?? null,
  ];
  if (
    JSON.stringify(classification(before)) !==
    JSON.stringify(classification(after))
  )
    changed.push("classification");
  const amounts = (v: ExpenseFields) => [
    v.amount,
    v.allocations
      ?.map((p) => [p.id, p.amount])
      .sort((a, b) => String(a[0]).localeCompare(String(b[0]))) ?? null,
  ];
  if (JSON.stringify(amounts(before)) !== JSON.stringify(amounts(after)))
    changed.push("amount");
  return changed;
}

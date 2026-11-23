import { z } from "zod";
import { directoryId, directoryKey, type CompanyRecord } from "./company.js";
const date = z
  .string()
  .regex(/^20\d{2}-\d{2}-\d{2}$/)
  .refine((v) => {
    const d = new Date(v + "T12:00:00Z");
    return !Number.isNaN(+d) && d.toISOString().slice(0, 10) === v;
  }, "Informe uma data válida.");
export const approvalConditionsSchema = z
  .object({
    teamId: directoryId.nullable(),
    merchant: z
      .string()
      .trim()
      .min(2)
      .max(160)
      .refine(
        (v) => /[\p{L}\p{N}]/u.test(v) && !/[\p{Cc}\p{Cf}]/u.test(v),
        "Informe um nome legível.",
      )
      .nullable(),
    validFrom: date.nullable(),
    validTo: date.nullable(),
  })
  .strict()
  .refine(
    (v) => !v.validFrom || !v.validTo || v.validFrom <= v.validTo,
    "O fim da vigência deve ser igual ou posterior ao início.",
  );
export type ApprovalConditions = z.infer<typeof approvalConditionsSchema>;
export const emptyApprovalConditions: ApprovalConditions = {
  teamId: null,
  merchant: null,
  validFrom: null,
  validTo: null,
};
export interface ApprovalRoutingContext {
  submissionDay: string;
  merchant: string;
  requester: Pick<CompanyRecord<"members">,"id"|"name"|"version"|"active"|"teamId"> | null;
  team: Pick<CompanyRecord<"teams">,"id"|"name"|"code"|"version"|"active"> | null;
}
export const approvalBusinessDay = (now: Date) =>
  new Intl.DateTimeFormat("en-CA", {
    timeZone: "America/Sao_Paulo",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(now);
export function matchesApprovalConditions(
  conditions: ApprovalConditions | undefined,
  facts: ApprovalRoutingContext,
) {
  if (!conditions) return true;
  return (
    (!conditions.validFrom || facts.submissionDay >= conditions.validFrom) &&
    (!conditions.validTo || facts.submissionDay <= conditions.validTo) &&
    (!conditions.merchant ||
      directoryKey(conditions.merchant) === directoryKey(facts.merchant)) &&
    (!conditions.teamId ||
      (!!facts.requester?.active &&
        !!facts.team?.active &&
        facts.requester.teamId === conditions.teamId &&
        facts.team.id === conditions.teamId))
  );
}

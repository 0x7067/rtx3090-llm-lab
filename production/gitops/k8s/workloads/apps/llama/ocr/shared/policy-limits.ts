import { z } from "zod";
import type { ActorContext } from "./actor.js";
import {
  categories,
  type Policy,
  type PolicyCheck,
  type PolicyEnforcement,
} from "./types.js";

export const policyEnforcements = ["advisory", "exception", "block"] as const;
export const policyLimitFields = {
  limit: z.number().int().min(0).max(100_000_000),
  version: z.number().int().positive(),
};
export const policyLimitCommand = z
  .object({
    ...policyLimitFields,
    enforcement: z.enum(policyEnforcements),
    reason: z.string().trim().min(5).max(1500),
    requestKey: z.string().uuid(),
  })
  .strict();
export const legacyPolicyLimitCommand = z.object(policyLimitFields).strict();
export const policyEnforcementLabel: Record<PolicyEnforcement, string> = {
  advisory: "Aviso sem exceção",
  exception: "Exceção com justificativa",
  block: "Bloqueio sem exceção",
};
export function policyEffects(check: PolicyCheck) {
  const result = { advisory: 0, exception: 0, block: 0 };
  for (const row of check.categoryChecks ?? [check])
    result[row.enforcement ?? "exception"] += row.excess;
  return result;
}
export const policyBlocks = (check: PolicyCheck) =>
  policyEffects(check).block > 0;
export const policyRequiresException = (check: PolicyCheck) =>
  policyEffects(check).exception > 0;
export const policyBehaviorExplanation = (
  mode: PolicyEnforcement = "exception",
) =>
  ({
    advisory:
      "O excesso gera um aviso; não exige exceção nem impede envio ou aprovação.",
    exception:
      "O excesso exige uma decisão humana com justificativa de exceção.",
    block:
      "O excesso bloqueia envio e aprovação; não admite exceção. Corrija os dados ou revise a configuração com motivo.",
  })[mode];
export interface PolicyLimitEvent {
  id: number;
  category: Policy["category"];
  before: Policy;
  after: Policy;
  actor: ActorContext;
  reason: string | null;
  legacyCommand: boolean;
  at: string;
}
export interface PolicyLimitWorkspace {
  policies: Policy[];
  history: PolicyLimitEvent[];
  legacyHistory: {
    id: number;
    category: Policy["category"];
    oldLimit: number;
    newLimit: number;
    at: string;
  }[];
}
export const policyCategory = z.enum(categories);

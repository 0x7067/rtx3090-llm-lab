import { z } from "zod";
import type { ActorContext } from "./actor.js";
import type { Budget } from "./types.js";

export const budgetChangeSchema = z
  .object({
    month: z.string().regex(/^20\d{2}-(0[1-9]|1[0-2])$/),
    costCenter: z.string().trim().min(1).max(100),
    limit: z.number().int().min(0).max(100_000_000),
    ownerId: z.string().regex(/^[a-zA-Z0-9-]{1,80}$/),
    version: z.number().int().min(0),
    reason: z.string().trim().min(5).max(1500),
    requestKey: z.string().uuid(),
  })
  .strict();
export type BudgetChange = z.infer<typeof budgetChangeSchema>;
export type BudgetOwner = Pick<
  ActorContext,
  "memberId" | "subjectId" | "companyId" | "displayName"
>;
export interface BudgetSnapshot extends Budget {
  owner: BudgetOwner | null;
}
export interface BudgetRevision {
  version: number;
  before: BudgetSnapshot | null;
  after: BudgetSnapshot;
  actor: ActorContext;
  reason: string;
  createdAt: string;
}
export interface BudgetPlan {
  current: BudgetSnapshot;
  ownerCurrent: BudgetOwner | null;
  revisions: BudgetRevision[];
}
export interface BudgetRegister {
  plans: BudgetPlan[];
  owners: BudgetOwner[];
}

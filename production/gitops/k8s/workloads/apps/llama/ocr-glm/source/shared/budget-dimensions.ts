import { z } from "zod";
import type { ActorContext } from "./actor.js";
import type { BudgetOwner } from "./budgets.js";
import type { BudgetForecastLine } from "./budget-forecast.js";

const id = z.string().uuid();
export const dimensionMonth = z.string().regex(/^20\d{2}-(0[1-9]|1[0-2])$/);
const reason = z.string().trim().min(5).max(1500);
const version = z.number().int().min(0);
const cents = z.number().int().min(0).max(100_000_000);
const command = { reason, requestKey: id };
export const dimensionKind = z.enum(["department", "project"]);
export type DimensionKind = z.infer<typeof dimensionKind>;
export const budgetDimensionSchema = z
  .object({
    id: id.optional(),
    kind: dimensionKind,
    name: z.string().trim().min(2).max(100),
    code: z
      .string()
      .trim()
      .regex(/^[A-Z0-9][A-Z0-9_-]{1,29}$/),
    active: z.boolean(),
    version,
    ...command,
  })
  .strict();
export const dimensionPlanSchema = z
  .object({
    dimensionId: id,
    dimensionVersion: version,
    month: dimensionMonth,
    limit: cents,
    ownerId: z.string().regex(/^[a-zA-Z0-9-]{1,80}$/),
    version,
    ...command,
  })
  .strict();
export const dimensionAssignmentSchema = z
  .object({
    dimensionId: id,
    dimensionVersion: version,
    planVersion: version,
    month: dimensionMonth,
    sourceKey: z.string().regex(/^[a-f0-9]{64}$/),
    sourceToken: z.string().regex(/^[a-f0-9]{64}$/),
    amount: cents.min(1),
    basis: reason,
    ...command,
  })
  .strict();
export const dimensionReleaseSchema = z
  .object({
    version,
    ...command,
  })
  .strict();
export const dimensionQuerySchema = z
  .object({ month: dimensionMonth })
  .strict();
export type BudgetDimensionChange = z.infer<typeof budgetDimensionSchema>;
export type DimensionPlanChange = z.infer<typeof dimensionPlanSchema>;
export type DimensionAssignmentChange = z.infer<
  typeof dimensionAssignmentSchema
>;
export type DimensionRelease = z.infer<typeof dimensionReleaseSchema>;
export interface BudgetDimension {
  id: string;
  kind: DimensionKind;
  name: string;
  code: string;
  active: boolean;
  version: number;
  createdAt: string;
  updatedAt: string;
}
export interface DimensionPlan {
  dimension: BudgetDimension;
  month: string;
  limit: number;
  owner: BudgetOwner;
  version: number;
}
export interface DimensionSource {
  key: string;
  token: string;
  line: BudgetForecastLine;
}
export interface DimensionAssignment {
  id: string;
  dimensionId: string;
  kind: DimensionKind;
  month: string;
  plan: DimensionPlan;
  source: DimensionSource;
  amount: number;
  basis: string;
  state: "active" | "released";
  version: number;
  createdBy: ActorContext;
  createdAt: string;
  releasedBy?: ActorContext;
  releaseReason?: string;
  releasedAt?: string;
}
export interface DimensionHistory {
  entity: string;
  before: BudgetDimension | DimensionPlan | DimensionAssignment | null;
  after: BudgetDimension | DimensionPlan | DimensionAssignment;
  actor: ActorContext;
  reason: string;
  createdAt: string;
}
export interface DimensionPosition {
  asOf: string;
  month: string;
  dimensions: BudgetDimension[];
  owners: BudgetOwner[];
  history: DimensionHistory[];
  assignments: (DimensionAssignment & { current: boolean; issues: string[] })[];
  sources: (DimensionSource & {
    available: Record<DimensionKind, number>;
    staleAssignments?: {
      id: string;
      kind: DimensionKind;
      month: string;
      dimensionName: string;
      amount: number;
    }[];
  })[];
  plans: {
    current: DimensionPlan;
    dimensionCurrent: BudgetDimension;
    ownerCurrent: BudgetOwner | null;
    baseline: number;
    estimates: number;
    total: number;
    remaining: number;
    staleAmount: number;
  }[];
  unassigned: Record<DimensionKind, { baseline: number; estimates: number }>;
  bankConfirmed: false;
  notice: string;
}

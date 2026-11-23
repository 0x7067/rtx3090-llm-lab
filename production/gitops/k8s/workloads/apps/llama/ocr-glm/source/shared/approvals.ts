import type { ApprovalDecisionAccess, ApprovalReviewer, ApprovalSubstitution } from "./approval-substitutions.js";
import { reviewDays, type StageReview, type StageDeadlineView } from "./approval-deadlines.js";
import { approvalConditionsSchema, type ApprovalRoutingContext } from "./approval-conditions.js";
import type { CompanyRecord } from "./company.js";
import type { ActorContext } from "./actor.js";
import type { ServicePeriod } from "./service-periods.js";
import type { QuoteComparison } from "./purchase-comparison.js";
import type { PurchaseMatchingSnapshot } from "./payable-matching.js";
import type { PurchaseQuote } from "./purchasing.js";
import { z } from "zod";
import { categories, type ExpenseAllocation } from "./types.js";
import { actorDisplayName, actorId, actorRole, type Actor } from "./actor.js";
import { directoryId } from "./company.js";
import type { MileageDeclaration } from "./mileage.js";

export const reviewerIds = ["manager", "finance"] as const;
export const demoMembers = [
  { id: "ana", name: "Ana Costa", role: "Colaboradora" },
  { id: "manager", name: "Marina Costa", role: "Gestora" },
  { id: "finance", name: "Rafael Lima", role: "Financeiro" },
] as const;
export const actorName = (actor: Actor | string) => {
  if (typeof actor === "string") {
    const member = demoMembers.find((m) => m.id === actor);
    return member
      ? `${member.name} · ${member.role.toLocaleLowerCase("pt-BR")}`
      : actor;
  }
  const legacy = demoMembers.find((member) => member.id === actorId(actor));
  if (
    legacy &&
    legacy.name === actorDisplayName(actor) &&
    actorRole(actor) === (legacy.id === "ana" ? "employee" : legacy.id)
  )
    return `${legacy.name} · ${legacy.role.toLocaleLowerCase("pt-BR")}`;
  const role = {
    employee: "colaborador",
    manager: "gestor",
    finance: "financeiro",
  }[actorRole(actor)];
  return `${actorDisplayName(actor)} · ${role}`;
};
export const approvalRuleSchema = z
  .object({
    name: z.string().trim().min(3).max(100),
    target: z.enum(["expense", "purchase", "payable"]),
    enabled: z.boolean(),
    priority: z.number().int().min(1).max(1000),
    minimum: z.number().int().min(0).max(100_000_000),
    maximum: z.number().int().min(0).max(100_000_000).nullable(),
    category: z.enum(categories).nullable(),
    costCenter: z.string().trim().min(1).max(100).nullable(),
    approvers: z.array(directoryId).min(1).max(12),
    conditions: approvalConditionsSchema.optional(),
    reviewWithinDays: reviewDays.optional(),
  })
  .strict()
  .refine(
    (v) => v.maximum === null || v.maximum >= v.minimum,
    "O valor máximo deve ser maior ou igual ao mínimo.",
  )
  .refine(
    (v) => new Set(v.approvers).size === v.approvers.length,
    "Cada etapa precisa de um aprovador distinto.",
  );

export type ApprovalRuleFields = z.infer<typeof approvalRuleSchema>;
export type ApprovalTarget = ApprovalRuleFields["target"];
export interface ApprovalRule extends ApprovalRuleFields {
  conditionTeam?: CompanyRecord<"teams">;
  id: string;
  version: number;
  updatedAt: string;
}
export interface ApprovalSubject {
  correctionPreparers?: ActorContext[];
  /** Candidate route; the purchase remains effective under its existing authority until final application. */
  purchaseAmendment?: { amendmentId: string; purchaseId: string; sourceToken: string };
  documentViews?: { documentId: string; view: import("./document-reading.js").DocumentReadingView }[];
  servicePeriod?: ServicePeriod;
  servicePeriodPreparers?: ActorContext[];
  quoteComparison?: QuoteComparison;
  kind: ApprovalTarget;
  id: string;
  owner: string;
  amount: number;
  category: string;
  costCenter: string;
  fingerprint: string;
  allocations?: ExpenseAllocation[];
  mileage?: MileageDeclaration;
  quoteDocuments?: PurchaseQuote[];
  purchaseMatching?: PurchaseMatchingSnapshot;
}
export interface ApprovalRun {
  reviewWithinDays?: number;
  deadline?: StageDeadlineView;
  decisionAccess?: ApprovalDecisionAccess;
  routingContext?: ApprovalRoutingContext;
  id: string;
  subject: ApprovalSubject;
  rule: ApprovalRule;
  scopeRules?: { category: string; costCenter: string; rule: ApprovalRule }[];
  legacy: boolean;
  state: "pending" | "approved" | "rejected" | "needs_info" | "cancelled";
  steps: {
    review?: StageReview;
    approver: string;
    performedBy?: ApprovalReviewer;
    substitution?: ApprovalSubstitution;
    reviewer?: {
      memberId: string;
      displayName: string;
      role: "manager" | "finance";
    };
    state: "waiting" | "approved" | "rejected" | "needs_info";
    reason?: string;
    at?: string;
  }[];
  createdAt: string;
  endedAt?: string;
  cancellation?: { actor: string; reason: string };
}

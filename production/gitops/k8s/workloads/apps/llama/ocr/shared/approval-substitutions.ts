import { z } from "zod";
import type { ActorContext } from "./actor.js";
import { directoryId } from "./company.js";
import { approvalConditionsSchema } from "./approval-conditions.js";

const date = approvalConditionsSchema.shape.validFrom.unwrap();
const reason = z.string().trim().min(5).max(1500);
export const substitutionProposal = z.object({
  principalId: directoryId,
  substituteId: directoryId,
  target: z.enum(["expense", "purchase", "payable"]),
  validFrom: date,
  validTo: date,
  reason,
  requestKey: z.string().uuid(),
}).strict().refine(v => v.principalId !== v.substituteId, "Escolha duas pessoas distintas.")
  .refine(v => v.validFrom <= v.validTo && Date.parse(v.validTo) - Date.parse(v.validFrom) <= 366 * 86400000,
    "Informe uma vigência ordenada de até 366 dias decorridos.");
export const substitutionRevocation = z.object({
  version: z.number().int().positive(), reason, requestKey: z.string().uuid(),
}).strict();
export interface ApprovalReviewer {
  memberId: string;
  displayName: string;
  role: "manager" | "finance";
}
export interface ApprovalSubstitution {
  id: string;
  version: number;
  target: "expense" | "purchase" | "payable";
  principal: ApprovalReviewer;
  substitute: ApprovalReviewer;
  validFrom: string;
  validTo: string;
  state: "active" | "revoked";
  reason: string;
  createdBy: ActorContext;
  createdAt: string;
  revokedBy?: ActorContext;
  revokedAt?: string;
  revocationReason?: string;
}
export interface SubstitutionHistory {
  before: ApprovalSubstitution | null;
  after: ApprovalSubstitution;
  actor: ActorContext;
  reason: string;
  at: string;
}
export interface SubstitutionView extends ApprovalSubstitution {
  availability: "active" | "future" | "expired" | "revoked" | "unavailable";
  warnings: string[];
  history: SubstitutionHistory[];
}
export interface SubstitutionWorkspace {
  consultedOn: string;
  members: ApprovalReviewer[];
  substitutions: SubstitutionView[];
}
export interface ApprovalDecisionAccess {
  actorId: string;
  canDecide: boolean;
  principalId: string | null;
  substitution: ApprovalSubstitution | null;
  consultedOn: string;
}

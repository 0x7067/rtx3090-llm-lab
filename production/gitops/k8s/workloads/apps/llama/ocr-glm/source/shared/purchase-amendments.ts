import { z } from "zod";
import {
  purchaseFieldsSchema,
  type PurchaseFields,
  type PurchaseRequest,
  type PurchaseQuote,
} from "./purchasing.js";
import type { ActorContext } from "./actor.js";
import type { ApprovalRun } from "./approvals.js";
import type { QuoteComparison } from "./purchase-comparison.js";
import { quoteComparisonFieldsSchema } from "./purchase-comparison.js";
import type { PurchaseAmendmentCompatibility } from "./purchase-amendment-compatibility.js";
import type { PurchaseAuthorityApplication } from "./purchase-authority.js";
const command = {
  purchaseVersion: z.number().int().positive(),
  basisToken: z.string().regex(/^[a-f0-9]{64}$/),
  reason: z.string().trim().min(10).max(1500),
  requestKey: z.string().uuid(),
};
export const purchaseAmendmentCreateSchema = z
  .object({
    ...command,
    fields: purchaseFieldsSchema,
    understood: z.literal(true),
  })
  .strict();
export const purchaseAmendmentActionSchema = z
  .object({
    ...command,
    version: z.number().int().positive(),
    action: z.enum(["edit", "withdraw", "reject", "reopen"]),
    fields: purchaseFieldsSchema.optional(),
    understood: z.literal(true),
  })
  .strict()
  .superRefine((v, ctx) => {
    if ((v.action === "edit") !== !!v.fields)
      ctx.addIssue({
        code: "custom",
        message: "Somente a edição deve informar o novo escopo.",
      });
  });
export type PurchaseAmendmentCreate = z.infer<
  typeof purchaseAmendmentCreateSchema
>;
export type PurchaseAmendmentAction = z.infer<
  typeof purchaseAmendmentActionSchema
>;
export const purchaseCandidateCommandSchema = z
  .object({
    ...command,
    version: z.number().int().positive(),
    action: z.enum([
      "compare",
      "submit",
      "approve",
      "request-info",
      "reject-route",
      "withdraw-route",
      "select-quote",
    ]),
    comparison: quoteComparisonFieldsSchema.nullable().optional(),
    documentId: z.string().uuid().optional(),
    selected: z.boolean().optional(),
    understood: z.literal(true),
  })
  .strict()
  .superRefine((v, ctx) => {
    if ((v.action === "compare") !== (v.comparison !== undefined))
      ctx.addIssue({
        code: "custom",
        message:
          "Somente a preparação deve informar a comparação da alteração.",
      });
    if (
      (v.action === "select-quote") !==
        (v.documentId !== undefined && v.selected !== undefined) ||
      (v.action !== "select-quote" &&
        (v.documentId !== undefined || v.selected !== undefined))
    )
      ctx.addIssue({
        code: "custom",
        message: "Somente a seleção deve informar uma cotação e sua inclusão.",
      });
  });
export const purchaseCandidateQuoteSchema = z
  .object({
    ...command,
    version: z.number().int().positive(),
    understood: z.literal(true),
  })
  .strict();
export const purchaseCandidateQuoteQuerySchema =
  purchaseCandidateQuoteSchema.extend({
    purchaseVersion: z.coerce.number().int().positive(),
    version: z.coerce.number().int().positive(),
    understood: z.literal("true").transform(() => true as const),
  });
export type PurchaseCandidateQuoteInput = z.infer<
  typeof purchaseCandidateQuoteSchema
>;
export type PurchaseCandidateCommand = z.infer<
  typeof purchaseCandidateCommandSchema
>;
export type AmendmentState =
  "proposed" | "withdrawn" | "rejected" | "reopened" | "applied";
export interface PurchaseAmendmentSource {
  fields: PurchaseFields;
  approval: PurchaseRequest["approval"];
  approvalRun: ApprovalRun | null;
  quotes: PurchaseQuote[];
  comparison: QuoteComparison | null;
}
export interface PurchaseAmendment {
  application?: PurchaseAuthorityApplication;
  /** A separate candidate; no effective purchase fields or original comparison are overwritten during review. */
  candidate?: {
    status:
      | "draft"
      | "submitted"
      | "needs_info"
      | "rejected"
      | "approved"
      | "cancelled";
    quotes: PurchaseQuote[];
    comparison?: QuoteComparison;
    comparisonReviewed?: boolean;
  };
  id: string;
  purchaseId: string;
  version: number;
  state: AmendmentState;
  fields: PurchaseFields;
  source: PurchaseAmendmentSource;
  sourceToken: string;
  preparedBy: ActorContext;
  preparedAt: string;
  current: boolean;
  events: {
    id: number;
    actor: ActorContext;
    at: string;
    action: string;
    reason: string;
    before: PurchaseFields | null;
    after: PurchaseFields;
    candidateBefore?: PurchaseAmendment["candidate"] | null;
    candidateAfter?: PurchaseAmendment["candidate"] | null;
    state: AmendmentState;
  }[];
}
export interface AmendmentBlocker {
  kind:
    "order" | "receiving" | "return" | "allocation" | "reservation" | "invoice";
  id: string;
  label: string;
  state: string;
}
export interface PurchaseAmendmentContext {
  candidateQuoteOptions?: { amendmentId: string; quotes: PurchaseQuote[] }[];
  candidateRoutes?: { amendmentId: string; history: ApprovalRun[] }[];
  /** Consulted capacity evidence; final commands recheck it before application. */
  compatibleChanges?: {
    amendmentId: string;
    assessment: PurchaseAmendmentCompatibility;
  }[];
  purchaseVersion: number;
  basisToken: string;
  source: PurchaseAmendmentSource;
  sourceToken: string;
  sourceIssues: string[];
  blockers: AmendmentBlocker[];
  records: PurchaseAmendment[];
  canPropose: boolean;
}
export const amendmentStateLabels: Record<AmendmentState, string> = {
  proposed: "Em conferência",
  withdrawn: "Proposta retirada",
  rejected: "Proposta recusada",
  reopened: "Reaberta para nova aprovação",
  applied: "Alteração aprovada e aplicada",
};

import { z } from "zod";
import type { PricedItem } from "./item-quantities.js";
import type { ActorContext } from "./actor.js";
import { purchaseMatchLinesSchema } from "./payable-matching.js";
import type { PurchaseMatchingSnapshot } from "./payable-matching.js";
import type { PurchaseMatchingContext } from "./payable-matching.js";
import type { BillFields, Vendor } from "./payables.js";
import type { Document } from "./types.js";

const reason = z.string().trim().min(10).max(1500);
const basisToken = z.string().regex(/^[a-f0-9]{64}$/);
export const purchaseTolerancePolicyFieldsSchema = z
  .object({
    enabled: z.boolean(),
    maximumUnitPriceBps: z.number().int().min(0).max(1000),
    maximumAbsoluteGross: z.number().int().min(0).max(1_000_000),
  })
  .strict()
  .refine(
    (v) =>
      !v.enabled || (v.maximumUnitPriceBps > 0 && v.maximumAbsoluteGross > 0),
    "Uma política ativa exige ambos os limites positivos.",
  );
export type PurchaseTolerancePolicyFields = z.infer<
  typeof purchaseTolerancePolicyFieldsSchema
>;
export interface PurchaseTolerancePolicy extends PurchaseTolerancePolicyFields {
  version: number;
  publishedAt: string | null;
  publishedBy: ActorContext | null;
  reason: string | null;
}
export const purchaseTolerancePolicyCommandSchema = z
  .object({
    version: z.number().int().nonnegative(),
    fields: purchaseTolerancePolicyFieldsSchema,
    reason,
    requestKey: z.string().uuid(),
    understood: z.literal(true),
  })
  .strict();
export type PurchaseTolerancePolicyCommand = z.infer<
  typeof purchaseTolerancePolicyCommandSchema
>;
export const purchaseTolerancePrepareSchema = z
  .object({
    billVersion: z.number().int().positive(),
    basisToken,
    lines: purchaseMatchLinesSchema,
    reason,
    requestKey: z.string().uuid(),
    understood: z.literal(true),
  })
  .strict();
export type PurchaseTolerancePrepare = z.infer<
  typeof purchaseTolerancePrepareSchema
>;
export const purchaseToleranceCommandSchema = z
  .object({
    billVersion: z.number().int().positive(),
    basisToken,
    version: z.number().int().positive(),
    action: z.enum([
      "accept",
      "reject",
      "withdraw",
      "request-retraction",
      "retract",
      "reject-retraction",
    ]),
    reason,
    requestKey: z.string().uuid(),
    understood: z.literal(true),
  })
  .strict();
export type PurchaseToleranceCommand = z.infer<
  typeof purchaseToleranceCommandSchema
>;
export interface PurchasePriceAssessment {
  hasPriceVariance: boolean;
  eligible: boolean;
  issues: string[];
  expectedGross: number | null;
  declaredGross: number;
  difference: number | null;
  absoluteItemDifference: number | null;
  items: {
    billItemIndex: number;
    expectedGross: number;
    declaredGross: number;
    difference: number;
  }[];
  rates: {
    billItemIndex: number;
    purchaseItemIndex: number;
    expectedUnitPrice: number;
    declaredUnitPrice: number;
    differenceBpsCeiling: number;
  }[];
}
export interface PurchaseToleranceSource {
  bill: BillFields;
  vendor: Vendor;
  documents: Document[];
  matching: Omit<PurchaseMatchingSnapshot, "priceTolerance">;
}
export interface PurchaseToleranceGrant {
  reviewId: string;
  billId: string;
  sourceToken: string;
  policy: PurchaseTolerancePolicy;
  assessment: PurchasePriceAssessment;
  preparedBy: ActorContext;
  preparedAt: string;
  preparationReason: string;
  acceptedBy: ActorContext;
  acceptedAt: string;
  reason: string;
  sha256: string;
}
export interface PurchaseToleranceRecord {
  id: string;
  billId: string;
  version: number;
  state:
    | "proposed"
    | "accepted"
    | "rejected"
    | "withdrawn"
    | "retraction-review"
    | "retracted";
  source: PurchaseToleranceSource;
  previousFields: BillFields;
  sourceToken: string;
  policy: PurchaseTolerancePolicy;
  assessment: PurchasePriceAssessment;
  preparedBy: ActorContext;
  preparedAt: string;
  reason: string;
  grant?: PurchaseToleranceGrant;
}
export interface PurchaseToleranceEvent {
  id: number;
  recordId: string | null;
  actor: ActorContext;
  at: string;
  action: string;
  reason: string;
  before: PurchaseToleranceRecord | PurchaseTolerancePolicy | null;
  after: PurchaseToleranceRecord | PurchaseTolerancePolicy;
}
export interface PurchaseToleranceContext {
  authorizedItems: PricedItem[] | null;
  billId: string;
  billVersion: number;
  basisToken: string;
  policy: PurchaseTolerancePolicy;
  source: PurchaseToleranceSource | null;
  assessment: PurchasePriceAssessment | null;
  sourceIssues: string[];
  records: (PurchaseToleranceRecord & { current: boolean; issues: string[] })[];
  events: PurchaseToleranceEvent[];
  matching: PurchaseMatchingContext;
}

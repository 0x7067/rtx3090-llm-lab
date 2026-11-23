import type {
  OrderEvidence,
  OrderFulfillmentContext,
} from "./purchase-orders.js";
import { z } from "zod";
import type { PurchaseFields, PurchaseRequest } from "./purchasing.js";
import type {
  ReceivingDocument,
  PurchaseReceiving,
} from "./purchase-receiving.js";
export const purchaseMatchLinesSchema = z
  .array(
    z
      .object({
        billItemIndex: z.number().int().min(0).max(49),
        purchaseItemIndex: z.number().int().min(0).max(49),
        receivingId: z.string().uuid(),
        receivingToken: z.string().regex(/^[a-f0-9]{64}$/),
        quantity: z.number().int().min(1).max(10_000),
      })
      .strict(),
  )
  .min(1)
  .max(200)
  .refine(
    (lines) =>
      new Set(
        lines.map(
          (l) => `${l.billItemIndex}:${l.purchaseItemIndex}:${l.receivingId}`,
        ),
      ).size === lines.length,
    "Não repita a mesma combinação de item e entrega.",
  );
export const purchaseMatchSchema = z
  .object({
    basis: z.string().trim().min(5).max(1500),
    lines: purchaseMatchLinesSchema,
  })
  .strict();
export type PurchaseMatch = z.infer<typeof purchaseMatchSchema>;
export const purchaseMatchCommandSchema = z
  .object({
    lines: purchaseMatchLinesSchema.nullable(),
    version: z.number().int().positive(),
    reason: z.string().trim().min(5).max(1500),
    requestKey: z.string().uuid(),
  })
  .strict();
export type PurchaseMatchCommand = z.infer<typeof purchaseMatchCommandSchema>;
export interface PurchaseMatchingSnapshot {
  order?: OrderEvidence;
  preparedBy: { memberId: string; name: string } | null;
  basis: string;
  lines: PurchaseMatch["lines"];
  invoiceItems: PurchaseFields["items"];
  purchase: {
    id: string;
    owner: string;
    status: string;
    fields: PurchaseFields;
    quoteComparison?: PurchaseRequest["quoteComparison"];
    approval: PurchaseRequest["approval"];
    quotes: [string, string][];
    approvalRun: { id: string; state: string; fingerprint: string } | null;
  } | null;
  receipts: (PurchaseReceiving | null)[];
}
export interface PurchaseMatchingContext {
  orderContext?: OrderFulfillmentContext;
  billVersion: number;
  purchaseId: string | null;
  purchaseTitle: string | null;
  sources: {
    order?: OrderEvidence;
    documents?: ReceivingDocument[];
    receivingId: string;
    receivingToken: string;
    reference: string;
    receivedOn: string;
    preparedByName: string;
    acceptedByName: string;
    purchaseItemIndex: number;
    description: string;
    unitPrice: number;
    accepted: number;
    available: number;
    purchaseAvailable: number;
  }[];
  snapshot: PurchaseMatchingSnapshot | null;
}

import { z } from "zod";
import { quantitySchema, sumQuantities, type ItemUnit } from "./item-quantities.js";
import type { ActorContext } from "./actor.js";
import type { PurchaseReceiving } from "./purchase-receiving.js";

const day = z
  .string()
  .regex(/^20\d{2}-\d{2}-\d{2}$/)
  .refine((v) => {
    const d = new Date(`${v}T12:00:00Z`);
    return Number.isFinite(+d) && d.toISOString().slice(0, 10) === v;
  }, "Informe uma data válida.");
export const purchaseReturnFieldsSchema = z
  .object({
    receivingId: z.string().uuid(),
    returnedOn: day,
    reference: z.string().trim().min(5).max(200),
    lines: z
      .array(
        z
          .object({
            itemIndex: z.number().int().min(0).max(49),
            quantity: quantitySchema,
          })
          .strict(),
      )
      .min(1)
      .max(50),
    understood: z.literal(true),
  })
  .strict()
  .refine(
    (v) => new Set(v.lines.map((l) => l.itemIndex)).size === v.lines.length,
    "Informe cada item uma única vez.",
  );
const command = {
  purchaseVersion: z.number().int().positive(),
  basisToken: z.string().regex(/^[a-f0-9]{64}$/),
  reason: z.string().trim().min(5).max(1500),
  requestKey: z.string().uuid(),
};
export const purchaseReturnCreateSchema = z
  .object({ ...command, fields: purchaseReturnFieldsSchema })
  .strict();
export const purchaseReturnActionSchema = z
  .object({
    ...command,
    version: z.number().int().positive(),
    action: z.enum([
      "edit",
      "withdraw",
      "accept",
      "reject",
      "request-retraction",
      "accept-retraction",
      "reject-retraction",
    ]),
    fields: purchaseReturnFieldsSchema.optional(),
  })
  .strict()
  .superRefine((v, ctx) => {
    if ((v.action === "edit") !== !!v.fields)
      ctx.addIssue({
        code: "custom",
        message: "Somente a edição deve informar campos.",
      });
  });
export type PurchaseReturnFields = z.infer<typeof purchaseReturnFieldsSchema>;
export type PurchaseReturnCreate = z.infer<typeof purchaseReturnCreateSchema>;
export type PurchaseReturnAction = z.infer<typeof purchaseReturnActionSchema>;
export type PurchaseReturnState =
  | "proposed"
  | "accepted"
  | "rejected"
  | "withdrawn"
  | "retraction-review"
  | "retracted";
export interface PurchaseReturn {
  id: string;
  purchaseId: string;
  version: number;
  state: PurchaseReturnState;
  fields: PurchaseReturnFields;
  createdBy: ActorContext;
  createdAt: string;
  source: { receiving: PurchaseReceiving; receivingToken: string };
  history: {
    id: number;
    actor: ActorContext;
    at: string;
    action: string;
    reason: string;
    before: PurchaseReturnFields | null;
    after: PurchaseReturnFields;
    state: PurchaseReturnState;
  }[];
  current: boolean;
}
export const returnReserves = (r: Pick<PurchaseReturn, "state">) =>
  ["proposed", "accepted", "retraction-review"].includes(r.state);
export const returnAccepted = (r: Pick<PurchaseReturn, "state">) =>
  ["accepted", "retraction-review"].includes(r.state);
export function receiptReturnUnits(
  records: PurchaseReturn[] | undefined,
  receivingId: string,
  itemIndex: number,
) {
  let accepted = 0,
    pending = 0;
  for (const r of records ?? []) {
    if (r.fields.receivingId !== receivingId) continue;
    const n =
      r.fields.lines.find((l) => l.itemIndex === itemIndex)?.quantity ?? 0;
    if (returnAccepted(r)) accepted = sumQuantities([accepted, n]);
    else if (r.state === "proposed") pending = sumQuantities([pending, n]);
  }
  return { accepted, pending, reserved: sumQuantities([accepted, pending]) };
}
export interface PurchaseReturnDiscrepancy {
  unit?: ItemUnit;
  receivingId: string;
  itemIndex: number;
  description: string;
  netReceived: number;
  returnPending: number;
  invoiced: number;
  shortfall: number;
}
export interface PurchaseReturnReview {
  purchaseVersion: number;
  basisToken: string;
  records: PurchaseReturn[];
  receipts: {
    receiving: PurchaseReceiving;
    issues: string[];
    balances: {
      itemIndex: number;
      description: string;
      received: number;
      returned: number;
      pending: number;
      available: number;
    }[];
  }[];
  discrepancies: PurchaseReturnDiscrepancy[];
}
export const purchaseReturnLabels: Record<PurchaseReturnState, string> = {
  proposed: "Aguardando conferência",
  accepted: "Devolução aceita localmente",
  rejected: "Recusada",
  withdrawn: "Proposta retirada",
  "retraction-review": "Retificação em análise",
  retracted: "Declaração retirada com revisão",
};

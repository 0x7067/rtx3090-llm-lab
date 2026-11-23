import type { OrderEvidence } from "./purchase-orders.js";
import { z } from "zod";
import {
  quantitySchema,
  sumQuantities,
  subtractQuantities,
} from "./item-quantities.js";
import type { Document } from "./types.js";
import type { PurchaseRequest } from "./purchasing.js";
import { receiptReturnUnits } from "./purchase-returns.js";

const receivingDate = z
  .string()
  .regex(/^20\d{2}-\d{2}-\d{2}$/)
  .refine((v) => {
    const d = new Date(`${v}T12:00:00Z`);
    return Number.isFinite(+d) && d.toISOString().slice(0, 10) === v;
  }, "Informe uma data válida.");
export const receivingFieldsSchema = z
  .object({
    order: z
      .object({
        id: z.string().uuid(),
        token: z.string().regex(/^[a-f0-9]{64}$/),
      })
      .strict()
      .optional(),
    receivedOn: receivingDate,
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
  })
  .strict()
  .refine(
    (v) => new Set(v.lines.map((l) => l.itemIndex)).size === v.lines.length,
    "Informe cada item apenas uma vez.",
  );
const command = {
  version: z.number().int().positive(),
  reason: z.string().trim().min(5).max(1500),
  requestKey: z.string().uuid(),
};
export const receivingCreateSchema = z
  .object({ ...command, fields: receivingFieldsSchema })
  .strict();
export const receivingDecisionSchema = z
  .object({
    ...command,
    action: z.enum(["accept", "reject", "withdraw", "void"]),
  })
  .strict();
export type ReceivingFields = z.infer<typeof receivingFieldsSchema>;
export type ReceivingCreate = z.infer<typeof receivingCreateSchema>;
export type ReceivingDecision = z.infer<typeof receivingDecisionSchema>;
export type ReceivingState =
  "proposed" | "accepted" | "rejected" | "withdrawn" | "void";
export interface ReceivingEvent {
  memberId: string;
  name: string;
  at: string;
  reason: string;
  state: ReceivingState;
}
export interface ReceivingDocument extends Document {
  reason: string;
  attachedAt: string;
  attachedBy: string;
  attachedByName: string;
}
export interface PurchaseReceiving extends Omit<ReceivingFields, "order"> {
  order?: OrderEvidence;
  documents?: ReceivingDocument[];
  id: string;
  state: ReceivingState;
  preparedBy: string;
  preparedByName: string;
  createdAt: string;
  reason: string;
  purchaseVersion: number;
  authorizationFingerprint: string;
  scope: {
    title: string;
    vendor: string;
    costCenter: string;
    authorizedAt: string;
    items: PurchaseRequest["items"];
  };
  events: ReceivingEvent[];
  /** Derived at read time; never an input or a silently refreshed snapshot. */
  current: boolean;
}
export const receivingStateLabels: Record<ReceivingState, string> = {
  proposed: "Aguardando conferência",
  accepted: "Aceito localmente",
  rejected: "Recusado",
  withdrawn: "Retirado pelo declarante",
  void: "Invalidado com motivo",
};
export function receivingQuantities(
  p: Pick<PurchaseRequest, "items" | "receiving" | "returns">,
) {
  return p.items.map((item, itemIndex) => {
    let accepted = 0,
      pending = 0;
    for (const r of p.receiving ?? []) {
      if (!r.current) continue;
      const quantity =
        r.lines.find((l) => l.itemIndex === itemIndex)?.quantity ?? 0;
      if (r.state === "accepted")
        accepted = sumQuantities([
          accepted,
          Math.max(
            0,
            subtractQuantities(
              quantity,
              receiptReturnUnits(p.returns, r.id, itemIndex).accepted,
            ),
          ),
        ]);
      if (r.state === "proposed") pending = sumQuantities([pending, quantity]);
    }
    return {
      itemIndex,
      description: item.description,
      ordered: item.quantity,
      accepted,
      pending,
      outstanding: subtractQuantities(item.quantity, accepted),
      available: subtractQuantities(item.quantity, accepted, pending),
    };
  });
}

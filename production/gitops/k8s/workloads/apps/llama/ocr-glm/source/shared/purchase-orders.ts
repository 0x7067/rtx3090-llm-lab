import { z } from "zod";
import type {
  PurchaseFields,
  PurchaseQuote,
  PurchaseRequest,
} from "./purchasing.js";
import type { ApprovalRun } from "./approvals.js";
const reason = z.string().trim().min(5).max(1500);
const day = z
  .string()
  .regex(/^20\d{2}-\d{2}-\d{2}$/)
  .refine((v) => {
    const d = new Date(`${v}T12:00:00Z`);
    return Number.isFinite(+d) && d.toISOString().slice(0, 10) === v;
  });
export const orderInstructionsSchema = z
  .object({
    shipTo: z.string().trim().min(5).max(500),
    attention: z.string().trim().min(2).max(160),
    requestedDeliveryOn: day.nullable(),
    notes: z.string().trim().max(1500),
  })
  .strict();
export const orderCreateSchema = z
  .object({
    purchaseId: z.string().uuid(),
    purchaseVersion: z.number().int().positive(),
    sourceFingerprint: z.string().regex(/^[a-f0-9]{64}$/),
    replacesId: z.string().uuid().nullable(),
    fields: orderInstructionsSchema,
    reason,
    requestKey: z.string().uuid(),
  })
  .strict();
export const orderCommandSchema = z
  .object({
    action: z.enum(["authorize", "simulate", "cancel", "void"]),
    version: z.number().int().positive(),
    reason,
    requestKey: z.string().uuid(),
  })
  .strict();
export type OrderCreate = z.infer<typeof orderCreateSchema>;
export type OrderCommand = z.infer<typeof orderCommandSchema>;
export type OrderInstructions = z.infer<typeof orderInstructionsSchema>;
export interface OrderSource {
  purchaseId: string;
  number: number;
  owner: string;
  employee: string;
  fields: PurchaseFields;
  amount: number;
  approval: NonNullable<PurchaseRequest["approval"]>;
  approvalRun: ApprovalRun;
  quotes: PurchaseQuote[];
  quoteComparison?: PurchaseRequest["quoteComparison"];
  company: { id: string; name: string; legalName: string; version: number };
}
export interface OrderSourceContext {
  revision?: OrderRevisionScope;
  source: OrderSource;
  purchaseVersion: number;
  fingerprint: string;
  activeOrderId: string | null;
  lastOrderId: string | null;
  hasEarlierReceiving: boolean;
  hasExpenseAllocations: boolean;
}
export interface OrderPerson {
  memberId: string;
  name: string;
}
export type OrderState =
  "prepared" | "authorized" | "simulated" | "cancelled" | "void";
export interface OrderPayload {
  revision?: OrderRevisionScope;
  format: "despesas-local-purchase-order";
  version: 1;
  notice: string;
  source: OrderSource;
  instructions: OrderInstructions;
  preparedBy: OrderPerson;
  preparedAt: string;
  reason: string;
  replacesId: string | null;
  supplierContacted: false;
  providerAcknowledged: false;
  bankConfirmed: false;
}
export interface OrderRevisionScope {
  cumulativeAuthorization: true;
  previousOrderId: string | null;
  notice: string;
  rows: { itemIndex: number; authorized: number; accepted: number; pending: number; remaining: number }[];
}
export interface PurchaseOrder {
  id: string;
  purchaseId: string;
  state: OrderState;
  version: number;
  sha256: string;
  payload: OrderPayload;
  authorizedBy?: OrderPerson;
  events: {
    state: OrderState;
    by: OrderPerson;
    at: string;
    reason: string;
    simulationReference?: string;
  }[];
  current: boolean;
  blockers: string[];
}
export interface OrderOption {
  id: string;
  number: number;
  title: string;
  vendor: string;
  amount: number;
}
export const orderLabels: Record<OrderState, string> = {
  prepared: "Preparado para revisão",
  authorized: "Autorizado para simulação",
  simulated: "Envio simulado · fornecedor não contatado",
  cancelled: "Cancelado localmente",
  void: "Simulação invalidada localmente",
};

/** Immutable evidence of an explicit local simulation, not supplier acknowledgment. */
export interface OrderEvidence {
  authorization: { by: OrderPerson; at: string; reason: string };
  id: string;
  version: number;
  sha256: string;
  payload: OrderPayload;
  authorizedBy: OrderPerson;
  simulation: {
    by: OrderPerson;
    at: string;
    reason: string;
    reference: string;
  };
}
export interface OrderFulfillmentContext {
  required: true;
  order: OrderEvidence | null;
  token: string | null;
  blockers: string[];
}

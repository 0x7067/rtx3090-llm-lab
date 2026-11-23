import type { ExpenseReservationPosition } from "./expense-reservations.js";
import type { OrderFulfillmentContext } from "./purchase-orders.js";
import type { QuoteComparison } from "./purchase-comparison.js";
import { z } from "zod";
import type { PurchaseReceiving } from "./purchase-receiving.js";
import {
  categories,
  type AuditEvent,
  type Status,
  type Document,
} from "./types.js";

export const purchaseFieldsSchema = z
  .object({
    title: z.string().trim().min(3).max(160),
    vendor: z.string().trim().min(2).max(160),
    purpose: z.string().trim().min(5).max(1500),
    month: z.string().regex(/^20\d{2}-(0[1-9]|1[0-2])$/),
    costCenter: z.string().trim().min(1).max(100),
    category: z.enum(categories),
    items: z
      .array(
        z
          .object({
            description: z.string().trim().min(2).max(200),
            quantity: z.number().int().min(1).max(10_000),
            unitPrice: z.number().int().min(1).max(100_000_000),
          })
          .strict(),
      )
      .min(1)
      .max(50),
  })
  .strict()
  .refine(
    (v) =>
      v.items.reduce((n, i) => n + i.quantity * i.unitPrice, 0) <= 100_000_000,
    "Total máximo de R$ 1.000.000,00.",
  );

export type PurchaseFields = z.infer<typeof purchaseFieldsSchema>;
export type PurchaseAction =
  "submit" | "clarify" | "approve" | "reject" | "cancel";
export interface PurchaseAllocation {
  id: string;
  expenseId: string;
  amount: number;
  reason: string;
  state: "active" | "released";
  counts: boolean;
  createdAt: string;
}
export interface PurchaseQuote extends Document {
  reason: string;
  attachedAt: string;
  attachedBy: string;
  attachedByName: string;
}
export interface PurchaseRequest extends PurchaseFields {
  expenseReservations?: ExpenseReservationPosition[];
  orderContext?: OrderFulfillmentContext;
  quoteComparison?: QuoteComparison;
  quotes?: PurchaseQuote[];
  receiving?: PurchaseReceiving[];
  id: string;
  number: number;
  owner: string;
  employee: string;
  status: Status;
  version: number;
  amount: number;
  allocated: number;
  committed: number;
  createdAt: string;
  updatedAt: string;
  approval?: {
    budgetContext?: {
      basis: "expenses-payables-verified-commitments-v1" | "expenses-payables-reservations-v2" | "expenses-payables-service-periods-v3";
      reservedExpenseCommitments?: number;
      reservationsNeedingReview?: number;
      approvedExpenses: number;
      pendingExpenses: number;
      approvedPayables: number;
      pendingPayables: number;
      residualCommitments: number;
      relievedCommitments: number;
      invoicesNeedingReview: number;
    };
    budgetLimit: number | null;
    budgetVersion: number | null;
    availableBefore: number | null;
    requested: number;
    actor: string;
    at: string;
    reason: string;
  };
  allocations: PurchaseAllocation[];
  history: AuditEvent[];
}

/** Missing approval evidence cannot establish the authorization day. */
export function purchaseExpenseDateIssue(
  purchase: Pick<PurchaseRequest, "approval">,
  expenseDate?: string,
): string | null {
  const day = purchaseAuthorizationDay(purchase);
  if (!day)
    return "Esta autorização não tem data verificável. Não é possível vincular uma nova despesa; preserve o histórico e confira a autorização.";
  if (expenseDate && expenseDate < day)
    return "A despesa antecede a autorização da compra. Vincule somente despesas na data da autorização ou posteriores; não altere a data do comprovante para criar o vínculo.";
  return null;
}

/** Final authorization day; never inferred from request creation or display names. */
export function purchaseAuthorizationDay(
  purchase: Pick<PurchaseRequest, "approval">,
): string | null {
  const at = purchase.approval?.at;
  return typeof at === "string" && Number.isFinite(Date.parse(at))
    ? new Date(at).toLocaleDateString("en-CA", {
        timeZone: "America/Sao_Paulo",
      })
    : null;
}

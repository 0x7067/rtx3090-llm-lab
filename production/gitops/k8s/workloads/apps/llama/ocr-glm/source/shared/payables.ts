import type { ActorContext } from "./actor.js";
import { servicePeriodSchema } from "./service-periods.js";
import { z } from "zod";
import { itemSchema, itemsTotal } from "./item-quantities.js";
import { purchaseMatchSchema } from "./payable-matching.js";
import {
  categories,
  type AuditEvent,
  type Document,
  type Status,
} from "./types.js";
import type { ApprovalRun } from "./approvals.js";

export const payableDate = z
  .string()
  .regex(/^20\d{2}-\d{2}-\d{2}$/)
  .refine((value) => {
    const parsed = new Date(value + "T12:00:00Z");
    return (
      !Number.isNaN(+parsed) && parsed.toISOString().slice(0, 10) === value
    );
  }, "Informe uma data válida.");
export const vendorSchema = z
  .object({
    name: z.string().trim().min(2).max(160),
    taxId: z.string().trim().max(40),
    email: z.union([
      z.string().trim().toLowerCase().email().max(200),
      z.literal(""),
    ]),
    active: z.boolean(),
    onHold: z.boolean(),
    notes: z.string().trim().max(1500),
  })
  .strict();
export type VendorFields = z.infer<typeof vendorSchema>;
export interface PayablePurchaseOption {
  id: string;
  number: number;
  title: string;
  vendor: string;
  costCenter: string;
  amount: number;
  available: number;
  authorizedOn?: string | null;
}
export interface Vendor extends VendorFields {
  id: string;
  version: number;
  createdAt: string;
  updatedAt: string;
}
export const billFieldsSchema = z
  .object({
    vendorId: z.string().uuid(),
    invoiceNumber: z.string().trim().min(1).max(80),
    issueDate: payableDate,
    dueDate: payableDate,
    category: z.enum(categories),
    costCenter: z.string().trim().min(1).max(100),
    purpose: z.string().trim().min(5).max(1500),
    amount: z.number().int().positive().max(100_000_000),
    freight: z.number().int().min(0).max(100_000_000),
    discount: z.number().int().min(0).max(100_000_000),
    items: z.array(itemSchema).min(1).max(50),
    servicePeriod: servicePeriodSchema.optional(),
    purchaseId: z.string().uuid().nullable(),
    purchaseMatch: purchaseMatchSchema.optional(),
    receipt: z
      .object({
        date: payableDate,
        reference: z.string().trim().min(5).max(200),
      })
      .strict()
      .nullable(),
  })
  .strict()
  .refine(
    (v) => v.dueDate >= v.issueDate,
    "O vencimento não pode anteceder a emissão.",
  )
  .refine(
    (v) =>
      v.items.some((i) => !itemSchema.safeParse(i).success) ||
      v.amount === itemsTotal(v.items) + v.freight - v.discount,
    "O total deve corresponder aos itens, frete e desconto.",
  )
  .refine(
    (v) =>
      !v.servicePeriod ||
      v.servicePeriod.months.reduce((n, m) => n + m.amount, 0) === v.amount,
    {
      message: "Os valores mensais devem somar exatamente o total da conta.",
      path: ["servicePeriod"],
    },
  )
  .refine(
    (v) => v.purchaseId !== null || (v.receipt === null && !v.purchaseMatch),
    "A declaração de recebimento exige uma compra vinculada.",
  );
export type BillFields = z.infer<typeof billFieldsSchema>;
export type BillAction =
  "submit" | "approve" | "reject" | "request_info" | "cancel" | "reopen";
export interface Bill extends BillFields {
  servicePeriodPreparers?: ActorContext[];
  id: string;
  owner: string;
  version: number;
  status: Status;
  vendorSnapshot: Vendor | null;
  createdAt: string;
  updatedAt: string;
  documents: Document[];
  history: AuditEvent[];
  approvals: ApprovalRun[];
  payment: { batchId: string; state: "reserved" | "simulated" } | null;
}
export interface PayableIssue {
  code:
    | "duplicate_invoice"
    | "duplicate_source"
    | "expense_source"
    | "vendor_changed"
    | "vendor_unavailable"
    | "missing_source"
    | "xml_mismatch"
    | "purchase_mismatch"
    | "purchase_date_unknown"
    | "purchase_invoice_date"
    | "purchase_receipt_date"
    | "purchase_match_missing"
    | "purchase_match_invalid"
    | "purchase_match_stale"
    | "purchase_match_capacity"
    | "purchase_price_variance"
    | "purchase_tolerance_review"
    | "receipt_missing"
    | "approval_changed";
  message: string;
  relatedId?: string;
  documentId?: string;
}
export const batchFieldsSchema = z
  .object({
    date: payableDate,
    reason: z.string().trim().min(5).max(1500),
    bills: z
      .array(
        z
          .object({
            id: z.string().uuid(),
            version: z.number().int().positive(),
          })
          .strict(),
      )
      .min(1)
      .max(50),
  })
  .strict()
  .refine(
    (v) => new Set(v.bills.map((b) => b.id)).size === v.bills.length,
    "Uma conta só pode aparecer uma vez no lote.",
  );
export type BatchFields = z.infer<typeof batchFieldsSchema>;
export interface PaymentBatch {
  id: string;
  mode: "simulation";
  state: "proposed" | "authorized" | "simulated" | "cancelled";
  version: number;
  date: string;
  reason: string;
  total: number;
  createdAt: string;
  updatedAt: string;
  createdBy: string;
  authorizedBy: string | null;
  simulatedReference: string | null;
  bills: {
    commercialCredit?: {
      token: string;
      originalGross: number;
      acceptedCredits: number;
      netObligation: number;
    };
    id: string;
    version: number;
    amount: number;
    vendor: Vendor;
    invoiceNumber: string;
    dueDate: string;
    fingerprint: string;
  }[];
  history: AuditEvent[];
}

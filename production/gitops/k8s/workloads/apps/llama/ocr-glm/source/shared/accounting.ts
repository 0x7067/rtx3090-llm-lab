import { z } from "zod";
import {
  categories,
  type AuditEvent,
  type Document,
  type Category,
  type ExpenseAllocation,
} from "./types.js";
import { directoryId } from "./company.js";
import type { MileageDeclaration } from "./mileage.js";

// Local bookkeeping proposals, not a fiscal ledger or certified ERP integration.
export const accountingVersion = z.number().int().positive();
export const accountingReason = z.string().trim().min(5).max(1500);
export const accountingMonth = z.string().regex(/^\d{4}-(0[1-9]|1[0-2])$/);
export const accountingDate = z
  .string()
  .regex(/^\d{4}-\d{2}-\d{2}$/)
  .refine((v) => {
    const d = new Date(`${v}T12:00:00Z`);
    return Number.isFinite(+d) && d.toISOString().slice(0, 10) === v;
  }, "Data inválida");
export const accountFieldsSchema = z
  .object({
    code: z
      .string()
      .trim()
      .regex(/^[A-Za-z0-9][A-Za-z0-9.-]{0,39}$/),
    name: z.string().trim().min(2).max(140),
    kind: z.enum(["asset", "liability", "equity", "income", "expense"]),
    active: z.boolean(),
  })
  .strict();
export const journalLineSchema = z
  .object({
    accountId: z.string().uuid(),
    side: z.enum(["debit", "credit"]),
    amount: z.number().int().positive().max(100_000_000),
    costCenter: z.string().trim().max(100),
    memo: z.string().trim().max(300),
  })
  .strict();
export const journalFieldsSchema = z
  .object({
    date: accountingDate,
    description: z.string().trim().min(5).max(500),
    reference: z.string().trim().min(1).max(120),
    lines: z.array(journalLineSchema).min(2).max(100),
  })
  .strict()
  .superRefine((journal, ctx) => {
    const debit = journal.lines
      .filter((l) => l.side === "debit")
      .reduce((n, l) => n + l.amount, 0);
    const credit = journal.lines
      .filter((l) => l.side === "credit")
      .reduce((n, l) => n + l.amount, 0);
    if (debit !== credit)
      ctx.addIssue({
        code: "custom",
        path: ["lines"],
        message: "Débitos e créditos precisam ter o mesmo total.",
      });
  });
export type AccountFields = z.infer<typeof accountFieldsSchema>;
export type JournalFields = z.infer<typeof journalFieldsSchema>;
export const accountingSourceId = z
  .string()
  .regex(/^[A-Za-z0-9][A-Za-z0-9_-]{0,99}$/);
export const recognitionSchema = z
  .object({
    kind: z.enum(["expense", "payable"]),
    sourceId: accountingSourceId,
    sourceToken: z.string().regex(/^[a-f0-9]{64}$/),
    date: accountingDate,
    creditAccountId: z.string().uuid(),
    allocations: z
      .array(
        z
          .object({
            accountId: z.string().uuid(),
            amount: z.number().int().positive().max(100_000_000),
            centerId: directoryId.nullable(),
            memo: z.string().trim().max(300),
          })
          .strict(),
      )
      .min(1)
      .max(99),
  })
  .strict();
export const accountingMappingSchema = z
  .object({
    sourceType: z.enum(["employee-expense", "card-receipt", "supplier-bill"]),
    category: z.enum(categories),
    debitAccountId: z.string().uuid(),
    creditAccountId: z.string().uuid(),
    enabled: z.boolean(),
  })
  .strict();
export type RecognitionFields = z.infer<typeof recognitionSchema>;
export interface AccountingSource {
  kind: "expense" | "payable";
  id: string;
  type: z.infer<typeof accountingMappingSchema>["sourceType"];
  label: string;
  date: string;
  category: Category;
  costCenter: string;
  amount: number;
  allocations?: ExpenseAllocation[];
  mileage?: MileageDeclaration;
  delegatedCorrections?: import("./expense-corrections.js").ExpenseCorrectionRecord[];
  documents: Document[];
  token: string;
  claimKeys: string[];
  approvalId: string;
  evidenceMode: "simulation" | "manual-unverified";
  fundingReference: string | null;
  cardFunding?: import("./card-receipts.js").CardFundingComposition;
  bankConfirmed: false;
}
export interface AccountingMapping extends z.infer<
  typeof accountingMappingSchema
> {
  id: string;
  version: number;
  history: AuditEvent[];
}
export interface AccountingAllocationMapping {
  allocationId: string;
  mapping: AccountingMapping | null;
  issue: string | null;
}
export interface LedgerAccount extends AccountFields {
  id: string;
  version: number;
  history: AuditEvent[];
}
export interface AccountingPeriod {
  budgetCreditSnapshot?: import("./payable-credit-budgets.js").CreditBudgetMonthReview | null;
  month: string;
  state: "open" | "locked";
  version: number;
  history: AuditEvent[];
  closeCycle?: number;
  closingSnapshot?: import("./accounting-close.js").ClosingSnapshot;
  balanceSnapshot?: import("./accounting-opening.js").AccountingBalanceSnapshot;
}
export interface Journal extends JournalFields {
  id: string;
  state: "draft" | "posted" | "void" | "reversed";
  version: number;
  createdBy: string;
  createdAt: string;
  postedBy?: string;
  postedAt?: string;
  reversalOf?: string;
  reversedByJournal?: string;
  accounts: Pick<LedgerAccount, "id" | "code" | "name" | "kind">[];
  provenance: "manual-local";
  source?: AccountingSource;
  opening?: import("./accounting-opening.js").OpeningDeclaration;
  payableSettlement?: import("./accounting-payable-settlements.js").PayableSettlementBasis;
  batchSettlement?: import('./accounting-payment-batches.js').PaymentBatchSettlementBasis;
  payableRefund?: import("./accounting-payable-refunds.js").PayableRefundAccountingBasis;
  payableCreditNote?: import("./accounting-payable-credits.js").PayableCreditAccountingBasis;
  settlement?: import("./accounting-settlements.js").ReimbursementSettlementBasis;
  advanceSettlement?: import("./accounting-advances.js").AdvanceAccountingBasis;
  cardRefund?: import("./accounting-card-refunds.js").CardRefundAccountingBasis;
  allocations?: {
    line: number;
    center: { id: string; code: string; name: string; version: number } | null;
  }[];
  history: AuditEvent[];
}

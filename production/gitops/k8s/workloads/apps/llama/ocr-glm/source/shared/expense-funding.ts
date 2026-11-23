import type { ReimbursementInstallmentState } from "./reimbursement-installments.js";
import type { ActorContext } from "./actor.js";
import type { Expense, ExpenseAllocation, Reimbursement } from "./types.js";
export interface ApprovedExpenseFundingSource extends Pick<
  Expense,
  | "id"
  | "owner"
  | "employee"
  | "merchant"
  | "amount"
  | "date"
  | "category"
  | "costCenter"
  | "purpose"
  | "allocations"
  | "documents"
  | "mileage"
> {
  approvalId: string;
  approvalFingerprint: string;
  fingerprint: string;
}
export type ExpenseFundingPart = Omit<
  ExpenseAllocation,
  "id" | "servicePeriod"
> & { allocationId: string };
export interface ReimbursementFundingSnapshot {
  state: Reimbursement["state"];
  date: string;
  reference: string;
  amount: number;
  source: ApprovedExpenseFundingSource;
  parts: ExpenseFundingPart[];
}
export interface ReimbursementFundingRecord {
  fundingVersion: 1;
  amount: number;
  source: ApprovedExpenseFundingSource;
  parts: ExpenseFundingPart[];
  preparedBy: ActorContext;
  preparedAt: string;
  history: {
    action: "schedule" | "report-paid" | "cancel";
    actor: ActorContext;
    at: string;
    reason: string;
    before: ReimbursementFundingSnapshot | null;
    legacyBefore?: Pick<
      Reimbursement,
      "state" | "date" | "reference" | "version" | "updatedAt"
    >;
    after: ReimbursementFundingSnapshot;
  }[];
}
// Legacy reports retain their historical whole-expense meaning. Unknown/corrupt
// explicit amounts conservatively reserve the gross instead of freeing capacity.
export function reimbursementAmount(
  expense: Pick<Expense, "amount" | "reimbursement">,
): number {
  const amount = expense.reimbursement?.amount;
  return typeof amount === "number" &&
    Number.isSafeInteger(amount) &&
    amount > 0
    ? amount
    : expense.amount;
}
export function reimbursementReservation(
  expense: Pick<Expense, "amount" | "reimbursement" | "reimbursementLedger">,
): number {
  if (expense.reimbursementLedger)
    return expense.reimbursementLedger.records
      .filter((r) => installmentReserves(r.state))
      .reduce((n, r) => n + safeInstallmentAmount(r.amount, expense.amount), 0);
  return ["scheduled", "paid"].includes(expense.reimbursement?.state ?? "")
    ? reimbursementAmount(expense)
    : 0;
}

function safeInstallmentAmount(amount: number, gross: number) {
  return Number.isSafeInteger(amount) && amount > 0 ? amount : gross;
}
export function reimbursementPaidAmount(
  expense: Pick<Expense, "amount" | "reimbursement" | "reimbursementLedger">,
): number {
  if (expense.reimbursementLedger)
    return expense.reimbursementLedger.records
      .filter((r) => installmentDeclared(r.state))
      .reduce((n, r) => n + safeInstallmentAmount(r.amount, expense.amount), 0);
  return expense.reimbursement?.state === "paid"
    ? reimbursementAmount(expense)
    : 0;
}
export function reimbursementScheduledAmount(
  expense: Pick<Expense, "amount" | "reimbursement" | "reimbursementLedger">,
): number {
  if (expense.reimbursementLedger)
    return expense.reimbursementLedger.records
      .filter((r) => r.state === "scheduled")
      .reduce((n, r) => n + safeInstallmentAmount(r.amount, expense.amount), 0);
  return expense.reimbursement?.state === "scheduled"
    ? reimbursementAmount(expense)
    : 0;
}
export function reimbursementPartReservation(
  expense: Pick<Expense, "amount" | "reimbursement" | "reimbursementLedger">,
  partId: string,
  partGross: number,
): number {
  if (expense.reimbursementLedger)
    return expense.reimbursementLedger.records
      .filter((r) => installmentReserves(r.state))
      .reduce((n, r) => {
        if (
          !Array.isArray(r.parts) ||
          r.parts.reduce((s, p) => s + p.amount, 0) !== r.amount ||
          r.parts.some((p) => !Number.isSafeInteger(p.amount) || p.amount <= 0)
        )
          return n + partGross;
        return (
          n +
          r.parts
            .filter((p) => p.allocationId === partId)
            .reduce((s, p) => s + p.amount, 0)
        );
      }, 0);
  if (!reimbursementReservation(expense)) return 0;
  const parts = expense.reimbursement?.funding?.parts;
  if (
    !Array.isArray(parts) ||
    !parts.length ||
    parts.some((p) => !Number.isSafeInteger(p.amount) || p.amount <= 0) ||
    parts.reduce((n, p) => n + p.amount, 0) !== reimbursementAmount(expense)
  )
    return partGross;
  return parts
    .filter((p) => p.allocationId === partId)
    .reduce((n, p) => n + p.amount, 0);
}

export const installmentReserves = (state: ReimbursementInstallmentState) =>
  !["cancelled", "retracted"].includes(state);
export const installmentDeclared = (state: ReimbursementInstallmentState) =>
  ["paid", "retraction-review"].includes(state);

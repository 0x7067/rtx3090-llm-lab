import type { ServicePeriod } from "./service-periods.js";
import type { Budget, Expense, Document } from "./types.js";
import type { PurchaseRequest } from "./purchasing.js";
import { directoryKey } from "./company.js";
import { expenseCenterAmount } from "./expense-allocations.js";
import { budgetUsage } from "./analytics.js";

/** Read-only planning facts, never a payment/recognition instruction. */
export interface BudgetPayable {
  id: string;
  version: number;
  invoiceNumber: string;
  vendor: string;
  status: "submitted" | "needs_info" | "approved";
  amount: number;
  costCenter: string;
  issueDate: string;
  dueDate: string;
  month: string;
  monthBasis: "purchase" | "issue";
  purchase: { id: string; number: number; title: string } | null;
  coverageEligible: boolean;
  warnings: string[];
  documents: Pick<Document, "id" | "name" | "sha256">[];
  simulated: boolean;
  servicePeriod?: ServicePeriod;
  budgetCredits?: import("./payable-credit-budgets.js").CreditBudgetEffect;
}

export const payableBudgetGross = (bill: BudgetPayable, month: string) =>
  bill.servicePeriod
    ? (bill.servicePeriod.months.find((m) => m.month === month)?.amount ?? 0)
    : bill.month === month
      ? bill.amount
      : 0;
export const payableBudgetAmount = (bill: BudgetPayable, month: string) =>
  Math.max(
    0,
    payableBudgetGross(bill, month) -
      (bill.budgetCredits?.current
        ? (bill.budgetCredits.applied.find((p) => p.month === month)?.amount ??
          0)
        : 0),
  );

export function budgetConsumption(
  budget: Budget,
  expenses: Expense[],
  purchases: PurchaseRequest[],
  payables: BudgetPayable[],
) {
  const base = budgetUsage(budget, expenses, purchases);
  const bills = payables.filter(
    (b) =>
      (payableBudgetGross(b, budget.month) > 0 || b.month === budget.month) &&
      directoryKey(b.costCenter) === directoryKey(budget.costCenter),
  );
  const approvedPayables = bills
    .filter((b) => b.status === "approved")
    .reduce((n, b) => n + payableBudgetAmount(b, budget.month), 0);
  const pendingPayables = bills
    .filter((b) => b.status !== "approved")
    .reduce((n, b) => n + payableBudgetAmount(b, budget.month), 0);
  const covered = new Map<string, number>();
  for (const p of purchases) {
    if (
      p.status !== "approved" ||
      p.month !== budget.month ||
      directoryKey(p.costCenter) !== directoryKey(budget.costCenter)
    )
      continue;
    let capacity = p.committed;
    // Stable allocation, with approved invoices before pending reservations.
    for (const bill of [...bills]
      .filter(
        (b) =>
          b.purchase?.id === p.id && b.month === p.month && b.coverageEligible,
      )
      .sort(
        (a, b) =>
          Number(b.status === "approved") - Number(a.status === "approved") ||
          a.id.localeCompare(b.id),
      )) {
      const amount = Math.max(0, Math.min(capacity, bill.amount));
      covered.set(bill.id, amount);
      capacity -= amount;
    }
  }
  const coveredCommitments = [...covered.values()].reduce(
    (n, value) => n + value,
    0,
  );
  const reservations = purchases.flatMap((p) =>
    (p.expenseReservations ?? [])
      .filter(
        (r) =>
          r.state === "active" &&
          r.purchase.month === budget.month &&
          directoryKey(r.purchase.costCenter) ===
            directoryKey(budget.costCenter),
      )
      .map((r) => ({ ...r, coveredCommitment: 0 })),
  );
  const reservationCovered = new Map<string, number>();
  for (const p of purchases.filter(
    (p) =>
      p.status === "approved" &&
      p.month === budget.month &&
      directoryKey(p.costCenter) === directoryKey(budget.costCenter),
  )) {
    let capacity = Math.max(
      0,
      p.committed -
        bills
          .filter((b) => b.purchase?.id === p.id)
          .reduce((n, b) => n + (covered.get(b.id) ?? 0), 0),
    );
    for (const r of reservations
      .filter((r) => r.purchaseId === p.id && r.current)
      .sort((a, b) => a.id.localeCompare(b.id))) {
      // Bound even corrupt/legacy claims to the expense gross and center residual.
      const e = expenses.find((e) => e.id === r.expenseId);
      if (!e || !["submitted", "approved"].includes(e.status)) continue;
      const allocated = purchases
        .flatMap((p) =>
          (p.allocations ?? []).map((a) => ({ ...a, center: p.costCenter })),
        )
        .filter((a) => a.state === "active" && a.expenseId === e.id);
      const already = reservations
        .filter((other) => other.expenseId === e.id)
        .reduce((n, other) => n + (reservationCovered.get(other.id) ?? 0), 0);
      const centerAlready = reservations
        .filter(
          (other) =>
            other.expenseId === e.id &&
            directoryKey(other.purchase.costCenter) ===
              directoryKey(p.costCenter),
        )
        .reduce((n, other) => n + (reservationCovered.get(other.id) ?? 0), 0);
      const gross =
        e.amount - allocated.reduce((n, a) => n + a.amount, 0) - already;
      const center =
        expenseCenterAmount(e, p.costCenter) -
        allocated
          .filter((a) => directoryKey(a.center) === directoryKey(p.costCenter))
          .reduce((n, a) => n + a.amount, 0) -
        centerAlready;
      const value = Math.max(0, Math.min(capacity, r.amount, gross, center));
      reservationCovered.set(r.id, value);
      capacity -= value;
    }
  }
  const reservedExpenseCommitments = [...reservationCovered.values()].reduce(
    (n, value) => n + value,
    0,
  );
  const committed = Math.max(
    0,
    base.committed - coveredCommitments - reservedExpenseCommitments,
  );
  return {
    budget,
    ...base,
    approvedPayables,
    ...(bills.some((b) => b.budgetCredits)
      ? {
          originalApprovedPayables: bills
            .filter((b) => b.status === "approved")
            .reduce((n, b) => n + payableBudgetGross(b, budget.month), 0),
          appliedPayableBudgetCredits: bills
            .filter((b) => b.status === "approved")
            .reduce(
              (n, b) =>
                n +
                payableBudgetGross(b, budget.month) -
                payableBudgetAmount(b, budget.month),
              0,
            ),
          creditBudgetIssues: bills.filter(
            (b) => b.budgetCredits && !b.budgetCredits.current,
          ).length,
        }
      : {}),
    pendingPayables,
    coveredCommitments,
    reservedExpenseCommitments,
    reservations: reservations.map((r) => ({
      ...r,
      coveredCommitment: reservationCovered.get(r.id) ?? 0,
    })),
    committed,
    remaining:
      budget.limit -
      base.approved -
      base.pending -
      approvedPayables -
      pendingPayables -
      committed,
    bills: bills.map((b) => ({
      ...b,
      budgetAmount: payableBudgetAmount(b, budget.month),
      budgetMonth: budget.month,
      coveredCommitment: covered.get(b.id) ?? 0,
    })),
  };
}
export type BudgetConsumptionReport = ReturnType<typeof budgetConsumption>;

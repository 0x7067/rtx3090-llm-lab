import { expenseBudgetCenterAmount } from "./expense-service-periods.js";
import { policyRequiresException } from "./policy-limits.js";
import type { Budget, Expense } from "./types.js";
import type { PurchaseRequest } from "./purchasing.js";
import { directoryKey } from "./company.js";
import { hasExpenseEvidence } from "./mileage.js";
import { expenseCategoryAmounts } from "./expense-allocations.js";

export function summarize(expenses: Expense[], month: string) {
  const rows = expenses.filter((e) => e.date.startsWith(month));
  const approved = rows.filter((e) => e.status === "approved");
  const pending = rows.filter((e) =>
    ["submitted", "needs_info"].includes(e.status),
  );
  const sum = (items: Expense[]) =>
    items.reduce((total, e) => total + e.amount, 0);
  const byCategory = new Map<string, { amount: number; count: number }>();
  for (const e of approved) {
    for (const { category, amount } of expenseCategoryAmounts(e)) {
      const group = byCategory.get(category) ?? { amount: 0, count: 0 };
      group.amount += amount;
      group.count++;
      byCategory.set(category, group);
    }
  }
  return {
    rows,
    approved: sum(approved),
    pending: sum(pending),
    paid: sum(approved.filter((e) => e.reimbursement?.state === "paid")),
    unpaid: sum(approved.filter((e) => e.reimbursement?.state !== "paid")),
    exceptions: approved.filter((e) => policyRequiresException(e.policy)).length,
    missing: pending.filter((e) => !hasExpenseEvidence(e)).length,
    categories: [...byCategory]
      .map(([name, group]) => ({ name, ...group }))
      .sort((a, b) => b.amount - a.amount),
  };
}
export function budgetUsage(
  budget: Budget,
  expenses: Expense[],
  purchases: PurchaseRequest[] = [],
) {
  const rows = expenses;
  const approved = rows
    .filter((e) => e.status === "approved")
    .reduce(
      (n, e) =>
        n + expenseBudgetCenterAmount(e, budget.month, budget.costCenter),
      0,
    );
  const pending = rows
    .filter((e) => ["submitted", "needs_info"].includes(e.status))
    .reduce(
      (n, e) =>
        n + expenseBudgetCenterAmount(e, budget.month, budget.costCenter),
      0,
    );
  const committed = purchases
    .filter(
      (p) =>
        p.month === budget.month &&
        directoryKey(p.costCenter) === directoryKey(budget.costCenter),
    )
    .reduce((n, p) => n + p.committed, 0);
  return {
    approved,
    pending,
    committed,
    remaining: budget.limit - approved - pending - committed,
  };
}

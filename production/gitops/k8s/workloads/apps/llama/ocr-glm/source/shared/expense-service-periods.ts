import type { ExpenseFields } from "./types.js";
import { directoryKey } from "./company.js";
import { effectiveExpenseAllocations } from "./expense-allocations.js";

// Planning attribution only. Preserve actual expense date and all original gross claims.
export function expenseBudgetMonths(expense: ExpenseFields): string[] {
  return [
    ...new Set(
      effectiveExpenseAllocations(expense).flatMap((row) =>
        row.servicePeriod
          ? row.servicePeriod.months.map((m) => m.month)
          : [expense.date.slice(0, 7)],
      ),
    ),
  ].sort();
}
export function expenseBudgetCenterAmount(
  expense: ExpenseFields,
  month: string,
  center: string,
): number {
  const sum = effectiveExpenseAllocations(expense)
    .filter((row) => directoryKey(row.costCenter) === directoryKey(center))
    .reduce(
      (n, row) =>
        n +
        (row.servicePeriod
          ? (row.servicePeriod.months.find((m) => m.month === month)?.amount ??
            0)
          : expense.date.startsWith(month)
            ? row.amount
            : 0),
      0,
    );
  if (!Number.isSafeInteger(sum))
    throw new Error("O valor mensal excede o limite seguro de centavos.");
  return sum;
}
export function expenseBudgetBasis(
  expense: ExpenseFields,
  month: string,
  center: string,
): string {
  const rows = effectiveExpenseAllocations(expense).filter(
    (row) => directoryKey(row.costCenter) === directoryKey(center),
  );
  if (!rows.some((row) => row.servicePeriod))
    return "Parcela da despesa por centro e mês da data declarada. Aprovação não significa pagamento.";
  return (
    `Atribuição mensal por parte classificada; total original da despesa ${expense.amount} centavos. ` +
    rows
      .map((row) =>
        row.servicePeriod
          ? `${row.description}: período ${row.servicePeriod.from} a ${row.servicePeriod.to}, ${row.servicePeriod.months.find((m) => m.month === month)?.amount ?? 0} centavos neste mês. Base: ${row.servicePeriod.basis}.`
          : `${row.description}: mês da data original ${expense.date}, sem período de serviço declarado.`,
      )
      .join(" ") +
    " Não é reconhecimento contábil ou pagamento."
  );
}

import type { Expense } from "./types.js";
import { policyEffects,policyRequiresException } from "./policy-limits.js";
import { hasExpenseEvidence, type MileageDeclaration } from "./mileage.js";

export const spendingQuestions = [
  {
    id: "largest-approved",
    label: "Quais são as maiores despesas aprovadas?",
  },
  {
    id: "policy-exceptions",
    label: "Quais exceções de política foram aprovadas?",
  },
  {
    id: "missing-original",
    label: "Quais pendências têm evidência incompleta?",
  },
] as const;
export type SpendingQuestion = (typeof spendingQuestions)[number]["id"];

export interface SpendingCitation {
  expenseId: string;
  version: number;
  merchant: string;
  date: string;
  amount: number;
  status: Expense["status"];
  policy: Expense["policy"] | null;
  mileage?: MileageDeclaration;
}
export interface SpendingAnswer {
  question: SpendingQuestion;
  month: string;
  mode: "local-rule";
  count: number;
  omitted: number;
  citations: SpendingCitation[];
}

// Every displayed answer is computed from current local expense records.
// It does not read original bytes, infer intent, or propose a financial action.
export function answerSpendingQuestion(
  expenses: Expense[],
  month: string,
  question: SpendingQuestion,
): SpendingAnswer {
  if (!/^\d{4}-(0[1-9]|1[0-2])$/.test(month)) throw new Error("Mês inválido.");
  const rows = expenses.filter((expense) => expense.date.startsWith(month));
  let eligible: Expense[];
  if (question === "largest-approved") {
    eligible = rows
      .filter((expense) => expense.status === "approved")
      .sort((a, b) => b.amount - a.amount || a.id.localeCompare(b.id));
  } else if (question === "policy-exceptions") {
    eligible = rows
      .filter(
        (expense) => expense.status === "approved" && policyRequiresException(expense.policy),
      )
      .sort(
        (a, b) => policyEffects(b.policy).exception - policyEffects(a.policy).exception || a.id.localeCompare(b.id),
      );
  } else {
    eligible = rows
      .filter(
        (expense) =>
          ["submitted", "needs_info"].includes(expense.status) &&
          !hasExpenseEvidence(expense),
      )
      .sort((a, b) => a.date.localeCompare(b.date) || a.id.localeCompare(b.id));
  }
  const citations = eligible.slice(0, 5).map((expense) => ({
    expenseId: expense.id,
    version: expense.version,
    merchant: expense.merchant,
    date: expense.date,
    amount: expense.amount,
    status: expense.status,
    ...(expense.mileage ? { mileage: expense.mileage } : {}),
    policy:
      question === "policy-exceptions"
        ? {
            ...expense.policy,
            ...(expense.policy.categoryChecks
              ? {
                  categoryChecks: expense.policy.categoryChecks.map(
                    (check) => ({ ...check }),
                  ),
                }
              : {}),
          }
        : null,
  }));
  return {
    question,
    month,
    mode: "local-rule",
    count: eligible.length,
    omitted: Math.max(eligible.length - citations.length, 0),
    citations,
  };
}
